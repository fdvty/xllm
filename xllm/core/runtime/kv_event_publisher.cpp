/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "kv_event_publisher.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>
#include <zmq.hpp>

#include <algorithm>
#include <chrono>

#include "common/metrics.h"
#include "framework/block/block_manager_pool.h"
#include "framework/kv_cache/kv_cache_event.h"
#include "xservice.pb.h"

namespace xllm {
namespace {

void fill_proto_cache_event(const KvCacheEvent& event,
                            xllm_service::proto::KvCacheEvent* proto_event) {
  if (event.stored_cache.size()) {
    proto_event->mutable_stored_cache()->Reserve(event.stored_cache.size());
    for (const auto& hash_key : event.stored_cache) {
      proto_event->add_stored_cache(hash_key.data, sizeof(hash_key.data));
    }
  }

  if (event.removed_cache.size()) {
    proto_event->mutable_removed_cache()->Reserve(event.removed_cache.size());
    for (const auto& hash_key : event.removed_cache) {
      proto_event->add_removed_cache(hash_key.data, sizeof(hash_key.data));
    }
  }
}

}  // namespace

KvEventPublisher::KvEventPublisher(Options options)
    : options_(std::move(options)) {}

KvEventPublisher::~KvEventPublisher() { stop(); }

bool KvEventPublisher::start() {
  if (started_.exchange(true)) {
    return true;
  }
  if (options_.block_manager_pool() == nullptr) {
    LOG(ERROR) << "KV event publisher requires block manager pool.";
    started_.store(false);
    return false;
  }
  if (options_.instance_name().empty() || options_.incarnation_id().empty() ||
      options_.public_endpoint().empty() || options_.port() <= 0) {
    LOG(ERROR) << "KV event publisher has invalid options.";
    started_.store(false);
    return false;
  }

  exited_.store(false);
  {
    std::lock_guard<std::mutex> lock(start_mutex_);
    bind_attempted_ = false;
    bind_ok_ = false;
  }
  publish_thread_ =
      std::make_unique<std::thread>(&KvEventPublisher::publish_loop, this);
  {
    std::unique_lock<std::mutex> lock(start_mutex_);
    start_cv_.wait(lock, [this] { return bind_attempted_; });
    if (bind_ok_) {
      return true;
    }
  }

  stop();
  return false;
}

void KvEventPublisher::stop() {
  exited_.store(true);
  if (publish_thread_ && publish_thread_->joinable()) {
    publish_thread_->join();
  }
  publish_thread_.reset();
  started_.store(false);
}

nlohmann::json KvEventPublisher::debug_summary() const {
  nlohmann::json summary;
  summary["started"] = started_.load();
  summary["endpoint"] = options_.public_endpoint();
  summary["next_seq_no"] = next_seq_no_.load();
  summary["publish_interval_ms"] = options_.publish_interval_ms();
  return summary;
}

void KvEventPublisher::notify_start_result(bool bind_ok) {
  {
    std::lock_guard<std::mutex> lock(start_mutex_);
    bind_attempted_ = true;
    bind_ok_ = bind_ok;
  }
  start_cv_.notify_all();
}

void KvEventPublisher::publish_loop() {
  zmq::context_t context(1);
  zmq::socket_t publisher(context, zmq::socket_type::pub);
  const std::string bind_endpoint =
      "tcp://*:" + std::to_string(options_.port());

  try {
    publisher.set(zmq::sockopt::linger, 0);
    publisher.bind(bind_endpoint);
  } catch (const zmq::error_t& e) {
    LOG(ERROR) << "Failed to bind KV event publisher, endpoint: "
               << bind_endpoint << ", error: " << e.what();
    notify_start_result(false);
    return;
  }
  notify_start_result(true);

  LOG(INFO) << "KV event publisher started, bind_endpoint: " << bind_endpoint
            << ", public_endpoint: " << options_.public_endpoint()
            << ", instance: " << options_.instance_name();

  const auto interval = std::chrono::milliseconds(
      std::max<int32_t>(1, options_.publish_interval_ms()));

  while (!exited_.load()) {
    std::this_thread::sleep_for(interval);

    KvCacheEvent event;
    options_.block_manager_pool()->get_merged_kvcache_event(&event);
    if (event.stored_cache.empty() && event.removed_cache.empty()) {
      continue;
    }

    xllm_service::proto::KvCacheEventEnvelope envelope;
    envelope.set_event_type(xllm_service::proto::KV_CACHE_EVENT_DELTA);
    envelope.set_incarnation_id(options_.incarnation_id());
    envelope.set_seq_no(next_seq_no_.fetch_add(1, std::memory_order_relaxed));
    envelope.set_publish_ts_ms(
        static_cast<uint64_t>(absl::ToUnixMillis(absl::Now())));
    fill_proto_cache_event(event, envelope.mutable_cache_event());

    std::string payload;
    if (!envelope.SerializeToString(&payload)) {
      LOG(ERROR) << "Failed to serialize KV cache event envelope, instance: "
                 << options_.instance_name();
      continue;
    }

    try {
      zmq::message_t topic(options_.instance_name().data(),
                           options_.instance_name().size());
      zmq::message_t body(payload.data(), payload.size());
      publisher.send(topic, zmq::send_flags::sndmore);
      publisher.send(body, zmq::send_flags::none);
      COUNTER_INC(kv_event_zmq_publish_total);
    } catch (const zmq::error_t& e) {
      COUNTER_INC(kv_event_zmq_publish_failure_total);
      LOG(ERROR) << "Failed to publish KV cache event, instance: "
                 << options_.instance_name() << ", error: " << e.what();
    }
  }

  publisher.close();
  context.close();
}

}  // namespace xllm
