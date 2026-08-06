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

#include "common/pd_transfer_telemetry.h"

#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <nlohmann/json.hpp>
#include <thread>
#include <vector>

#include "common/metrics.h"
#include "core/framework/config/disagg_pd_config.h"

namespace xllm {

namespace {

constexpr uint64_t kSamplingScale = 1000000;

void shutdown_pd_transfer_telemetry_sink();

class PDTransferTelemetrySink final {
 public:
  void enqueue(PDTransferTelemetryEvent event) {
    const DisaggPDConfig& config = DisaggPDConfig::get_instance();
    const double sample_rate = config.pd_transfer_telemetry_sample_rate();
    if (!(sample_rate > 0.0) || !should_sample(event, sample_rate)) {
      std::lock_guard<std::mutex> lock(mutex_);
      ++sampled_out_events_;
      return;
    }

    const int32_t queue_capacity = std::max(
        int32_t{1}, config.pd_transfer_telemetry_queue_capacity());
    const int32_t batch_size = std::clamp(
        config.pd_transfer_telemetry_batch_size(),
        int32_t{1},
        queue_capacity);

    std::unique_lock<std::mutex> lock(mutex_);
    queue_capacity_ = queue_capacity;
    batch_size_ = batch_size;
    if (!started_) {
      started_ = true;
      worker_ = std::thread(&PDTransferTelemetrySink::run, this);
      std::atexit(shutdown_pd_transfer_telemetry_sink);
    }
    if (stopping_ ||
        queue_.size() >= static_cast<size_t>(queue_capacity_)) {
      ++dropped_events_;
      COUNTER_INC(pd_transfer_telemetry_dropped_total);
      return;
    }

    queue_.emplace_back(std::move(event));
    ++enqueued_events_;
    lock.unlock();
    condition_.notify_one();
  }

  PDTransferTelemetryStats stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    PDTransferTelemetryStats result;
    result.enqueued_events = enqueued_events_;
    result.emitted_events = emitted_events_;
    result.dropped_events = dropped_events_;
    result.sampled_out_events = sampled_out_events_;
    result.pending_events =
        static_cast<uint64_t>(queue_.size()) + in_flight_events_;
    return result;
  }

  void flush() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!started_) {
      return;
    }
    flushed_.wait(lock, [this] {
      return queue_.empty() && in_flight_events_ == 0;
    });
  }

  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!started_ || stopping_) {
        return;
      }
      stopping_ = true;
    }
    condition_.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

 private:
  static bool should_sample(const PDTransferTelemetryEvent& event,
                            double sample_rate) {
    if (sample_rate >= 1.0) {
      return true;
    }
    constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
    constexpr uint64_t kFnvPrime = 1099511628211ULL;
    uint64_t request_hash = kFnvOffsetBasis;
    for (char character : event.request_id) {
      request_hash ^= static_cast<uint8_t>(character);
      request_hash *= kFnvPrime;
    }
    const uint64_t threshold = static_cast<uint64_t>(
        std::clamp(sample_rate, 0.0, 1.0) * kSamplingScale);
    return request_hash % kSamplingScale < threshold;
  }

  void run() {
    while (true) {
      std::vector<PDTransferTelemetryEvent> batch;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] {
          return stopping_ || !queue_.empty();
        });
        if (queue_.empty() && stopping_) {
          return;
        }

        const size_t batch_count =
            std::min(queue_.size(), static_cast<size_t>(batch_size_));
        batch.reserve(batch_count);
        for (size_t index = 0; index < batch_count; ++index) {
          batch.emplace_back(std::move(queue_.front()));
          queue_.pop_front();
        }
        in_flight_events_ += batch.size();
      }

      std::string payload;
      for (const PDTransferTelemetryEvent& event : batch) {
        if (!payload.empty()) {
          payload.push_back('\n');
        }
        payload.append(kPDTransferTelemetryPrefix);
        payload.append(serialize_pd_transfer_telemetry(event));
      }
      LOG(INFO) << payload;

      {
        std::lock_guard<std::mutex> lock(mutex_);
        emitted_events_ += batch.size();
        in_flight_events_ -= batch.size();
        if (queue_.empty() && in_flight_events_ == 0) {
          flushed_.notify_all();
        }
      }
    }
  }

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::condition_variable flushed_;
  std::deque<PDTransferTelemetryEvent> queue_;
  std::thread worker_;
  int32_t queue_capacity_ = 1;
  int32_t batch_size_ = 1;
  uint64_t enqueued_events_ = 0;
  uint64_t emitted_events_ = 0;
  uint64_t dropped_events_ = 0;
  uint64_t sampled_out_events_ = 0;
  uint64_t in_flight_events_ = 0;
  bool started_ = false;
  bool stopping_ = false;
};

PDTransferTelemetrySink& telemetry_sink() {
  static PDTransferTelemetrySink* sink = new PDTransferTelemetrySink();
  return *sink;
}

void shutdown_pd_transfer_telemetry_sink() {
  telemetry_sink().flush();
  telemetry_sink().shutdown();
}

}  // namespace

int64_t pd_transfer_monotonic_time_ns() {
  const std::chrono::steady_clock::time_point now =
      std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             now.time_since_epoch())
      .count();
}

bool pd_transfer_telemetry_enabled() {
  return DisaggPDConfig::get_instance().enable_pd_transfer_telemetry();
}

std::string serialize_pd_transfer_telemetry(
    const PDTransferTelemetryEvent& event) {
  nlohmann::json data;
  data["schema"] = kPDTransferTelemetrySchema;
  data["request_id"] = event.request_id;
  data["attempt_id"] = event.attempt_id;
  data["event"] = event.event;
  data["monotonic_ns"] = event.monotonic_ns;
  data["transfer_backend"] = event.transfer_backend;
  data["transfer_mode"] = event.transfer_mode;
  data["source_rank"] = event.source_rank;
  data["destination_cluster_id"] = event.destination_cluster_id;
  data["destination_addr"] = event.destination_addr;
  data["layer_index"] = event.layer_index;
  data["num_layers"] = event.num_layers;
  data["bytes"] = event.bytes;
  data["result"] = event.result;
  data["cancelled"] = event.cancelled;
  data["cancellation_reason"] = event.cancellation_reason;
  return data.dump();
}

void log_pd_transfer_telemetry(PDTransferTelemetryEvent event) {
  if (!pd_transfer_telemetry_enabled()) {
    return;
  }
  telemetry_sink().enqueue(std::move(event));
}

PDTransferTelemetryStats pd_transfer_telemetry_stats() {
  return telemetry_sink().stats();
}

void flush_pd_transfer_telemetry() {
  telemetry_sink().flush();
}

}  // namespace xllm
