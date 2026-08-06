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

#include "core/framework/request/request_profile.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <string>

#include "util/env_var.h"

namespace xllm {

RequestProfile::RequestProfile() : RequestProfile(globally_enabled()) {}

RequestProfile::RequestProfile(bool enabled)
    : enabled_(enabled), created_ns_(now_ns()) {}

bool RequestProfile::globally_enabled() {
  static const bool enabled = util::get_bool_env("XLLM_REQUEST_PROFILE", false);
  return enabled;
}

int64_t RequestProfile::now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void RequestProfile::update_max(std::atomic<int64_t>* target, int64_t value) {
  int64_t current = target->load(std::memory_order_relaxed);
  while (current < value &&
         !target->compare_exchange_weak(current,
                                        value,
                                        std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
  }
}

void RequestProfile::update_max(std::atomic<uint64_t>* target, uint64_t value) {
  uint64_t current = target->load(std::memory_order_relaxed);
  while (current < value &&
         !target->compare_exchange_weak(current,
                                        value,
                                        std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
  }
}

int64_t RequestProfile::elapsed_from_creation_ns(int64_t event_ns) const {
  if (event_ns <= 0) {
    return 0;
  }
  return std::max<int64_t>(0, event_ns - created_ns_);
}

void RequestProfile::mark_enqueued() {
  if (!enabled_) {
    return;
  }
  int64_t now = now_ns();
  int64_t expected = 0;
  enqueued_ns_.compare_exchange_strong(expected,
                                       now,
                                       std::memory_order_relaxed,
                                       std::memory_order_relaxed);
}

void RequestProfile::mark_batch_selected(size_t batch_sequences,
                                          size_t batch_tokens,
                                          const std::string& batch_type,
                                          size_t kv_cache_tokens,
                                          size_t kv_cache_blocks) {
  if (!enabled_) {
    return;
  }
  int64_t now = now_ns();
  int64_t expected = 0;
  first_batch_ns_.compare_exchange_strong(expected,
                                          now,
                                          std::memory_order_relaxed,
                                          std::memory_order_relaxed);
  batch_selection_count_.fetch_add(1, std::memory_order_relaxed);
  if (batch_type.find("DECODE") != std::string::npos) {
    batch_decode_selection_count_.fetch_add(1, std::memory_order_relaxed);
  }
  if (batch_type.find("PREFILL") != std::string::npos) {
    batch_prefill_selection_count_.fetch_add(1, std::memory_order_relaxed);
  }
  if (batch_type.find("MIXED") != std::string::npos) {
    batch_mixed_selection_count_.fetch_add(1, std::memory_order_relaxed);
  }
  update_max(&max_batch_sequences_, static_cast<uint64_t>(batch_sequences));
  update_max(&max_batch_tokens_, static_cast<uint64_t>(batch_tokens));
  update_max(&max_kv_cache_tokens_, static_cast<uint64_t>(kv_cache_tokens));
  update_max(&max_kv_cache_blocks_, static_cast<uint64_t>(kv_cache_blocks));
}

void RequestProfile::mark_engine_step(int64_t duration_ns,
                                      const std::string& batch_type) {
  if (!enabled_) {
    return;
  }
  engine_step_count_.fetch_add(1, std::memory_order_relaxed);
  engine_step_total_ns_.fetch_add(std::max<int64_t>(0, duration_ns),
                                  std::memory_order_relaxed);
  update_max(&engine_step_max_ns_, std::max<int64_t>(0, duration_ns));
  const bool is_mixed = batch_type.find("MIXED") != std::string::npos;
  if (batch_type.find("DECODE") != std::string::npos || is_mixed) {
    engine_decode_step_count_.fetch_add(1, std::memory_order_relaxed);
  }
  if (batch_type.find("PREFILL") != std::string::npos || is_mixed) {
    engine_prefill_step_count_.fetch_add(1, std::memory_order_relaxed);
  }
  if (is_mixed) {
    engine_mixed_step_count_.fetch_add(1, std::memory_order_relaxed);
  }
}

std::string RequestProfile::serialize_completed(
    const std::string& request_id,
    const std::string& x_request_id,
    const std::string& service_request_id,
    const std::string& source_xservice_addr,
    double total_latency_ms,
    const std::string& terminal_status) const {
  nlohmann::json data;
  data["schema"] = "xllm.request_profile.v1";
  data["request_id"] = request_id;
  data["x_request_id"] = x_request_id;
  data["service_request_id"] = service_request_id;
  data["source_xservice_addr"] = source_xservice_addr;
  data["terminal_status"] = terminal_status;
  data["total_latency_ms"] = total_latency_ms;
  data["created_to_enqueue_ms"] =
      elapsed_from_creation_ns(enqueued_ns_.load(std::memory_order_relaxed)) /
      1e6;
  data["created_to_first_batch_ms"] =
      elapsed_from_creation_ns(first_batch_ns_.load(std::memory_order_relaxed)) /
      1e6;
  data["batch_selection_count"] =
      batch_selection_count_.load(std::memory_order_relaxed);
  data["batch_prefill_selection_count"] =
      batch_prefill_selection_count_.load(std::memory_order_relaxed);
  data["batch_decode_selection_count"] =
      batch_decode_selection_count_.load(std::memory_order_relaxed);
  data["batch_mixed_selection_count"] =
      batch_mixed_selection_count_.load(std::memory_order_relaxed);
  data["engine_step_count"] =
      engine_step_count_.load(std::memory_order_relaxed);
  data["engine_prefill_step_count"] =
      engine_prefill_step_count_.load(std::memory_order_relaxed);
  data["engine_decode_step_count"] =
      engine_decode_step_count_.load(std::memory_order_relaxed);
  data["engine_mixed_step_count"] =
      engine_mixed_step_count_.load(std::memory_order_relaxed);
  data["engine_step_total_ms"] =
      engine_step_total_ns_.load(std::memory_order_relaxed) / 1e6;
  data["engine_step_max_ms"] =
      engine_step_max_ns_.load(std::memory_order_relaxed) / 1e6;
  data["max_batch_sequences"] =
      max_batch_sequences_.load(std::memory_order_relaxed);
  data["max_batch_tokens"] = max_batch_tokens_.load(std::memory_order_relaxed);
  data["max_kv_cache_tokens"] =
      max_kv_cache_tokens_.load(std::memory_order_relaxed);
  data["max_kv_cache_blocks"] =
      max_kv_cache_blocks_.load(std::memory_order_relaxed);
  return data.dump();
}

}  // namespace xllm
