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

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

namespace xllm {

// Per-request diagnostic data used to explain scheduler/engine latency.
// Collection is opt-in through XLLM_REQUEST_PROFILE=1 and is intentionally
// separate from Prometheus metrics so request IDs never become metric labels.
class RequestProfile final {
 public:
  RequestProfile();
  explicit RequestProfile(bool enabled);

  static bool globally_enabled();

  bool enabled() const { return enabled_; }

  void mark_enqueued();
  void mark_batch_selected(size_t batch_sequences,
                           size_t batch_tokens,
                           const std::string& batch_type,
                           size_t kv_cache_tokens,
                           size_t kv_cache_blocks);
  void mark_engine_step(int64_t duration_ns, const std::string& batch_type);

  std::string serialize_completed(const std::string& request_id,
                                  const std::string& x_request_id,
                                  const std::string& service_request_id,
                                  const std::string& source_xservice_addr,
                                  double total_latency_ms,
                                  const std::string& terminal_status) const;

 private:
  static int64_t now_ns();
  static void update_max(std::atomic<int64_t>* target, int64_t value);
  static void update_max(std::atomic<uint64_t>* target, uint64_t value);
  int64_t elapsed_from_creation_ns(int64_t event_ns) const;

  const bool enabled_;
  const int64_t created_ns_;
  std::atomic<int64_t> enqueued_ns_{0};
  std::atomic<int64_t> first_batch_ns_{0};
  std::atomic<uint64_t> batch_selection_count_{0};
  std::atomic<uint64_t> batch_prefill_selection_count_{0};
  std::atomic<uint64_t> batch_decode_selection_count_{0};
  std::atomic<uint64_t> batch_mixed_selection_count_{0};
  std::atomic<uint64_t> engine_step_count_{0};
  std::atomic<uint64_t> engine_prefill_step_count_{0};
  std::atomic<uint64_t> engine_decode_step_count_{0};
  std::atomic<uint64_t> engine_mixed_step_count_{0};
  std::atomic<int64_t> engine_step_total_ns_{0};
  std::atomic<int64_t> engine_step_max_ns_{0};
  std::atomic<uint64_t> max_batch_sequences_{0};
  std::atomic<uint64_t> max_batch_tokens_{0};
  std::atomic<uint64_t> max_kv_cache_tokens_{0};
  std::atomic<uint64_t> max_kv_cache_blocks_{0};
};

}  // namespace xllm
