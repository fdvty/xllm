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

#include <cstdint>
#include <string>

namespace xllm {

inline constexpr char kPDTransferTelemetryPrefix[] = "PD_TRANSFER_TELEMETRY ";
inline constexpr char kPDTransferTelemetrySchema[] = "xllm.pd_transfer.v1";
inline constexpr uint64_t kLegacyPDAttemptId = 0;

struct PDTransferTelemetryEvent {
  std::string request_id;
  uint64_t attempt_id = kLegacyPDAttemptId;
  std::string event;
  int64_t monotonic_ns = 0;
  std::string transfer_backend;
  std::string transfer_mode;
  int32_t source_rank = -1;
  uint64_t destination_cluster_id = 0;
  std::string destination_addr;
  int64_t layer_index = -1;
  int64_t num_layers = 0;
  uint64_t bytes = 0;
  std::string result;
  bool cancelled = false;
  std::string cancellation_reason;
};

struct PDTransferTelemetryStats {
  uint64_t enqueued_events = 0;
  uint64_t emitted_events = 0;
  uint64_t dropped_events = 0;
  uint64_t sampled_out_events = 0;
  uint64_t pending_events = 0;
};

int64_t pd_transfer_monotonic_time_ns();

bool pd_transfer_telemetry_enabled();

std::string serialize_pd_transfer_telemetry(
    const PDTransferTelemetryEvent& event);

void log_pd_transfer_telemetry(PDTransferTelemetryEvent event);

PDTransferTelemetryStats pd_transfer_telemetry_stats();

void flush_pd_transfer_telemetry();

}  // namespace xllm
