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

#include <chrono>
#include <nlohmann/json.hpp>

#include "core/framework/config/disagg_pd_config.h"

namespace xllm {

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

void log_pd_transfer_telemetry(const PDTransferTelemetryEvent& event) {
  if (!pd_transfer_telemetry_enabled()) {
    return;
  }
  LOG(INFO) << kPDTransferTelemetryPrefix
            << serialize_pd_transfer_telemetry(event);
}

}  // namespace xllm
