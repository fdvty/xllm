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

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

namespace xllm {

TEST(PDTransferTelemetryTest, SerializesStableSchemaAndEscapesRequestId) {
  PDTransferTelemetryEvent event;
  event.request_id = "request-\"quoted\"";
  event.attempt_id = 7;
  event.event = "transfer_complete";
  event.monotonic_ns = 123456789;
  event.transfer_backend = "LlmDataDist";
  event.transfer_mode = "PUSH";
  event.source_rank = 3;
  event.destination_cluster_id = 42;
  event.destination_addr = "127.0.0.1:5000";
  event.layer_index = 39;
  event.num_layers = 40;
  event.bytes = 8192;
  event.result = "completed";

  const nlohmann::json data =
      nlohmann::json::parse(serialize_pd_transfer_telemetry(event));

  EXPECT_EQ(data["schema"], kPDTransferTelemetrySchema);
  EXPECT_EQ(data["request_id"], event.request_id);
  EXPECT_EQ(data["attempt_id"], 7);
  EXPECT_EQ(data["event"], "transfer_complete");
  EXPECT_EQ(data["monotonic_ns"], 123456789);
  EXPECT_EQ(data["transfer_backend"], "LlmDataDist");
  EXPECT_EQ(data["transfer_mode"], "PUSH");
  EXPECT_EQ(data["source_rank"], 3);
  EXPECT_EQ(data["destination_cluster_id"], 42);
  EXPECT_EQ(data["destination_addr"], "127.0.0.1:5000");
  EXPECT_EQ(data["layer_index"], 39);
  EXPECT_EQ(data["num_layers"], 40);
  EXPECT_EQ(data["bytes"], 8192);
  EXPECT_EQ(data["result"], "completed");
  EXPECT_FALSE(data["cancelled"]);
  EXPECT_EQ(data["cancellation_reason"], "");
}

TEST(PDTransferTelemetryTest, MonotonicTimestampDoesNotGoBackwards) {
  const int64_t first = pd_transfer_monotonic_time_ns();
  const int64_t second = pd_transfer_monotonic_time_ns();

  EXPECT_GT(first, 0);
  EXPECT_GE(second, first);
}

}  // namespace xllm
