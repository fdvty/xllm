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

#include <gtest/gtest.h>

#include <string>

namespace xllm {

TEST(RequestProfileTest, DisabledProfileDoesNotEmitMeasurements) {
  RequestProfile profile(false);
  profile.mark_enqueued();
  profile.mark_batch_selected(8, 1024, "DECODE", 64, 12);
  profile.mark_engine_step(3'000'000, "DECODE");

  const std::string serialized = profile.serialize_completed(
      "request", "x-request", "service", "127.0.0.1", 10.0, "ok");
  EXPECT_NE(serialized.find("\"engine_step_count\":0"), std::string::npos);
  EXPECT_NE(serialized.find("\"batch_selection_count\":0"),
            std::string::npos);
}

TEST(RequestProfileTest, RecordsSchedulerAndEngineDimensions) {
  RequestProfile profile(true);
  profile.mark_enqueued();
  profile.mark_batch_selected(8, 1024, "DECODE", 64, 12);
  profile.mark_batch_selected(4, 512, "PREFILL", 128, 20);
  profile.mark_engine_step(3'000'000, "DECODE");
  profile.mark_engine_step(5'000'000, "PREFILL");

  const std::string serialized = profile.serialize_completed(
      "request", "x-request", "service", "127.0.0.1", 10.0, "ok");
  EXPECT_NE(serialized.find("\"schema\":\"xllm.request_profile.v1\""),
            std::string::npos);
  EXPECT_NE(serialized.find("\"batch_selection_count\":2"),
            std::string::npos);
  EXPECT_NE(serialized.find("\"engine_step_count\":2"), std::string::npos);
  EXPECT_NE(serialized.find("\"engine_prefill_step_count\":1"),
            std::string::npos);
  EXPECT_NE(serialized.find("\"engine_decode_step_count\":1"),
            std::string::npos);
  EXPECT_NE(serialized.find("\"engine_step_total_ms\":8.0"),
            std::string::npos);
  EXPECT_NE(serialized.find("\"engine_step_max_ms\":5.0"),
            std::string::npos);
  EXPECT_NE(serialized.find("\"max_batch_sequences\":8"),
            std::string::npos);
  EXPECT_NE(serialized.find("\"max_batch_tokens\":1024"),
            std::string::npos);
  EXPECT_NE(serialized.find("\"max_kv_cache_tokens\":128"),
            std::string::npos);
  EXPECT_NE(serialized.find("\"max_kv_cache_blocks\":20"),
            std::string::npos);
}

}  // namespace xllm
