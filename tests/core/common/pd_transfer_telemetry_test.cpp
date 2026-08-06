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
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <utility>

#include "core/framework/config/disagg_pd_config.h"

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

TEST(PDTransferTelemetryTest, HonorsDisaggregatedPDConfigSwitch) {
  DisaggPDConfig& config = DisaggPDConfig::get_instance();
  const bool original_value = config.enable_pd_transfer_telemetry();

  config.enable_pd_transfer_telemetry(false);
  EXPECT_FALSE(pd_transfer_telemetry_enabled());
  config.enable_pd_transfer_telemetry(original_value);
}

TEST(PDTransferTelemetryTest, QueuesAndFlushesEventsAsynchronously) {
  DisaggPDConfig& config = DisaggPDConfig::get_instance();
  const bool original_enabled = config.enable_pd_transfer_telemetry();
  const int32_t original_capacity =
      config.pd_transfer_telemetry_queue_capacity();
  const int32_t original_batch_size =
      config.pd_transfer_telemetry_batch_size();
  const double original_sample_rate =
      config.pd_transfer_telemetry_sample_rate();
  config.enable_pd_transfer_telemetry(true)
      .pd_transfer_telemetry_queue_capacity(128)
      .pd_transfer_telemetry_batch_size(8)
      .pd_transfer_telemetry_sample_rate(1.0);

  const PDTransferTelemetryStats before = pd_transfer_telemetry_stats();
  constexpr int32_t kEventCount = 16;
  for (int32_t index = 0; index < kEventCount; ++index) {
    PDTransferTelemetryEvent event;
    event.request_id = "async-request-" + std::to_string(index);
    event.event = "transfer_complete";
    event.monotonic_ns = pd_transfer_monotonic_time_ns();
    event.transfer_backend = "LlmDataDist";
    log_pd_transfer_telemetry(std::move(event));
  }
  flush_pd_transfer_telemetry();

  const PDTransferTelemetryStats after = pd_transfer_telemetry_stats();
  EXPECT_EQ(after.enqueued_events, before.enqueued_events + kEventCount);
  EXPECT_EQ(after.emitted_events, before.emitted_events + kEventCount);
  EXPECT_EQ(after.dropped_events, before.dropped_events);
  EXPECT_EQ(after.pending_events, 0);

  config.enable_pd_transfer_telemetry(original_enabled)
      .pd_transfer_telemetry_queue_capacity(original_capacity)
      .pd_transfer_telemetry_batch_size(original_batch_size)
      .pd_transfer_telemetry_sample_rate(original_sample_rate);
}

TEST(PDTransferTelemetryTest, CountsSampledOutEventsWithoutQueueing) {
  DisaggPDConfig& config = DisaggPDConfig::get_instance();
  const bool original_enabled = config.enable_pd_transfer_telemetry();
  const double original_sample_rate =
      config.pd_transfer_telemetry_sample_rate();
  config.enable_pd_transfer_telemetry(true)
      .pd_transfer_telemetry_sample_rate(0.0);

  const PDTransferTelemetryStats before = pd_transfer_telemetry_stats();
  PDTransferTelemetryEvent event;
  event.request_id = "sampled-out-request";
  log_pd_transfer_telemetry(std::move(event));
  const PDTransferTelemetryStats after = pd_transfer_telemetry_stats();

  EXPECT_EQ(after.enqueued_events, before.enqueued_events);
  EXPECT_EQ(after.sampled_out_events, before.sampled_out_events + 1);

  config.enable_pd_transfer_telemetry(original_enabled)
      .pd_transfer_telemetry_sample_rate(original_sample_rate);
}

TEST(PDTransferTelemetryTest, SamplesAllEventsForARequestConsistently) {
  DisaggPDConfig& config = DisaggPDConfig::get_instance();
  const bool original_enabled = config.enable_pd_transfer_telemetry();
  const double original_sample_rate =
      config.pd_transfer_telemetry_sample_rate();
  config.enable_pd_transfer_telemetry(true)
      .pd_transfer_telemetry_sample_rate(0.5);

  const PDTransferTelemetryStats before = pd_transfer_telemetry_stats();
  constexpr int32_t kEventCount = 8;
  for (int32_t index = 0; index < kEventCount; ++index) {
    PDTransferTelemetryEvent event;
    event.request_id = "stable-sampling-request";
    event.event = "event-" + std::to_string(index);
    log_pd_transfer_telemetry(std::move(event));
  }
  flush_pd_transfer_telemetry();

  const PDTransferTelemetryStats after = pd_transfer_telemetry_stats();
  const uint64_t enqueued_delta =
      after.enqueued_events - before.enqueued_events;
  const uint64_t sampled_out_delta =
      after.sampled_out_events - before.sampled_out_events;
  EXPECT_TRUE((enqueued_delta == kEventCount && sampled_out_delta == 0) ||
              (enqueued_delta == 0 && sampled_out_delta == kEventCount));

  config.enable_pd_transfer_telemetry(original_enabled)
      .pd_transfer_telemetry_sample_rate(original_sample_rate);
}

TEST(PDTransferTelemetryTest, DropsWithoutBlockingWhenQueueIsFull) {
  DisaggPDConfig& config = DisaggPDConfig::get_instance();
  const bool original_enabled = config.enable_pd_transfer_telemetry();
  const int32_t original_capacity =
      config.pd_transfer_telemetry_queue_capacity();
  const int32_t original_batch_size =
      config.pd_transfer_telemetry_batch_size();
  const double original_sample_rate =
      config.pd_transfer_telemetry_sample_rate();
  const int32_t original_minloglevel = FLAGS_minloglevel;
  config.enable_pd_transfer_telemetry(true)
      .pd_transfer_telemetry_queue_capacity(1)
      .pd_transfer_telemetry_batch_size(1)
      .pd_transfer_telemetry_sample_rate(1.0);
  FLAGS_minloglevel = google::GLOG_FATAL;

  const PDTransferTelemetryStats before = pd_transfer_telemetry_stats();
  constexpr int32_t kEventCount = 4096;
  for (int32_t index = 0; index < kEventCount; ++index) {
    PDTransferTelemetryEvent event;
    event.request_id = "queue-pressure-" + std::to_string(index);
    log_pd_transfer_telemetry(std::move(event));
  }
  flush_pd_transfer_telemetry();

  const PDTransferTelemetryStats after = pd_transfer_telemetry_stats();
  const uint64_t enqueued_delta =
      after.enqueued_events - before.enqueued_events;
  const uint64_t dropped_delta =
      after.dropped_events - before.dropped_events;
  EXPECT_EQ(enqueued_delta + dropped_delta, kEventCount);
  EXPECT_GT(dropped_delta, 0);
  EXPECT_EQ(after.pending_events, 0);

  FLAGS_minloglevel = original_minloglevel;
  config.enable_pd_transfer_telemetry(original_enabled)
      .pd_transfer_telemetry_queue_capacity(original_capacity)
      .pd_transfer_telemetry_batch_size(original_batch_size)
      .pd_transfer_telemetry_sample_rate(original_sample_rate);
}

}  // namespace xllm
