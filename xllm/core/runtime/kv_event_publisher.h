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
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <utility>

#include "common/macros.h"

namespace xllm {

class BlockManagerPool;

class KvEventPublisher final {
 public:
  using BlockManagerPoolPtr = const BlockManagerPool*;

  struct Options {
    PROPERTY(std::string, instance_name);
    PROPERTY(std::string, incarnation_id);
    PROPERTY(std::string, public_endpoint);
    PROPERTY(int32_t, port) = 0;
    PROPERTY(int32_t, publish_interval_ms) = 50;
    PROPERTY(int32_t, snapshot_interval_ms) = 60000;
    PROPERTY(BlockManagerPoolPtr, block_manager_pool) = nullptr;
  };

  explicit KvEventPublisher(Options options);
  ~KvEventPublisher();

  bool start();
  void stop();

  const std::string& endpoint() const { return options_.public_endpoint(); }
  nlohmann::json debug_summary() const;

 private:
  DISALLOW_COPY_AND_ASSIGN(KvEventPublisher);

  void notify_start_result(bool bind_ok);
  void publish_loop();

 private:
  Options options_;
  std::atomic_bool exited_{false};
  std::atomic_bool started_{false};
  std::atomic<uint64_t> next_seq_no_{1};
  std::unique_ptr<std::thread> publish_thread_;
  std::mutex start_mutex_;
  std::condition_variable start_cv_;
  bool bind_attempted_ = false;
  bool bind_ok_ = false;
};

}  // namespace xllm
