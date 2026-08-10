/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <Mooncake/mooncake-transfer-engine/include/transfer_engine.h>
#include <brpc/channel.h>
#include <brpc/server.h>

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mooncake_transfer_engine.pb.h"
#include "platform/device.h"

namespace xllm {

using namespace mooncake;

class MooncakeTransferEngineService;

struct MooncakeTransferTrace {
  int64_t submit_monotonic_ns = 0;
  int64_t complete_monotonic_ns = 0;
  uint64_t bytes = 0;
  std::string result = "not_submitted";
  bool cancelled = false;
  std::string cancellation_reason;
};

// Singleton core that holds the actual TransferEngine and brpc Server.
// Multiple MooncakeTransferEngine instances share this core.
class MooncakeTransferEngineCore {
 public:
  static MooncakeTransferEngineCore& get_instance() {
    static MooncakeTransferEngineCore instance;
    return instance;
  }

  // Initialize the shared core. Only the first call takes effect.
  bool initialize(uint16_t listen_port, const torch::Device& device);

  TransferEngine* engine() { return engine_.get(); }

  const std::string& addr() const { return addr_; }
  const std::string& host_ip() const { return host_ip_; }

  // Session state is shared across all MooncakeTransferEngine instances.
  bool open_session(const uint64_t cluster_id, const std::string& remote_addr);
  bool close_session(const uint64_t cluster_id, const std::string& remote_addr);
  SegmentHandle get_handle(const std::string& remote_addr);

  // Lazily create and cache the RPC stub for a remote cluster.
  proto::MooncakeTransferEngineService_Stub* get_or_create_stub(
      uint64_t cluster_id);

  bool is_initialized() const { return initialized_; }

 private:
  proto::MooncakeTransferEngineService_Stub* get_or_create_stub_locked(
      uint64_t cluster_id);

  MooncakeTransferEngineCore() = default;
  ~MooncakeTransferEngineCore();
  MooncakeTransferEngineCore(const MooncakeTransferEngineCore&) = delete;
  MooncakeTransferEngineCore& operator=(const MooncakeTransferEngineCore&) =
      delete;

  std::mutex mutex_;
  bool initialized_ = false;

  std::string addr_;
  std::string host_ip_;
  int32_t rpc_port_ = 0;
  uint16_t listen_port_ = 0;

  std::unique_ptr<TransferEngine> engine_;
  brpc::Server server_;
  std::shared_ptr<MooncakeTransferEngineService> service_;

  // Keep a shared session handle so kv cache and weight transfer can reuse it.
  struct SessionInfo {
    SegmentHandle handle = static_cast<SegmentHandle>(-1);
    int32_t ref_count = 0;
    uint64_t generation = 0;
  };
  std::unordered_map<std::string, SessionInfo> handles_;
  uint64_t next_session_generation_ = 1;
  std::unordered_map<uint64_t, proto::MooncakeTransferEngineService_Stub*>
      stub_map_;
  std::condition_variable session_rpc_cv_;
  std::unordered_set<std::string> session_rpc_inflight_;
};

class MooncakeTransferEngine final {
 public:
  enum class MoveOpcode { READ = 0, WRITE = 1 };

  MooncakeTransferEngine(const uint16_t listen_port,
                         const torch::Device& device);
  virtual ~MooncakeTransferEngine() = default;

  std::string initialize();

  bool register_memory(std::vector<void*> addrs,
                       std::vector<size_t> lens,
                       std::vector<uint64_t> buf_bytes);

  bool move_memory_blocks(const std::string& remote_addr,
                          const std::vector<uint64_t>& src_blocks,
                          const std::vector<uint64_t>& dst_blocks,
                          const std::vector<int64_t>& buf_ids,
                          MoveOpcode move_opcode,
                          MooncakeTransferTrace* trace = nullptr);

  bool pull_memory_blocks(const std::string& remote_addr,
                          const std::vector<uint64_t>& src_blocks,
                          const std::vector<uint64_t>& dst_blocks,
                          const std::vector<int64_t>& buf_ids);

  bool push_memory_blocks(const std::string& remote_addr,
                          const std::vector<uint64_t>& src_blocks,
                          const std::vector<uint64_t>& dst_blocks,
                          const std::vector<int64_t>& buf_ids,
                          MooncakeTransferTrace* trace = nullptr);

  uint64_t transfer_bytes_for_blocks(size_t block_count,
                                     const std::vector<int64_t>& buf_ids) const;

  // XTensor mode uses raw offsets in the GlobalXTensor region in buffer[0].
  bool move_memory_by_global_offsets(const std::string& remote_addr,
                                     const std::vector<uint64_t>& src_offsets,
                                     const std::vector<uint64_t>& dst_offsets,
                                     size_t transfer_size,
                                     MoveOpcode move_opcode,
                                     MooncakeTransferTrace* trace = nullptr);

  bool open_session(const uint64_t cluster_id, const std::string& remote_addr);

  bool close_session(const uint64_t cluster_id, const std::string& remote_addr);

  proto::MooncakeTransferEngineService_Stub* create_rpc_channel(
      uint64_t cluster_id);

 private:
  uint16_t listen_port_;
  std::vector<uint64_t> buf_bytes_;
  Device device_;
  MooncakeTransferEngineCore& core_;
};

class MooncakeTransferEngineService
    : public proto::MooncakeTransferEngineService {
 public:
  MooncakeTransferEngineService() = default;

  virtual ~MooncakeTransferEngineService() = default;

  virtual void OpenSession(google::protobuf::RpcController* controller,
                           const proto::SessionInfo* request,
                           proto::Status* response,
                           google::protobuf::Closure* done) override;

  virtual void CloseSession(google::protobuf::RpcController* controller,
                            const proto::SessionInfo* request,
                            proto::Status* response,
                            google::protobuf::Closure* done) override;
};

}  // namespace xllm
