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

#include "framework/kv_cache_transfer/mooncake_transfer_engine.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <numeric>

#include "common/pd_transfer_telemetry.h"
#include "framework/config/service_config.h"
#include "util/net.h"

namespace xllm {

namespace {

void initialize_transfer_trace(MooncakeTransferTrace* trace) {
  if (trace != nullptr) {
    *trace = MooncakeTransferTrace{};
  }
}

bool fail_transfer_trace(MooncakeTransferTrace* trace,
                         const std::string& result) {
  if (trace != nullptr) {
    trace->complete_monotonic_ns = pd_transfer_monotonic_time_ns();
    trace->result = result;
  }
  return false;
}

constexpr int kDefaultSessionRpcTimeoutMs = 1000;

int session_rpc_timeout_ms() {
  const int configured_timeout_ms =
      ServiceConfig::get_instance().rpc_channel_timeout_ms();
  return configured_timeout_ms > 0 ? configured_timeout_ms
                                   : kDefaultSessionRpcTimeoutMs;
}

bool open_remote_session(proto::MooncakeTransferEngineService_Stub* stub,
                         const std::string& local_addr,
                         const std::string& remote_addr) {
  proto::SessionInfo session_info;
  session_info.set_addr(local_addr);
  proto::Status response;
  brpc::Controller cntl;
  cntl.set_timeout_ms(session_rpc_timeout_ms());
  stub->OpenSession(&cntl, &session_info, &response, nullptr);
  if (cntl.Failed() || !response.ok()) {
    LOG(ERROR) << "OpenSession failed for " << remote_addr << ", "
               << cntl.ErrorText();
    return false;
  }
  return true;
}

bool close_remote_session(proto::MooncakeTransferEngineService_Stub* stub,
                          const std::string& local_addr,
                          const std::string& remote_addr) {
  proto::SessionInfo session_info;
  session_info.set_addr(local_addr);
  proto::Status response;
  brpc::Controller cntl;
  cntl.set_timeout_ms(session_rpc_timeout_ms());
  stub->CloseSession(&cntl, &session_info, &response, nullptr);
  if (cntl.Failed() || !response.ok()) {
    LOG(ERROR) << "CloseSession failed for " << remote_addr << ", "
               << cntl.ErrorText();
    return false;
  }
  return true;
}

bool check_buf_range(uint64_t buf_len,
                     uint64_t buf_bytes,
                     uint64_t block_id,
                     uint64_t block_len,
                     int64_t buf_id) {
  if (buf_bytes == 0) {
    LOG(ERROR) << "buf bytes is zero, buf_id=" << buf_id;
    return false;
  }
  if (buf_len % buf_bytes != 0) {
    LOG(ERROR) << "buf len is not aligned with block bytes, buf_id=" << buf_id
               << ", buf_len=" << buf_len << ", buf_bytes=" << buf_bytes;
    return false;
  }

  uint64_t block_cnt = buf_len / buf_bytes;
  if (block_id > block_cnt || block_len > block_cnt - block_id) {
    LOG(ERROR) << "block range out of bounds, buf_id=" << buf_id
               << ", block_cnt=" << block_cnt << ", block_id=" << block_id
               << ", block_len=" << block_len;
    return false;
  }
  return true;
}

}  // namespace

// ============================================================================
// MooncakeTransferEngineCore (Singleton)
// ============================================================================

MooncakeTransferEngineCore::~MooncakeTransferEngineCore() {
  for (auto& pair : stub_map_) {
    if (pair.second != nullptr) {
      delete pair.second->channel();
      delete pair.second;
    }
  }
  stub_map_.clear();

  if (initialized_) {
    server_.Stop(0);
    server_.Join();
  }
}

bool MooncakeTransferEngineCore::initialize(uint16_t listen_port,
                                            const torch::Device& device) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (initialized_) {
    LOG(INFO) << "MooncakeTransferEngineCore already initialized, reusing";
    return true;
  }

  listen_port_ = listen_port;
  host_ip_ = net::get_local_ip_addr();

  engine_ = std::make_unique<TransferEngine>(true);

  Device dev(device);
  dev.set_device();
  dev.init_device_context();

  std::string hostname = host_ip_ + ":" + std::to_string(listen_port_);

  if (engine_->init("P2PHANDSHAKE", hostname, "", 0)) {
    LOG(ERROR) << "engine init failed, hostname=" << hostname;
    return false;
  }

  LOG(INFO) << "TransferEngine init success, hostname=" << hostname;

  service_ = std::make_shared<MooncakeTransferEngineService>();
  if (server_.AddService(service_.get(), brpc::SERVER_DOESNT_OWN_SERVICE) !=
      0) {
    LOG(ERROR) << "Failed to add service to server";
    return false;
  }

  brpc::ServerOptions options;
  if (server_.Start(listen_port_, &options) != 0) {
    LOG(ERROR) << "Fail to start Brpc rpc server on port " << listen_port_;
    return false;
  }

  rpc_port_ = engine_->getRpcPort();
  addr_ = host_ip_ + ":" + std::to_string(rpc_port_);

  initialized_ = true;
  LOG(INFO) << "MooncakeTransferEngineCore initialize success, addr_=" << addr_;

  return true;
}

bool MooncakeTransferEngineCore::open_session(const uint64_t cluster_id,
                                              const std::string& remote_addr) {
  std::unique_lock<std::mutex> lock(mutex_);

  LOG(INFO) << "open_session, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  if (cluster_id != 0) {
    session_rpc_cv_.wait(lock, [this, &remote_addr]() {
      return !session_rpc_inflight_.contains(remote_addr);
    });
    session_rpc_inflight_.insert(remote_addr);

    const bool result = [this, cluster_id, &remote_addr, &lock]() {
      auto it = handles_.find(remote_addr);
      if (it != handles_.end()) {
        it->second.ref_count++;
        LOG(INFO) << "Reusing existing session for " << remote_addr
                  << ", ref_count=" << it->second.ref_count;
        return true;
      }

      proto::MooncakeTransferEngineService_Stub* stub =
          get_or_create_stub_locked(cluster_id);
      if (stub == nullptr) {
        LOG(ERROR) << "create_rpc_channel failed";
        return false;
      }

      const std::string local_addr = addr_;
      lock.unlock();
      const bool opened = open_remote_session(stub, local_addr, remote_addr);
      lock.lock();
      if (!opened) {
        return false;
      }

      LOG(INFO) << "OpenSession RPC to " << remote_addr
                << ", local_addr=" << local_addr;
#if !defined(USE_DCU)
      return true;
#else
      it = handles_.find(remote_addr);
      if (it != handles_.end()) {
        it->second.ref_count++;
        LOG(INFO) << "Reusing existing session for " << remote_addr
                  << ", ref_count=" << it->second.ref_count;
        return true;
      }

      Transport::SegmentHandle handle = engine_->openSegment(remote_addr);
      if (handle == static_cast<Transport::SegmentHandle>(-1)) {
        LOG(ERROR) << "Fail to connect to " << remote_addr;
        return false;
      }

      SessionInfo session_info;
      session_info.handle = handle;
      session_info.ref_count = 1;
      session_info.generation = next_session_generation_++;
      handles_[remote_addr] = session_info;
      LOG(INFO) << "Created new session for " << remote_addr << ", ref_count=1";
      return true;
#endif
    }();

    session_rpc_inflight_.erase(remote_addr);
    lock.unlock();
    session_rpc_cv_.notify_all();
    return result;
  }

  auto it = handles_.find(remote_addr);
  if (it != handles_.end()) {
    // Reuse the existing session until the last caller releases it.
    it->second.ref_count++;
    LOG(INFO) << "Reusing existing session for " << remote_addr
              << ", ref_count=" << it->second.ref_count;
    return true;
  }

  Transport::SegmentHandle handle = engine_->openSegment(remote_addr);
  if (handle == static_cast<Transport::SegmentHandle>(-1)) {
    LOG(ERROR) << "Fail to connect to " << remote_addr;
    return false;
  }

  SessionInfo session_info;
  session_info.handle = handle;
  session_info.ref_count = 1;
  session_info.generation = next_session_generation_++;
  handles_[remote_addr] = session_info;

  LOG(INFO) << "Created new session for " << remote_addr << ", ref_count=1";

  return true;
}

bool MooncakeTransferEngineCore::close_session(const uint64_t cluster_id,
                                               const std::string& remote_addr) {
  std::unique_lock<std::mutex> lock(mutex_);

  LOG(INFO) << "close_session, cluster_id=" << cluster_id
            << ", remote_addr=" << remote_addr;

  if (cluster_id != 0) {
    session_rpc_cv_.wait(lock, [this, &remote_addr]() {
      return !session_rpc_inflight_.contains(remote_addr);
    });
    session_rpc_inflight_.insert(remote_addr);

    const bool result = [this, cluster_id, &remote_addr, &lock]() {
      auto it = handles_.find(remote_addr);
      if (it != handles_.end()) {
        if (it->second.ref_count <= 0) {
          LOG(ERROR) << "Invalid session ref_count for " << remote_addr << ": "
                     << it->second.ref_count;
          return false;
        }
        if (it->second.ref_count > 1) {
          it->second.ref_count--;
          LOG(INFO) << "Decremented ref_count for " << remote_addr
                    << ", ref_count=" << it->second.ref_count;
          return true;
        }
      }

      proto::MooncakeTransferEngineService_Stub* stub =
          get_or_create_stub_locked(cluster_id);
      if (stub == nullptr) {
        LOG(ERROR) << "create_rpc_channel failed for cluster_id=" << cluster_id;
        return false;
      }

      Transport::SegmentHandle handle =
          static_cast<Transport::SegmentHandle>(-1);
      uint64_t generation = 0;
      if (it != handles_.end()) {
        handle = it->second.handle;
        generation = it->second.generation;
      }

      const std::string local_addr = addr_;
      lock.unlock();
      const bool closed = close_remote_session(stub, local_addr, remote_addr);
      lock.lock();
#if defined(USE_DCU)
      if (closed && handle != static_cast<Transport::SegmentHandle>(-1)) {
        it = handles_.find(remote_addr);
        if (it != handles_.end() && it->second.generation == generation) {
          if (it->second.ref_count > 1) {
            it->second.ref_count--;
          } else {
            engine_->closeSegment(handle);
            handles_.erase(it);
          }
        }
      }
#endif
      return closed;
    }();

    session_rpc_inflight_.erase(remote_addr);
    lock.unlock();
    session_rpc_cv_.notify_all();
    return result;
  }

  auto it = handles_.find(remote_addr);
  if (it == handles_.end()) {
    return true;
  }

  it->second.ref_count--;
  LOG(INFO) << "Decremented ref_count for " << remote_addr
            << ", ref_count=" << it->second.ref_count;

  if (it->second.ref_count > 0) {
    return true;
  }

  SegmentHandle handle = it->second.handle;
  if (handle != static_cast<SegmentHandle>(-1)) {
    engine_->closeSegment(handle);
  }
  handles_.erase(it);

  LOG(INFO) << "Closed session for " << remote_addr;

  return true;
}

SegmentHandle MooncakeTransferEngineCore::get_handle(
    const std::string& remote_addr) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = handles_.find(remote_addr);
  if (it == handles_.end()) {
    return static_cast<SegmentHandle>(-1);
  }
  return it->second.handle;
}

proto::MooncakeTransferEngineService_Stub*
MooncakeTransferEngineCore::get_or_create_stub(uint64_t cluster_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  return get_or_create_stub_locked(cluster_id);
}

proto::MooncakeTransferEngineService_Stub*
MooncakeTransferEngineCore::get_or_create_stub_locked(uint64_t cluster_id) {
  auto it = stub_map_.find(cluster_id);
  if (it == stub_map_.end()) {
    auto [remote_ip, remote_port] = net::convert_uint64_to_ip_port(cluster_id);
    std::string remote_addr = remote_ip + ":" + std::to_string(remote_port);

    brpc::Channel* channel = new brpc::Channel();
    brpc::ChannelOptions options;
    options.timeout_ms = -1;
    std::string load_balancer = "";
    if (channel->Init(remote_addr.c_str(), load_balancer.c_str(), &options) !=
        0) {
      LOG(ERROR) << "Fail to initialize channel for " << remote_addr;
      delete channel;
      return nullptr;
    }

    proto::MooncakeTransferEngineService_Stub* stub =
        new proto::MooncakeTransferEngineService_Stub(channel);
    stub_map_[cluster_id] = stub;
    return stub;
  }

  return it->second;
}

// ============================================================================
// MooncakeTransferEngine
// ============================================================================

MooncakeTransferEngine::MooncakeTransferEngine(const uint16_t listen_port,
                                               const torch::Device& device)
    : listen_port_(listen_port),
      device_(device),
      core_(MooncakeTransferEngineCore::get_instance()) {}

std::string MooncakeTransferEngine::initialize() {
  if (!core_.initialize(listen_port_, device_)) {
    LOG(ERROR) << "Failed to initialize MooncakeTransferEngineCore";
    return "";
  }
  return core_.addr();
}

bool MooncakeTransferEngine::register_memory(std::vector<void*> addrs,
                                             std::vector<size_t> lens,
                                             std::vector<uint64_t> buf_bytes) {
  if (addrs.size() != lens.size() || addrs.size() != buf_bytes.size()) {
    LOG(ERROR) << "register_memory input size mismatch, addrs=" << addrs.size()
               << ", lens=" << lens.size()
               << ", buf_bytes=" << buf_bytes.size();
    return false;
  }

  TransferEngine* engine = core_.engine();
  for (size_t i = 0; i < addrs.size(); ++i) {
    int32_t ret = engine->registerLocalMemory(
        addrs[i], lens[i], kWildcardLocation, true, true);
    if (ret != 0) {
      LOG(ERROR) << "registerLocalMemory failed, buf_id=" << i
                 << ", addr=" << addrs[i] << ", len=" << lens[i]
                 << ", ret=" << ret;
      return false;
    }
  }

  buf_bytes_.insert(buf_bytes_.end(), buf_bytes.begin(), buf_bytes.end());
  LOG(INFO) << "register_memory success, total_buf_num=" << buf_bytes_.size();

  return true;
}

proto::MooncakeTransferEngineService_Stub*
MooncakeTransferEngine::create_rpc_channel(uint64_t cluster_id) {
  return core_.get_or_create_stub(cluster_id);
}

bool MooncakeTransferEngine::open_session(const uint64_t cluster_id,
                                          const std::string& remote_addr) {
  return core_.open_session(cluster_id, remote_addr);
}

bool MooncakeTransferEngine::close_session(const uint64_t cluster_id,
                                           const std::string& remote_addr) {
  return core_.close_session(cluster_id, remote_addr);
}

// Merge the source and destination block ids into a single block when both are
// consecutive.
void merge_block_ids(const std::vector<uint64_t>& src_blocks,
                     const std::vector<uint64_t>& dst_blocks,
                     std::vector<uint64_t>& merged_src_blocks,
                     std::vector<uint64_t>& merged_dst_blocks,
                     std::vector<uint64_t>& block_lengths) {
  size_t block_num = src_blocks.size();
  if (block_num == 0) {
    return;
  }

  std::vector<uint64_t> indices(block_num);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(
      indices.begin(), indices.end(), [&src_blocks](uint64_t i, uint64_t j) {
        return src_blocks[i] < src_blocks[j];
      });

  std::vector<uint64_t> sorted_src_blocks;
  std::vector<uint64_t> sorted_dst_blocks;
  sorted_src_blocks.reserve(block_num);
  sorted_dst_blocks.reserve(block_num);
  for (uint64_t id : indices) {
    sorted_src_blocks.emplace_back(src_blocks[id]);
    sorted_dst_blocks.emplace_back(dst_blocks[id]);
  }

  uint64_t current_src_id = sorted_src_blocks[0];
  uint64_t current_dst_id = sorted_dst_blocks[0];
  uint64_t current_length = 1;
  merged_src_blocks.reserve(block_num);
  merged_dst_blocks.reserve(block_num);
  block_lengths.reserve(block_num);
  for (size_t i = 1; i < sorted_src_blocks.size(); ++i) {
    if (sorted_src_blocks[i] == sorted_src_blocks[i - 1] + 1 &&
        sorted_dst_blocks[i] == sorted_dst_blocks[i - 1] + 1) {
      current_length++;
    } else {
      merged_src_blocks.emplace_back(current_src_id);
      merged_dst_blocks.emplace_back(current_dst_id);
      block_lengths.emplace_back(current_length);
      current_src_id = sorted_src_blocks[i];
      current_dst_id = sorted_dst_blocks[i];
      current_length = 1;
    }
  }
  merged_src_blocks.emplace_back(current_src_id);
  merged_dst_blocks.emplace_back(current_dst_id);
  block_lengths.emplace_back(current_length);
}

uint64_t MooncakeTransferEngine::transfer_bytes_for_blocks(
    size_t block_count,
    const std::vector<int64_t>& buf_ids) const {
  std::vector<int64_t> active_buf_ids;
  if (buf_ids.empty()) {
    active_buf_ids.resize(buf_bytes_.size());
    std::iota(active_buf_ids.begin(), active_buf_ids.end(), 0);
  } else {
    active_buf_ids = buf_ids;
  }

  uint64_t bytes_per_block = 0;
  for (int64_t buf_id : active_buf_ids) {
    if (buf_id < 0 || static_cast<size_t>(buf_id) >= buf_bytes_.size()) {
      LOG(ERROR) << "buf_id out of range while calculating transfer bytes, "
                 << "buf_id=" << buf_id << ", buf_cnt=" << buf_bytes_.size();
      return 0;
    }
    const uint64_t buf_bytes = buf_bytes_[static_cast<size_t>(buf_id)];
    if (bytes_per_block > std::numeric_limits<uint64_t>::max() - buf_bytes) {
      LOG(ERROR) << "transfer bytes per block overflow";
      return 0;
    }
    bytes_per_block += buf_bytes;
  }
  if (block_count > 0 &&
      bytes_per_block > std::numeric_limits<uint64_t>::max() / block_count) {
    LOG(ERROR) << "transfer byte count overflow";
    return 0;
  }
  return bytes_per_block * static_cast<uint64_t>(block_count);
}

bool MooncakeTransferEngine::move_memory_blocks(
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<int64_t>& buf_ids,
    MoveOpcode move_opcode,
    MooncakeTransferTrace* trace) {
  initialize_transfer_trace(trace);
  if (src_blocks.size() != dst_blocks.size()) {
    LOG(ERROR) << "src_blocks size must equal dst_blocks size, src="
               << src_blocks.size() << ", dst=" << dst_blocks.size();
    return fail_transfer_trace(trace, "invalid_block_count");
  }

  SegmentHandle remote_handle = core_.get_handle(remote_addr);
  if (remote_handle == static_cast<SegmentHandle>(-1)) {
    LOG(ERROR) << "remote addr does not exist: " << remote_addr;
    return fail_transfer_trace(trace, "remote_not_linked");
  }

  TransferEngine* engine = core_.engine();
  std::shared_ptr<TransferMetadata::SegmentDesc> remote_segment_desc =
      engine->getMetadata()->getSegmentDescByID(remote_handle);
  if (!remote_segment_desc) {
    LOG(ERROR) << "remote_segment_desc is null";
    return fail_transfer_trace(trace, "missing_remote_segment");
  }

  std::shared_ptr<TransferMetadata::SegmentDesc> local_segment_desc =
      engine->getMetadata()->getSegmentDescByID(LOCAL_SEGMENT_ID);
  if (!local_segment_desc) {
    LOG(ERROR) << "local_segment_desc is null";
    return fail_transfer_trace(trace, "missing_local_segment");
  }

  size_t local_buf_cnt = local_segment_desc->buffers.size();
  size_t remote_buf_cnt = remote_segment_desc->buffers.size();
  if (local_buf_cnt != remote_buf_cnt) {
    LOG(ERROR) << "buffer count mismatch, local=" << local_buf_cnt
               << ", remote=" << remote_buf_cnt;
    return fail_transfer_trace(trace, "buffer_count_mismatch");
  }
  if (local_buf_cnt != buf_bytes_.size()) {
    LOG(ERROR) << "registered buffer count mismatch, local=" << local_buf_cnt
               << ", block_bytes=" << buf_bytes_.size();
    return fail_transfer_trace(trace, "registered_buffer_count_mismatch");
  }

  std::vector<uint64_t> merged_src_blocks;
  std::vector<uint64_t> merged_dst_blocks;
  std::vector<uint64_t> block_lengths;
  merge_block_ids(src_blocks,
                  dst_blocks,
                  merged_src_blocks,
                  merged_dst_blocks,
                  block_lengths);

  std::vector<int64_t> active_buf_ids;
  if (buf_ids.empty()) {
    active_buf_ids.resize(buf_bytes_.size());
    std::iota(active_buf_ids.begin(), active_buf_ids.end(), 0);
  } else {
    active_buf_ids = buf_ids;
  }

  TransferRequest::OpCode opcode = TransferRequest::READ;
  if (move_opcode == MoveOpcode::WRITE) {
    opcode = TransferRequest::WRITE;
  }

  std::vector<TransferRequest> entries;
  for (int64_t buf_id : active_buf_ids) {
    if (buf_id < 0 || static_cast<size_t>(buf_id) >= local_buf_cnt) {
      LOG(ERROR) << "buf_id out of range, buf_id=" << buf_id
                 << ", buf_cnt=" << local_buf_cnt;
      return fail_transfer_trace(trace, "invalid_buffer_id");
    }

    size_t local_buf_id = static_cast<size_t>(buf_id);
    uint64_t buf_bytes = buf_bytes_[local_buf_id];
    uint64_t local_buf_len = local_segment_desc->buffers[local_buf_id].length;
    uint64_t remote_buf_len = remote_segment_desc->buffers[local_buf_id].length;

    char* local_base =
        reinterpret_cast<char*>(local_segment_desc->buffers[local_buf_id].addr);
    uint64_t remote_base = remote_segment_desc->buffers[local_buf_id].addr;
    for (size_t i = 0; i < merged_src_blocks.size(); ++i) {
      uint64_t src_block_id = merged_src_blocks[i];
      uint64_t dst_block_id = merged_dst_blocks[i];
      uint64_t block_length = block_lengths[i];
      uint64_t local_block_id = src_block_id;
      uint64_t remote_block_id = dst_block_id;
      if (move_opcode == MoveOpcode::READ) {
        local_block_id = dst_block_id;
        remote_block_id = src_block_id;
      }
      if (!check_buf_range(
              local_buf_len, buf_bytes, local_block_id, block_length, buf_id) ||
          !check_buf_range(remote_buf_len,
                           buf_bytes,
                           remote_block_id,
                           block_length,
                           buf_id)) {
        return fail_transfer_trace(trace, "block_range_error");
      }

      uint64_t local_bias = local_block_id * buf_bytes;
      uint64_t remote_bias = remote_block_id * buf_bytes;
      uint64_t len = block_length * buf_bytes;

      TransferRequest entry;
      entry.opcode = opcode;
      entry.length = len;
      entry.source = reinterpret_cast<void*>(local_base + local_bias);
      entry.target_id = remote_handle;
      entry.target_offset = remote_base + remote_bias;
      entry.advise_retry_cnt = 0;
      entries.push_back(entry);
    }
  }

  if (entries.empty()) {
    if (trace != nullptr) {
      const int64_t now_ns = pd_transfer_monotonic_time_ns();
      trace->submit_monotonic_ns = now_ns;
      trace->complete_monotonic_ns = now_ns;
      trace->result = "empty";
    }
    return true;
  }

  if (trace != nullptr) {
    for (const TransferRequest& entry : entries) {
      trace->bytes += entry.length;
    }
  }

  size_t batch_size = entries.size();
  auto batch_id = engine->allocateBatchID(batch_size);
  if (trace != nullptr) {
    trace->submit_monotonic_ns = pd_transfer_monotonic_time_ns();
  }
  mooncake::Status s = engine->submitTransfer(batch_id, entries);
  if (!s.ok()) {
    LOG(ERROR) << "submit failed";
    engine->freeBatchID(batch_id);
    return fail_transfer_trace(trace, "submit_failed");
  }

  TransferStatus status;
  bool completed = false;
  bool transfer_success = true;
  std::string transfer_result = "completed";
  while (!completed) {
    s = engine->getBatchTransferStatus(batch_id, status);
    if (!s.ok()) {
      LOG(ERROR) << "getBatchTransferStatus not ok";
      transfer_success = false;
      transfer_result = "status_error";
      break;
    }

    if (status.s == TransferStatusEnum::COMPLETED) {
      completed = true;
    } else if (status.s == TransferStatusEnum::FAILED) {
      LOG(ERROR) << "getBatchTransferStatus failed";
      transfer_success = false;
      transfer_result = "failed";
      completed = true;
    } else if (status.s == TransferStatusEnum::TIMEOUT) {
      LOG(ERROR) << "Sync data transfer timeout";
      transfer_success = false;
      transfer_result = "timeout";
      completed = true;
    }
  }
  if (trace != nullptr) {
    trace->complete_monotonic_ns = pd_transfer_monotonic_time_ns();
    trace->result = transfer_result;
  }

  s = engine->freeBatchID(batch_id);
  if (!s.ok()) {
    LOG(ERROR) << "freeBatchID failed";
    return fail_transfer_trace(trace, "free_failed");
  }

  return transfer_success;
}

bool MooncakeTransferEngine::move_memory_by_global_offsets(
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_offsets,
    const std::vector<uint64_t>& dst_offsets,
    size_t transfer_size,
    MoveOpcode move_opcode,
    MooncakeTransferTrace* trace) {
  initialize_transfer_trace(trace);
  if (src_offsets.size() != dst_offsets.size()) {
    LOG(ERROR) << "src_offsets size must equal dst_offsets size, src="
               << src_offsets.size() << ", dst=" << dst_offsets.size();
    return fail_transfer_trace(trace, "invalid_offset_count");
  }
  SegmentHandle remote_handle = core_.get_handle(remote_addr);
  if (remote_handle == static_cast<SegmentHandle>(-1)) {
    LOG(ERROR) << "remote addr does not exist: " << remote_addr;
    return fail_transfer_trace(trace, "remote_not_linked");
  }

  TransferEngine* engine = core_.engine();
  std::shared_ptr<TransferMetadata::SegmentDesc> remote_segment_desc =
      engine->getMetadata()->getSegmentDescByID(remote_handle);
  if (!remote_segment_desc) {
    LOG(ERROR) << "remote_segment_desc is null";
    return fail_transfer_trace(trace, "missing_remote_segment");
  }

  std::shared_ptr<TransferMetadata::SegmentDesc> local_segment_desc =
      engine->getMetadata()->getSegmentDescByID(LOCAL_SEGMENT_ID);
  if (!local_segment_desc) {
    LOG(ERROR) << "local_segment_desc is null";
    return fail_transfer_trace(trace, "missing_local_segment");
  }

  if (local_segment_desc->buffers.empty() ||
      remote_segment_desc->buffers.empty()) {
    LOG(ERROR) << "No buffers registered for XTensor mode";
    return fail_transfer_trace(trace, "missing_registered_buffer");
  }

  char* local_base =
      reinterpret_cast<char*>(local_segment_desc->buffers[0].addr);
  char* remote_base =
      reinterpret_cast<char*>(remote_segment_desc->buffers[0].addr);

  TransferRequest::OpCode opcode = TransferRequest::READ;
  if (move_opcode == MoveOpcode::WRITE) {
    opcode = TransferRequest::WRITE;
  }

  std::vector<TransferRequest> entries;
  entries.reserve(src_offsets.size());

  for (size_t i = 0; i < src_offsets.size(); ++i) {
    TransferRequest entry;
    entry.opcode = opcode;
    entry.length = transfer_size;
    entry.source = reinterpret_cast<void*>(local_base + src_offsets[i]);
    entry.target_id = remote_handle;
    entry.target_offset =
        reinterpret_cast<uint64_t>(remote_base + dst_offsets[i]);
    entry.advise_retry_cnt = 0;
    entries.push_back(entry);
  }

  if (entries.empty()) {
    if (trace != nullptr) {
      const int64_t now_ns = pd_transfer_monotonic_time_ns();
      trace->submit_monotonic_ns = now_ns;
      trace->complete_monotonic_ns = now_ns;
      trace->result = "empty";
    }
    return true;
  }
  if (trace != nullptr) {
    if (transfer_size > 0 &&
        entries.size() > std::numeric_limits<uint64_t>::max() / transfer_size) {
      return fail_transfer_trace(trace, "transfer_byte_count_overflow");
    }
    trace->bytes = static_cast<uint64_t>(entries.size()) * transfer_size;
  }

  size_t batch_size = entries.size();
  auto batch_id = engine->allocateBatchID(batch_size);
  if (trace != nullptr) {
    trace->submit_monotonic_ns = pd_transfer_monotonic_time_ns();
  }
  mooncake::Status s = engine->submitTransfer(batch_id, entries);
  if (!s.ok()) {
    LOG(ERROR) << "submit failed in move_memory_by_global_offsets";
    engine->freeBatchID(batch_id);
    return fail_transfer_trace(trace, "submit_failed");
  }

  TransferStatus status;
  bool completed = false;
  bool transfer_success = true;
  std::string transfer_result = "completed";
  while (!completed) {
    s = engine->getBatchTransferStatus(batch_id, status);
    if (!s.ok()) {
      LOG(ERROR) << "getBatchTransferStatus not ok";
      transfer_success = false;
      transfer_result = "status_error";
      break;
    }

    if (status.s == TransferStatusEnum::COMPLETED) {
      completed = true;
    } else if (status.s == TransferStatusEnum::FAILED) {
      LOG(ERROR) << "getBatchTransferStatus failed";
      transfer_success = false;
      transfer_result = "failed";
      completed = true;
    } else if (status.s == TransferStatusEnum::TIMEOUT) {
      LOG(ERROR) << "Sync data transfer timeout";
      transfer_success = false;
      transfer_result = "timeout";
      completed = true;
    }
  }
  if (trace != nullptr) {
    trace->complete_monotonic_ns = pd_transfer_monotonic_time_ns();
    trace->result = transfer_result;
  }

  s = engine->freeBatchID(batch_id);
  if (!s.ok()) {
    LOG(ERROR) << "freeBatchID failed";
    return fail_transfer_trace(trace, "free_failed");
  }

  return transfer_success;
}

bool MooncakeTransferEngine::pull_memory_blocks(
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<int64_t>& buf_ids) {
  bool ret = move_memory_blocks(
      remote_addr, src_blocks, dst_blocks, buf_ids, MoveOpcode::READ);
  if (!ret) {
    LOG(ERROR) << "Pull memory blocks failed, ret = " << ret;
    return false;
  }

  return true;
}

bool MooncakeTransferEngine::push_memory_blocks(
    const std::string& remote_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<int64_t>& buf_ids,
    MooncakeTransferTrace* trace) {
  bool ret = move_memory_blocks(
      remote_addr, src_blocks, dst_blocks, buf_ids, MoveOpcode::WRITE, trace);
  if (!ret) {
    LOG(ERROR) << "Push memory blocks failed, ret = " << ret;
    return false;
  }

  return true;
}

// ============================================================================
// MooncakeTransferEngineService
// ============================================================================

void MooncakeTransferEngineService::OpenSession(
    ::google::protobuf::RpcController* controller,
    const proto::SessionInfo* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (request == nullptr || response == nullptr || controller == nullptr) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  if (request->addr().empty()) {
    LOG(ERROR) << "OpenSession request missing addr";
    response->set_ok(false);
    return;
  }

  std::string remote_addr(request->addr());
  bool result =
      MooncakeTransferEngineCore::get_instance().open_session(0, remote_addr);

  response->set_ok(result);
}

void MooncakeTransferEngineService::CloseSession(
    ::google::protobuf::RpcController* controller,
    const proto::SessionInfo* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  if (request == nullptr || response == nullptr || controller == nullptr) {
    LOG(ERROR) << "brpc request | response | controller is null";
    return;
  }

  if (request->addr().empty()) {
    LOG(ERROR) << "CloseSession request missing addr";
    response->set_ok(false);
    return;
  }

  std::string remote_addr(request->addr());
  bool result =
      MooncakeTransferEngineCore::get_instance().close_session(0, remote_addr);

  response->set_ok(result);
}

}  // namespace xllm
