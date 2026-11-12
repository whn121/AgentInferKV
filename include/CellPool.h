#pragma once

#include "Interfaces.h"
#include "ModelConfig.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace agentinferkv
{
    // 自管的物理 cell 池（L1：块 → 真实内存区间）。
    //
    // 和接 llama.cpp 的区别：llama.cpp 不把 KV 的 cell 地址暴露给外部，
    // 所以在真实执行器上我们只能做到"账目对齐"。这里用一块自管内存模拟真实 KV：
    //   block_id 的 cell 区间 = [block_id * block_size, (block_id+1) * block_size)
    // 每个 token 占 kv_bytes_per_token 字节（按模型口径），
    // 实际只写 1 字节标记（省内存），但写入量按真实字节数统计。
    //
    // 它的价值：能回答"你的分页到底分了什么内存"——
    //   - 块就是连续 cell；
    //   - 命中前缀的块不重复写（复用真的省了写）；
    //   - 块回收后 cell 可被重新分配。
    class CellPool
    {
    public:
        CellPool(int num_blocks, int block_size, std::size_t kv_bytes_per_token)
            : block_size_(block_size),
              bytes_per_token_(kv_bytes_per_token == 0 ? 1 : kv_bytes_per_token),
              markers_(static_cast<std::size_t>(num_blocks) * static_cast<std::size_t>(block_size), 0)
        {
        }

        int block_size() const { return block_size_; }
        std::size_t BytesPerBlock() const
        {
            return static_cast<std::size_t>(block_size_) * bytes_per_token_;
        }
        std::size_t CapacityBytes() const { return markers_.size() * bytes_per_token_; }

        // 把某个块里 [token_offset, token_offset + tokens) 这段"KV"写进去。
        // 返回写入的真实字节数（按模型口径）。
        std::size_t WriteTokens(BlockId block_id, int token_offset, int tokens, std::uint8_t tag)
        {
            if (tokens <= 0) return 0;
            const std::size_t base = static_cast<std::size_t>(block_id) *
                                     static_cast<std::size_t>(block_size_);
            for (int i = 0; i < tokens; ++i)
            {
                const std::size_t idx = base + static_cast<std::size_t>(token_offset + i);
                if (idx >= markers_.size()) break;
                markers_[idx] = tag;
            }
            const std::size_t bytes = static_cast<std::size_t>(tokens) * bytes_per_token_;
            written_bytes_ += bytes;
            if (written_bytes_ > peak_bytes_) peak_bytes_ = written_bytes_;
            return bytes;
        }

        std::uint8_t Marker(BlockId block_id, int token_offset) const
        {
            const std::size_t idx = static_cast<std::size_t>(block_id) *
                                        static_cast<std::size_t>(block_size_) +
                                    static_cast<std::size_t>(token_offset);
            return idx < markers_.size() ? markers_[idx] : 0xFF;
        }

        std::size_t WrittenBytes() const { return written_bytes_; }
        std::size_t PeakBytes() const { return peak_bytes_; }

        void ResetStats()
        {
            written_bytes_ = 0;
            peak_bytes_ = 0;
        }

    private:
        int block_size_;
        std::size_t bytes_per_token_;
        std::vector<std::uint8_t> markers_;
        std::size_t written_bytes_ = 0;
        std::size_t peak_bytes_ = 0;
    };
}
