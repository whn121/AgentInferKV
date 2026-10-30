#pragma once

#include "Interfaces.h"

#include <unordered_map>
#include <vector>

namespace agentinferkv
{
    // 基线: 按prompt_len + max_new_tokens 一次性预分配
    // 这是vLLM最早期的朴素策略, 也是 Paged 要对比的对象
    class NaivePreallocator : public IBlockSpaceManager
    {
    public:
        explicit NaivePreallocator(const ModelConfig& config);

        bool CanAllocate(const Request& req) const override;
        BlockTableView Allocate(const Request& req) override;
        void Free(RequestId id) override;
        // Naive 已按 max_len 分满，扩块是空操作，直接成功。
        // 仍把当前 view 写回 out_view，保持接口语义一致。
        bool AppendToken(RequestId id, int new_total_tokens, BlockTableView& out_view) override;
        // Naive 按 prompt + max_new 估算块数。
        int EstimateBlocks(const Request& req) const override;
        int NumFreeBlocks() const override;

    private:
        ModelConfig config_;
        std::vector<BlockId> free_blocks_; // 空闲块 ID 栈
        std::unordered_map<RequestId, BlockTableView> tables_; // 请求 -> 快视图 
    };
}