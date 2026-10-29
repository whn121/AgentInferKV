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
        int NumFreeBlocks() const override;

    private:
        ModelConfig config_;
        std::vector<BlockId> free_blocks_; // 空闲块 ID 栈
        std::unordered_map<RequestId, BlockTableView> tables_; // 请求 -> 快视图 
    };
}