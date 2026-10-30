#pragma once

#include "Interfaces.h"

#include <unordered_map>
#include <vector>

namespace agentinferkv
{
    // 按需分页分配, Allocate 只按prompt_len分, Decode时AppendToken补块
    class PagedBlockManager : public IBlockSpaceManager
    {
    public:
        explicit PagedBlockManager(const ModelConfig& config);

        bool CanAllocate(const Request& req) const override;
        BlockTableView Allocate(const Request& req) override;
        void Free(RequestId id) override;
        int NumFreeBlocks() const override;
        bool AppendToken(RequestId id, int new_total_tokens, BlockTableView& out_view) override;
        int EstimateBlocks(const Request& req) const override;

        // 所有已分配请求最后一块中未使用的token容量和
        // 用于对比“碎片”：Paged 只浪费最后一块的尾部，Naive 浪费整个预留区间
        int TailUnusedCapacity() const;

    private:
        ModelConfig config_;
        std::vector<BlockId> free_blocks_;
        std::unordered_map<RequestId, BlockTableView> tables_;
        std::unordered_map<RequestId, int> current_tokens_;  // 每请求当前已分配的 token 容量
    };
}
