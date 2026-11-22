#pragma once

#include "Interfaces.h"

#include <mutex>
#include <unordered_map>
#include <vector>

namespace agentinferkv
{
    // 按需分页分配, Allocate 只按prompt_len分, Decode时AppendToken补块
    //
    // 线程模型：free_blocks_ / tables_ / ref_counts_ 的读写全部由 mu_ 保护。
    // 单线程模拟里这把锁是多余的，但它让 RetainIfAlive 成为"查活 + 加引用"的原子操作，
    // 消除了 PrefixCache 匹配到 Retain 之间块被回收、再被分配给别人的窗口。
    class PagedBlockManager : public IBlockSpaceManager
    {
    public:
        explicit PagedBlockManager(const ModelConfig& config);

        bool CanAllocate(const Request& req, const std::vector<BlockId>& prefix_blocks = {}) const override;
        BlockTableView Allocate(const Request& req, const std::vector<BlockId>& prefix_blocks = {}) override;
        bool IsBlockAlive(BlockId id) const override;
        void Free(RequestId id) override;
        int NumFreeBlocks() const override;
        bool AppendToken(RequestId id, int new_total_tokens, BlockTableView& out_view) override;
        void RetainSharedBlock(BlockId id) override;
        void ReleaseSharedBlock(BlockId id) override;
        bool RetainIfAlive(BlockId id) override;
        int EstimateBlocks(const Request& req) const override;
        bool NeedsBlockForNextToken(RequestId id) const override;

        // 所有已分配请求最后一块中未使用的token容量和
        // 用于对比“碎片”：Paged 只浪费最后一块的尾部，Naive 浪费整个预留区间
        int TailUnusedCapacity() const;

    private:
        // 准入与分配必须同源：都按 prompt + resume_tokens 扣掉命中前缀后的净需求算。
        int SuffixTokens(const Request& req, const std::vector<BlockId>& prefix_blocks) const;
        bool CanAllocateLocked(const Request& req, const std::vector<BlockId>& prefix_blocks) const;
        void ReleaseLocked(BlockId id);

        ModelConfig config_;
        mutable std::mutex mu_;
        std::vector<BlockId> free_blocks_;
        std::unordered_map<RequestId, BlockTableView> tables_;
        std::unordered_map<RequestId, int> current_tokens_;  // 每请求当前已分配的 token 容量
        std::unordered_map<BlockId, int> ref_counts_;
    };
}
