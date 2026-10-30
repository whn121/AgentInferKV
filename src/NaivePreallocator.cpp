#include "NaivePreallocator.h"

namespace agentinferkv
{
    NaivePreallocator::NaivePreallocator (const ModelConfig& config) : config_ (config)
    {
        free_blocks_.reserve(static_cast<std::size_t> (config_.num_blocks));
        // 从高到低入栈, 这样分配时 pop_back 会先拿到0,1,2...
        for (int i = config_.num_blocks - 1; i >= 0; --i)
        {
            free_blocks_.push_back(static_cast<BlockId> (i));
        }
    }

    bool NaivePreallocator::CanAllocate(const Request& req, const std::vector<BlockId>& prefix_blocks) const
    {
        if (tables_.count(req.id)) return false;
        int prefix_tokens = static_cast<int>(prefix_blocks.size()) * config_.block_size;
        int suffix_tokens = req.TotalTokens() - prefix_tokens;
        if (suffix_tokens < 0) suffix_tokens = 0;
        int need = config_.BlocksForTokens(suffix_tokens);
        return static_cast<int>(free_blocks_.size()) >= need;
    }

    BlockTableView NaivePreallocator::Allocate(const Request& req, const std::vector<BlockId>& prefix_blocks)
    {
        if (!CanAllocate(req, prefix_blocks)) return BlockTableView{};

        int prefix_tokens = static_cast<int>(prefix_blocks.size()) * config_.block_size;
        int suffix_tokens = req.TotalTokens() - prefix_tokens;
        if (suffix_tokens < 0) suffix_tokens = 0;
        int need = config_.BlocksForTokens(suffix_tokens);

        BlockTableView view;
        view.logical_to_physical.reserve(prefix_blocks.size() + static_cast<std::size_t>(need));
        // 先放 prefix 块, 引用计数已由调用方 Retain，这里不再设 ref
        for (BlockId bid : prefix_blocks) 
        {
            view.logical_to_physical.push_back(bid);
        }
        // 再分配后缀新块
        for (int i = 0; i < need; ++i) 
        {
            BlockId bid = free_blocks_.back();
            free_blocks_.pop_back();
            view.logical_to_physical.push_back(bid);
            ref_counts_[bid] = 1;
        }
        tables_.emplace(req.id, view);
        return view;
    }

    bool NaivePreallocator::IsBlockAlive(BlockId id) const 
    {
        return ref_counts_.count(id) > 0;
    }

    void NaivePreallocator::Free(RequestId id)
    {
        auto it = tables_.find(id);
        if (it == tables_.end()) return;  // 重复 Free

        for (BlockId bid : it->second.logical_to_physical) 
        {
            ReleaseSharedBlock(bid);
        }
        tables_.erase(it);
    }

    bool NaivePreallocator::AppendToken(RequestId id, int new_total_tokens, BlockTableView& out_view)
    {
        (void)new_total_tokens;  // Naive 不看目标容量，已经分满

        auto it = tables_.find(id);
        if (it == tables_.end()) return false;  

        out_view = it->second;  // 写回当前视图，内容不变
        return true;
    }

    void NaivePreallocator::RetainSharedBlock(BlockId id) 
    {
        auto it = ref_counts_.find(id);
        if (it == ref_counts_.end()) return;  // 未分配的块，防御性忽略
        it->second++;
    }

    void NaivePreallocator::ReleaseSharedBlock(BlockId id) 
    {
        auto rc = ref_counts_.find(id);
        if (rc == ref_counts_.end()) return;
        rc->second--;
        if (rc->second <= 0) 
        {
            free_blocks_.push_back(id);
            ref_counts_.erase(rc);
        }
    }

    int NaivePreallocator::EstimateBlocks(const Request& req) const 
    {
        return config_.BlocksForTokens(req.TotalTokens());
    }

    int NaivePreallocator::NumFreeBlocks() const 
    {
        return static_cast<int>(free_blocks_.size());
    }
}