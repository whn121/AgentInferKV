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

    bool NaivePreallocator::CanAllocate(const Request& req) const
    {
        if (tables_.count(req.id)) return false; // 已分配, 不重复分配
        int need = config_.BlocksForTokens(req.TotalTokens());
        return static_cast<int>(free_blocks_.size()) >= need;
    }

    BlockTableView NaivePreallocator::Allocate(const Request& req)
    {
        if (!CanAllocate(req)) return BlockTableView{};

        int need = config_.BlocksForTokens(req.TotalTokens());
        BlockTableView view;
        view.logical_to_physical.reserve(static_cast<std::size_t>(need));
        for (int i = 0; i < need; ++i) 
        {
            view.logical_to_physical.push_back(free_blocks_.back());
            free_blocks_.pop_back();
        }
        tables_.emplace(req.id, view);
        return view;
    }

    void NaivePreallocator::Free(RequestId id)
    {
        auto it = tables_.find(id);
        if (it == tables_.end()) return;  // 重复 Free

        for (BlockId bid : it->second.logical_to_physical) 
        {
            free_blocks_.push_back(bid);
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

    int NaivePreallocator::EstimateBlocks(const Request& req) const 
    {
        return config_.BlocksForTokens(req.TotalTokens());
    }

    int NaivePreallocator::NumFreeBlocks() const 
    {
        return static_cast<int>(free_blocks_.size());
    }
}