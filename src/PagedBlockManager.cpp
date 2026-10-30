#include "PagedBlockManager.h"

namespace agentinferkv
{
    PagedBlockManager::PagedBlockManager (const ModelConfig& config) : config_ (config)
    {
        free_blocks_.reserve(static_cast<std::size_t>(config_.num_blocks));
        for (int i = config_.num_blocks - 1; i >= 0; --i)
        {
            free_blocks_.push_back(static_cast<BlockId>(i));
        }
    }

    bool PagedBlockManager::CanAllocate (const Request& req) const
    {
        if (tables_.count(req.id)) return false;
        int need = config_.BlocksForTokens(req.PromptLen()); // 只按prompr估
        return static_cast<int> (free_blocks_.size()) >= need;
    }

    BlockTableView PagedBlockManager::Allocate (const Request& req)
    {
        if (!CanAllocate(req)) return BlockTableView{};

        int need = config_.BlocksForTokens (req.PromptLen());
        BlockTableView view;
        view.logical_to_physical.reserve(static_cast<std::size_t>(need));
        for (int i = 0; i < need; ++i) 
        {
            view.logical_to_physical.push_back(free_blocks_.back());
            free_blocks_.pop_back();
        }
        tables_.emplace(req.id, view);
        current_tokens_[req.id] = req.PromptLen();
        return view;
    }

    void PagedBlockManager::Free(RequestId id)
    {
        auto it = tables_.find(id);
        if (it == tables_.end()) return;

        for (BlockId bid : it->second.logical_to_physical) 
        {
            free_blocks_.push_back(bid);
        }
        tables_.erase(it);
        current_tokens_.erase(id);
    }

    int PagedBlockManager::NumFreeBlocks() const
    {
        return static_cast<int> (free_blocks_.size());
    }

    bool PagedBlockManager::AppendToken(RequestId id, int new_total_tokens, BlockTableView& out_view)
    {
        auto it = tables_.find(id);
        if (it == tables_.end()) return false;

        int need = config_.BlocksForTokens(new_total_tokens);
        int have = it->second.NumBlocks();
        int extra = need - have;

        if (extra <= 0)
        {
            //当前容量够用，不扩块，但更新 token 计数（影响 TailUnusedCapacity）
            current_tokens_[id] = new_total_tokens;
            out_view = it->second;
            return true;
        }

        if (static_cast<int>(free_blocks_.size()) < extra) 
        {
            return false;  // 全有或全无,一块都补不成就整体失败
        }

        for (int i = 0; i < extra; ++i)
        {
            it->second.logical_to_physical.push_back(free_blocks_.back());
            free_blocks_.pop_back();
        }
        current_tokens_[id] = new_total_tokens;
        out_view = it->second;
        return true;
    }

    int PagedBlockManager::EstimateBlocks(const Request& req) const 
    {
        return config_.BlocksForTokens(req.PromptLen());
    }

    int PagedBlockManager::TailUnusedCapacity() const 
    {
        int tail = 0;
        for (const auto& kv : current_tokens_) 
        {
            int mod = kv.second % config_.block_size;
            if (mod != 0) 
            {
                tail += config_.block_size - mod;
            }
        }
        return tail;
    }
}