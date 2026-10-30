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

    bool PagedBlockManager::CanAllocate (const Request& req, const std::vector<BlockId>& prefix_blocks) const
    {
        if (tables_.count(req.id)) return false;
        int prefix_tokens = static_cast<int>(prefix_blocks.size()) * config_.block_size;
        int suffix_tokens = req.PromptLen() - prefix_tokens;
        if (suffix_tokens < 0) suffix_tokens = 0;
        int need = config_.BlocksForTokens(suffix_tokens);
        return static_cast<int>(free_blocks_.size()) >= need;
    }

    BlockTableView PagedBlockManager::Allocate (const Request& req, const std::vector<BlockId>& prefix_blocks)
    {
        if (!CanAllocate(req, prefix_blocks)) return BlockTableView{};

        int prefix_tokens = static_cast<int>(prefix_blocks.size()) * config_.block_size;
        int suffix_tokens = req.PromptLen() - prefix_tokens;
        if (suffix_tokens < 0) suffix_tokens = 0;
        int need = config_.BlocksForTokens(suffix_tokens);

        BlockTableView view;
        view.logical_to_physical.reserve(prefix_blocks.size() + static_cast<std::size_t>(need));
        for (BlockId bid : prefix_blocks) 
        {
            view.logical_to_physical.push_back(bid);
        }
        for (int i = 0; i < need; ++i) 
        {
            BlockId bid = free_blocks_.back();
            free_blocks_.pop_back();
            view.logical_to_physical.push_back(bid);
            ref_counts_[bid] = 1;
        }
        tables_.emplace(req.id, view);
        current_tokens_[req.id] = req.PromptLen();
        return view;
    }

    bool PagedBlockManager::IsBlockAlive(BlockId id) const 
    {
        return ref_counts_.count(id) > 0;   
    }

    void PagedBlockManager::Free(RequestId id)
    {
        auto it = tables_.find(id);
        if (it == tables_.end()) return;

        for (BlockId bid : it->second.logical_to_physical) 
        {
            ReleaseSharedBlock(bid);
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
            BlockId bid = free_blocks_.back();
            free_blocks_.pop_back();
            it->second.logical_to_physical.push_back(bid);
            ref_counts_[bid] = 1;
        }
        current_tokens_[id] = new_total_tokens;
        out_view = it->second;
        return true;
    }

    void PagedBlockManager::RetainSharedBlock(BlockId id) 
    {        
        auto it = ref_counts_.find(id);
        if (it == ref_counts_.end()) return;
        it->second++;
    }

    void PagedBlockManager::ReleaseSharedBlock(BlockId id) 
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