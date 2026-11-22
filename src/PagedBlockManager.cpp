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

    int PagedBlockManager::SuffixTokens (const Request& req, const std::vector<BlockId>& prefix_blocks) const
    {
        int prefix_tokens = static_cast<int>(prefix_blocks.size()) * config_.block_size;
        // 保留进度重算时，已生成的 token 也要占块。
        // 准入和分配都走这一个函数，避免"预算按 A 算、分配按 B 算"。
        int suffix_tokens = req.PromptLen() + req.resume_tokens - prefix_tokens;
        if (suffix_tokens < 0) suffix_tokens = 0;
        return suffix_tokens;
    }

    bool PagedBlockManager::CanAllocateLocked (const Request& req, const std::vector<BlockId>& prefix_blocks) const
    {
        if (tables_.count(req.id)) return false;
        int need = config_.BlocksForTokens(SuffixTokens(req, prefix_blocks));
        return static_cast<int>(free_blocks_.size()) >= need;
    }

    bool PagedBlockManager::CanAllocate (const Request& req, const std::vector<BlockId>& prefix_blocks) const
    {
        std::lock_guard<std::mutex> lk(mu_);
        return CanAllocateLocked(req, prefix_blocks);
    }

    BlockTableView PagedBlockManager::Allocate (const Request& req, const std::vector<BlockId>& prefix_blocks)
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (!CanAllocateLocked(req, prefix_blocks)) return BlockTableView{};

        int need = config_.BlocksForTokens(SuffixTokens(req, prefix_blocks));

        BlockTableView view;
        view.logical_to_physical.reserve(prefix_blocks.size() + static_cast<std::size_t>(need));
        for (BlockId bid : prefix_blocks)
        {
            // 引用已由调用方通过 RetainIfAlive 加过，这里只挂到视图上
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
        current_tokens_[req.id] = req.PromptLen() + req.resume_tokens;
        return view;
    }

    bool PagedBlockManager::IsBlockAlive (BlockId id) const
    {
        std::lock_guard<std::mutex> lk(mu_);
        return ref_counts_.count(id) > 0;
    }

    void PagedBlockManager::Free (RequestId id)
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = tables_.find(id);
        if (it == tables_.end()) return;

        for (BlockId bid : it->second.logical_to_physical)
        {
            ReleaseLocked(bid);
        }
        tables_.erase(it);
        current_tokens_.erase(id);
    }

    int PagedBlockManager::NumFreeBlocks() const
    {
        std::lock_guard<std::mutex> lk(mu_);
        return static_cast<int>(free_blocks_.size());
    }

    bool PagedBlockManager::AppendToken (RequestId id, int new_total_tokens, BlockTableView& out_view)
    {
        std::lock_guard<std::mutex> lk(mu_);
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

    void PagedBlockManager::RetainSharedBlock (BlockId id)
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = ref_counts_.find(id);
        if (it == ref_counts_.end()) return;
        it->second++;
    }

    void PagedBlockManager::ReleaseSharedBlock (BlockId id)
    {
        std::lock_guard<std::mutex> lk(mu_);
        ReleaseLocked(id);
    }

    void PagedBlockManager::ReleaseLocked (BlockId id)
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

    bool PagedBlockManager::RetainIfAlive (BlockId id)
    {
        // "还活着"和"把引用加一"必须在同一把锁里完成：
        // 否则两步之间块可能被别处减到 0 回收、再被分配给另一个请求。
        std::lock_guard<std::mutex> lk(mu_);
        auto it = ref_counts_.find(id);
        if (it == ref_counts_.end()) return false;
        it->second++;
        return true;
    }

    int PagedBlockManager::EstimateBlocks (const Request& req) const
    {
        // 预算要按【净需求】估：命中前缀的部分可以直接共享，不该占预算。
        // 否则命中率越高越容易被调度器误拒（这是用户指出的真实缺陷）。
        int net = req.PromptLen() + req.resume_tokens - req.cached_prefix_len;
        if (net < 0) net = 0;
        return config_.BlocksForTokens(net);
    }

    bool PagedBlockManager::NeedsBlockForNextToken (RequestId id) const
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto tit = tables_.find(id);
        auto cit = current_tokens_.find(id);
        if (tit == tables_.end() || cit == current_tokens_.end()) return false;
        const int next = cit->second + 1;
        return config_.BlocksForTokens(next) > tit->second.NumBlocks();
    }

    int PagedBlockManager::TailUnusedCapacity() const
    {
        std::lock_guard<std::mutex> lk(mu_);
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
