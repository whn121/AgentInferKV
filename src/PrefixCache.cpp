#include "PrefixCache.h"

#include <algorithm>

namespace agentinferkv
{
    PrefixCache::PrefixCache(IBlockSpaceManager* block_mgr, std::size_t max_cached_blocks)
        : block_mgr_(block_mgr), max_cached_blocks_(max_cached_blocks)
    {
    }

    PrefixCache::Node* PrefixCache::FindChild(Node* cur, const std::vector<TokenId>& tokens)
    {
        for (auto& child : cur->children)
        {
            if (child->tokens == tokens) return child.get();
        }
        return nullptr;
    }

    void PrefixCache::Touch(Node* node) const
    {
        // 缓存块数通常只有几十到几百，线性 remove 足够；换成迭代器表反而要维护悬挂指针
        lru_.remove(node);
        lru_.push_front(node);
    }

    PrefixCache::MatchResult PrefixCache::Match(const std::vector<TokenId>& prompt, int block_size) const
    {
        ++total_queries_;

        MatchResult result;
        Node* cur = const_cast<Node*>(&root_);
        int offset = 0;
        const int n = static_cast<int> (prompt.size());

        std::vector<Node*> touched;
        while (offset + block_size <= n)
        {
            std::vector<TokenId> block_tokens(prompt.begin() + offset, prompt.begin() + offset + block_size);
            Node* matched = FindChild (cur, block_tokens);
            if (matched == nullptr) break;

            result.matched_blocks.push_back(matched->block_id);
            result.matched_tokens += block_size;
            touched.push_back(matched);
            cur = matched;
            offset += block_size;
        }

        // 命中一次就把整条路径记为最近使用，避免热前缀被 LRU 淘汰
        for (Node* node : touched) Touch(node);

        if (result.matched_tokens > 0) total_hits_++;
        return result;
    }

    std::vector<BlockId> PrefixCache::Insert(const std::vector<TokenId>& prompt, const std::vector<BlockId>& blocks
                                            , int block_size)
    {
        std::vector<BlockId> newly_created;
        Node* cur = &root_;
        int offset = 0;
        const int n = static_cast<int>(prompt.size());

        for (BlockId block_id : blocks)
        {
            if (offset + block_size > n) break;

            std::vector<TokenId> block_tokens(prompt.begin() + offset, prompt.begin() + offset + block_size);
            Node* found = FindChild(cur, block_tokens);
            if (found == nullptr)
            {
                auto node = std::make_unique<Node>();
                node->tokens = block_tokens;
                node->block_id = block_id;
                node->parent = cur;
                node->ref_count = 1;
                found = node.get();
                cur->children.push_back(std::move(node));
                by_block_[block_id] = found;
                newly_created.push_back(block_id);
            }
            else
            {
                found->ref_count++;
            }
            Touch(found);
            cur = found;
            offset += block_size;
        }

        EvictIfNeeded();
        return newly_created;
    }

    void PrefixCache::Release(const std::vector<TokenId>& prompt, int block_size)
    {
        Node* cur = &root_;
        int offset = 0;
        const int n = static_cast<int>(prompt.size());

        while (offset + block_size <= n)
        {
            std::vector<TokenId> block_tokens(prompt.begin() + offset, prompt.begin() + offset + block_size);
            Node* found = FindChild(cur, block_tokens);
            if (found == nullptr) break;
            if (found->ref_count > 0) found->ref_count--;
            cur = found;
            offset += block_size;
        }
    }

    void PrefixCache::Trim()
    {
        EvictIfNeeded();
    }

    void PrefixCache::EvictIfNeeded()
    {
        if (max_cached_blocks_ == 0 || block_mgr_ == nullptr) return;

        // 每次淘汰一个最久未使用的叶子；淘汰后父节点可能也变成叶子，下一轮继续
        while (by_block_.size() > max_cached_blocks_)
        {
            if (!EvictOne()) break;
        }
    }

    bool PrefixCache::EvictOne()
    {
        for (auto it = lru_.end(); it != lru_.begin(); )
        {
            --it;
            Node* node = *it;
            if (!node->children.empty() || node->ref_count > 0) continue;

            Node* parent = node->parent;
            const BlockId bid = node->block_id;

            if (parent != nullptr)
            {
                auto& kids = parent->children;
                kids.erase(std::remove_if(kids.begin(), kids.end(),
                    [node](const std::unique_ptr<Node>& p) { return p.get() == node; }),
                    kids.end());
            }

            lru_.erase(it);
            by_block_.erase(bid);
            ++evicted_blocks_;

            // 缓存持有的那一份引用在这里归还；仍被请求块表引用的块不会真的回池
            block_mgr_->ReleaseSharedBlock(bid);
            return true;
        }
        return false;
    }
}
