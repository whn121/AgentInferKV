#include "PrefixCache.h"

namespace agentinferkv
{
    PrefixCache::Node* PrefixCache::FindChild(Node* cur, const std::vector<TokenId>& tokens)
    {
        for (auto& child : cur->children)
        {
            if (child->tokens == tokens) return child.get();
        }
        return nullptr;
    }

    const PrefixCache::Node* PrefixCache::FindChild(const Node* cur, const std::vector<TokenId>& tokens)
    {
        for (const auto& child : cur->children) 
        {
            if (child->tokens == tokens) return child.get();
        }
        return nullptr;
    }

    PrefixCache::MatchResult PrefixCache::Match(const std::vector<TokenId>& prompt, int block_size) const
    {
        total_queries_++;

        MatchResult result;
        const Node* cur = &root_;
        int offset = 0;
        const int n = static_cast<int> (prompt.size());

        while (offset + block_size <= n)
        {
            std::vector<TokenId> block_tokens(prompt.begin() + offset, prompt.begin() + offset + block_size);
            const Node* matched = FindChild (cur, block_tokens);
            if (matched == nullptr) break;

            result.matched_blocks.push_back(matched->block_id);
            result.matched_tokens += block_size;
            cur = matched;
            offset += block_size;
        }

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
                found = node.get();
                cur->children.push_back(std::move(node));
                newly_created.push_back(block_id);
            }
            found->ref_count++;
            cur = found;
            offset += block_size;
        }
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
}