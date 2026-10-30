#pragma once

#include "Interfaces.h"

#include <cstdint>
#include <memory>
#include <vector>


namespace agentinferkv
{
    // 前缀缓存, 一棵按 block 对齐的树, 每个节点对应一个完整 block 的 token 序列
    // 只共享“完整填满的 block”, 不做 CoW
    class PrefixCache
    {
    public:
        // Match 的结果; 命中的块列表 + 命中token数
        struct MatchResult
        {
            std::vector<BlockId> matched_blocks;
            int matched_tokens = 0;
        };
        
        PrefixCache() = default;

        // 查找最长前缀(按 block_size 对齐)
        // 返回命中的 BlockId 列表和命中 token 数 const, 不改树
        MatchResult Match(const std::vector<TokenId>& prompt, int block_size) const;

        // 插入新计算的前缀。返回本次新建的块 ID 列表。
        // 调用方应对返回的每个块 RetainSharedBlock，让缓存持有引用。
        std::vector<BlockId> Insert(const std::vector<TokenId>& prompt, const std::vector<BlockId>& blocks, int block_size);

        // 释放某请求占用的前缀引用：沿 prompt 路径逐节点 ref_count--
        // 第一版不删节点, ref_count 减到 0 就停在那
        void Release(const std::vector<TokenId>& prompt, int block_size);

        // 观测：累计查询次数和命中次数(命中指 matched_tokens > 0)
        int TotalQueries() const { return total_queries_; }
        int TotalHits() const { return total_hits_; }

    private:
        struct Node 
        {
            std::vector<TokenId> tokens;              // block_size 个 token
            BlockId block_id = -1;                    // 对应物理块 root 是 -1
            std::vector<std::unique_ptr<Node>> children;
            int ref_count = 0;
        };

        Node root_;
        mutable int total_queries_ = 0;
        mutable int total_hits_ = 0;

        // 从 cur 的子节点里找 tokens 完全一致的节点,没有返回 nullptr
        static Node* FindChild(Node* cur, const std::vector<TokenId>& tokens);
        static const Node* FindChild(const Node* cur, const std::vector<TokenId>& tokens);
    };
}