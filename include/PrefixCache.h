#pragma once

#include "Interfaces.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>


namespace agentinferkv
{
    // 前缀缓存, 一棵按 block 对齐的树, 每个节点对应一个完整 block 的 token 序列
    // 只共享“完整填满的 block”, 不做 CoW
    //
    // LRU 说明：
    //   - max_cached_blocks == 0（默认）表示不淘汰，缓存持有块的引用直到进程结束；
    //   - max_cached_blocks > 0 且 block_mgr 非空时，超出容量会按 LRU 淘汰叶子节点，
    //     并通过 IBlockSpaceManager::ReleaseSharedBlock 把物理块还给池子。
    //   - 只有“叶子 + ref_count == 0”的节点可被淘汰：非叶子节点下面还挂着别的前缀，
    //     有活跃引用的节点可能正被某个请求的块表使用。
    class PrefixCache
    {
    public:
        // Match 的结果; 命中的块列表 + 命中token数
        struct MatchResult
        {
            std::vector<BlockId> matched_blocks;
            int matched_tokens = 0;
        };

        explicit PrefixCache(IBlockSpaceManager* block_mgr = nullptr,
                             std::size_t max_cached_blocks = 0);

        // 查找最长前缀(按 block_size 对齐)，同时把命中路径标记为最近使用
        // 返回命中的 BlockId 列表和命中 token 数
        MatchResult Match(const std::vector<TokenId>& prompt, int block_size) const;

        // 插入新计算的前缀。返回本次新建的块 ID 列表。
        // 调用方应对返回的每个块 RetainSharedBlock，让缓存持有引用。
        std::vector<BlockId> Insert(const std::vector<TokenId>& prompt, const std::vector<BlockId>& blocks, int block_size);

        // 释放某请求占用的前缀引用：沿 prompt 路径逐节点 ref_count--
        // 不删节点；ref_count 只表示“还有多少个活跃请求走过这个节点”
        void Release(const std::vector<TokenId>& prompt, int block_size);

        // 观测：累计查询次数和命中次数(命中指 matched_tokens > 0)
        int TotalQueries() const { return total_queries_; }
        int TotalHits() const { return total_hits_; }

        // 观测：当前缓存持有的块数 / 累计淘汰的块数
        std::size_t CachedBlocks() const { return by_block_.size(); }
        std::size_t EvictedBlocks() const { return evicted_blocks_; }

        // 在请求之间的间隙调用，把仍然超额的缓存压回 max_cached_blocks。
        // Insert 时也会尝试淘汰，但如果当时所有叶子都被活跃请求引用，就会暂时超额。
        void Trim();

    private:
        struct Node
        {
            std::vector<TokenId> tokens;              // block_size 个 token
            BlockId block_id = -1;                    // 对应物理块 root 是 -1
            Node* parent = nullptr;                   // 淘汰时从父节点摘除
            std::vector<std::unique_ptr<Node>> children;
            int ref_count = 0;                        // 活跃请求数（Insert +1 / Release -1）
        };

        // 从 cur 的子节点里找 tokens 完全一致的节点,没有返回 nullptr
        static Node* FindChild(Node* cur, const std::vector<TokenId>& tokens);

        void Touch(Node* node) const;
        void EvictIfNeeded();
        bool EvictOne();

        Node root_;
        IBlockSpaceManager* block_mgr_ = nullptr;
        std::size_t max_cached_blocks_ = 0;

        mutable std::list<Node*> lru_;                        // front = 最近使用
        mutable std::unordered_map<BlockId, Node*> by_block_;

        mutable int total_queries_ = 0;
        mutable int total_hits_ = 0;
        std::size_t evicted_blocks_ = 0;
    };
}
