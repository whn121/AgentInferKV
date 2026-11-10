// 块 → 物理 cell 的映射演示（回答"你的分页是不是假的"）。
//
// 背景：接 llama.cpp 时，物理 KV 的 cell 由它内部管理，我们只能做到"账目对齐"。
// 这个工具用一个自管的 KvCellPool 把"块就是一段连续 cell"这件事做实：
//   - 每个 BlockId 映射到 [cell_begin, cell_end)；
//   - 写入假 KV（每个 cell 写一个字节标记）；
//   - 释放后重新分配，验证同一段 cell 被复用（说明池子真的在用这块内存）；
//   - 统计内部碎片（最后一块没用满的容量）。
//
// 用法：./agentinferkv_kvcell_demo

#include "Interfaces.h"
#include "ModelConfig.h"
#include "PagedBlockManager.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace agentinferkv;

namespace
{
    // 自管的物理 cell 池：一个 BlockId 对应 block_size 个连续 cell。
    // 真实系统里这块内存就是 KV 张量；这里用 1 字节/cell 的标记代替，
    // 目的是验证"块地址"是可计算的、可复用的。
    class KvCellPool
    {
    public:
        KvCellPool(int num_blocks, int block_size)
            : block_size_(block_size), cells_(static_cast<std::size_t>(num_blocks) * block_size, 0)
        {
        }

        int Begin(BlockId id) const { return static_cast<int>(id) * block_size_; }
        int End(BlockId id) const { return Begin(id) + block_size_; }

        // 写一块"KV"：每个 cell 记一个标记
        void Write(BlockId id, std::uint8_t tag)
        {
            for (int c = Begin(id); c < End(id); ++c)
            {
                cells_[static_cast<std::size_t>(c)] = tag;
            }
        }

        std::uint8_t Read(int cell) const { return cells_[static_cast<std::size_t>(cell)]; }

    private:
        int block_size_;
        std::vector<std::uint8_t> cells_;
    };

    std::string Hex(const std::vector<BlockId>& ids)
    {
        std::string out;
        for (BlockId id : ids)
        {
            if (!out.empty()) out += ",";
            out += std::to_string(id);
        }
        return out;
    }
}

int main()
{
    ModelConfig config;
    config.block_size = 16;
    config.num_blocks = 8;      // 小池子，方便观察复用

    PagedBlockManager mgr(config);
    KvCellPool pool(config.num_blocks, config.block_size);

    std::printf("pool: blocks=%d block_size=%d cells=%d\n",
                config.num_blocks, config.block_size, config.num_blocks * config.block_size);

    // ── 第一个请求：prompt 40 token（3 块）──
    Request r1;
    r1.id = 1;
    r1.prompt_tokens.resize(40);
    for (int i = 0; i < 40; ++i) r1.prompt_tokens[static_cast<std::size_t>(i)] = i;

    BlockTableView v1 = mgr.Allocate(r1, {});
    std::printf("req1 blocks=[%s]\n", Hex(v1.logical_to_physical).c_str());

    // 写入并验证：块 i 的 cell 区间 = [i*16, i*16+16)
    std::uint8_t tag = 0xA0;
    for (BlockId id : v1.logical_to_physical)
    {
        pool.Write(id, tag);
        const int b = pool.Begin(id);
        std::printf("  block %u -> cells [%d,%d) tag=0x%02X first=%02X last=%02X\n",
                    id, b, pool.End(id), tag, pool.Read(b), pool.Read(pool.End(id) - 1));
        ++tag;
    }

    // ── 内部碎片：最后一块没用满 ──
    std::printf("internal_fragmentation_tokens=%d (40 token 用 3 块，最后一块剩 %d)\n",
                mgr.TailUnusedCapacity(), 3 * config.block_size - 40);

    // ── 释放后重新分配，验证 cell 被复用 ──
    mgr.Free(r1.id);
    std::printf("after free: free_blocks=%d\n", static_cast<int>(mgr.NumFreeBlocks()));

    Request r2;
    r2.id = 2;
    r2.prompt_tokens.resize(16);   // 只要 1 块
    BlockTableView v2 = mgr.Allocate(r2, {});
    std::printf("req2 blocks=[%s] (应当复用刚释放的块)\n", Hex(v2.logical_to_physical).c_str());

    const bool reused = !v2.logical_to_physical.empty() &&
                        v2.logical_to_physical[0] == v1.logical_to_physical.back();
    std::printf("cell_reuse_verified=%s\n", reused ? "yes" : "no(取决于空闲栈顺序)");

    // ── 全部分配完再申请，应当失败（全有或全无）──
    Request r3;
    r3.id = 3;
    r3.prompt_tokens.resize(16 * 20);   // 20 块，池子只有 8
    BlockTableView v3 = mgr.Allocate(r3, {});
    std::printf("over_budget_allocate_blocks=%zu (期望 0：全有或全无)\n",
                v3.logical_to_physical.size());

    mgr.Free(r2.id);
    std::printf("final_free_blocks=%d\n", static_cast<int>(mgr.NumFreeBlocks()));
    return 0;
}
