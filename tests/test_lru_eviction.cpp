// 回归测试：PrefixCache 的 LRU 淘汰。
// 旧版本不淘汰，缓存持有的块直到进程结束都不还，块池会被慢慢吃光。
// 这里用 8 块池 + 2 块缓存上限，验证淘汰后物理块回到池子、命中率随之下降。
#include "Interfaces.h"
#include "PagedBlockManager.h"
#include "PrefixCache.h"

#include <cstdio>
#include <vector>

using namespace agentinferkv;

static int g_failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("CHECK failed: %s at line %d\n", #cond, __LINE__); \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

namespace
{
    ModelConfig MakeConfig()
    {
        ModelConfig c;
        c.block_size = 4;
        c.num_blocks = 8;
        return c;
    }

    // 4 个 token == 1 个 block
    Request MakeOneBlockReq(RequestId id, int base)
    {
        Request r;
        r.id = id;
        r.max_new_tokens = 1;
        r.prompt_tokens.resize(4);
        for (int i = 0; i < 4; ++i) r.prompt_tokens[static_cast<std::size_t>(i)] = base + i;
        return r;
    }

    // 模拟“一个请求走完：分配 → 插入缓存 → 请求结束”
    void RunOneRequest(PagedBlockManager& mgr, PrefixCache& cache, const ModelConfig& cfg,
                       RequestId id, int base)
    {
        Request r = MakeOneBlockReq(id, base);

        auto match = cache.Match(r.prompt_tokens, cfg.block_size);
        for (BlockId bid : match.matched_blocks) mgr.RetainSharedBlock(bid);

        BlockTableView view = mgr.Allocate(r, match.matched_blocks);
        auto created = cache.Insert(r.prompt_tokens, view.logical_to_physical, cfg.block_size);
        for (BlockId bid : created) mgr.RetainSharedBlock(bid);

        cache.Release(r.prompt_tokens, cfg.block_size);
        mgr.Free(r.id);
    }
}

static void TestEvictsLeastRecentlyUsed()
{
    ModelConfig cfg = MakeConfig();
    PagedBlockManager mgr(cfg);
    PrefixCache cache(&mgr, 2);          // 最多缓存 2 块

    RunOneRequest(mgr, cache, cfg, 1, 100);   // A
    RunOneRequest(mgr, cache, cfg, 2, 200);   // B
    CHECK(cache.CachedBlocks() == 2);
    CHECK(mgr.NumFreeBlocks() == 6);

    RunOneRequest(mgr, cache, cfg, 3, 300);   // C → 触发淘汰 A
    CHECK(cache.CachedBlocks() == 2);
    CHECK(cache.EvictedBlocks() == 1);
    CHECK(mgr.NumFreeBlocks() == 6);          // A 的块回到池子，C 的块进缓存

    // A 已被淘汰，不再命中；C 还在
    CHECK(cache.Match(MakeOneBlockReq(9, 100).prompt_tokens, cfg.block_size).matched_blocks.empty());
    CHECK(cache.Match(MakeOneBlockReq(9, 300).prompt_tokens, cfg.block_size).matched_blocks.size() == 1);
}

static void TestEvictionRespectsCapacityAndAccounting()
{
    ModelConfig cfg = MakeConfig();
    PagedBlockManager mgr(cfg);
    PrefixCache cache(&mgr, 3);

    for (int i = 0; i < 10; ++i)
    {
        RunOneRequest(mgr, cache, cfg, static_cast<RequestId>(i + 1), 1000 + i * 10);
    }

    CHECK(cache.CachedBlocks() <= 3);
    CHECK(cache.EvictedBlocks() == 7);
    CHECK(mgr.NumFreeBlocks() == 5);                              // 8 - 3
    CHECK(mgr.NumFreeBlocks() + static_cast<int>(cache.CachedBlocks()) == cfg.num_blocks);
}

// 没有 block_mgr 时退化成“不淘汰”，保持旧行为，不能崩
static void TestNoManagerMeansNoEviction()
{
    ModelConfig cfg = MakeConfig();
    PrefixCache cache(nullptr, 1);
    for (int i = 0; i < 3; ++i)
    {
        Request r = MakeOneBlockReq(static_cast<RequestId>(i + 1), 5000 + i * 10);
        auto created = cache.Insert(r.prompt_tokens, {static_cast<BlockId>(i + 1)}, cfg.block_size);
        CHECK(created.size() == 1);
    }
    CHECK(cache.CachedBlocks() == 3);
    CHECK(cache.EvictedBlocks() == 0);
}

int main()
{
    TestEvictsLeastRecentlyUsed();
    TestEvictionRespectsCapacityAndAccounting();
    TestNoManagerMeansNoEviction();

    if (g_failures == 0)
    {
        std::printf("All lru-eviction tests passed.\n");
        return 0;
    }
    std::printf("%d checks failed.\n", g_failures);
    return 1;
}
