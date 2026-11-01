// 回归测试：命中的前缀块必须真的进入请求的块表。
// 历史 bug：SimulationLoop 调 Allocate(req) 时没传 prefix_blocks，
// 于是前缀只降低了模拟 TTFT，物理块一份没省，而且 Retain 多出来的引用永远不还。
#include "FCFSScheduler.h"
#include "Interfaces.h"
#include "PagedBlockManager.h"
#include "PrefixCache.h"
#include "SimulatedExecutor.h"
#include "SimulationLoop.h"

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
    ModelConfig MakeConfig(int num_blocks)
    {
        ModelConfig c;
        c.block_size = 16;
        c.num_blocks = num_blocks;
        return c;
    }

    // 32 token：前 16 个共享，后 16 个各不相同
    Request MakeSharedPrefixReq(RequestId id, int unique_base)
    {
        Request r;
        r.id = id;
        r.arrival_time = 0.0;
        r.max_new_tokens = 1;
        r.prompt_tokens.resize(32);
        for (int i = 0; i < 16; ++i) r.prompt_tokens[static_cast<std::size_t>(i)] = 1000 + i;
        for (int i = 16; i < 32; ++i) r.prompt_tokens[static_cast<std::size_t>(i)] = unique_base + i;
        return r;
    }
}

// 直接测 BlockManager + PrefixCache 的协作：第二个请求只应新分配 1 块
static void TestSharedBlockEntersView()
{
    ModelConfig config = MakeConfig(16);
    PagedBlockManager mgr(config);
    PrefixCache cache(&mgr, 0);

    Request r1 = MakeSharedPrefixReq(1, 2000);
    BlockTableView view1 = mgr.Allocate(r1, {});
    CHECK(view1.NumBlocks() == 2);

    auto created = cache.Insert(r1.prompt_tokens, view1.logical_to_physical, config.block_size);
    CHECK(created.size() == 2);
    for (BlockId bid : created) mgr.RetainSharedBlock(bid);

    mgr.Free(1);
    CHECK(mgr.NumFreeBlocks() == 14);   // 16 - 2，这两块由缓存持有

    Request r2 = MakeSharedPrefixReq(2, 3000);
    auto match = cache.Match(r2.prompt_tokens, config.block_size);
    CHECK(match.matched_blocks.size() == 1);
    CHECK(match.matched_tokens == 16);

    for (BlockId bid : match.matched_blocks) mgr.RetainSharedBlock(bid);
    BlockTableView view2 = mgr.Allocate(r2, match.matched_blocks);

    CHECK(view2.NumBlocks() == 2);
    CHECK(view2.logical_to_physical[0] == match.matched_blocks[0]);  // 真的复用了同一块
    CHECK(mgr.NumFreeBlocks() == 13);                                 // 只新分配了 1 块

    mgr.Free(2);
    CHECK(mgr.NumFreeBlocks() == 14);   // 请求的引用归还，缓存的 2 块仍在
}

// 端到端：开缓存跑一段，账目必须守恒（free + cached == num_blocks）
static void TestBlockAccountingUnderSoak()
{
    ModelConfig config = MakeConfig(32);
    FCFSScheduler sched;
    PagedBlockManager mgr(config);
    SimulatedExecutor exec;
    PrefixCache cache(&mgr, 8);          // LRU 上限 8 块
    SimulationLoop loop(config, sched, mgr, exec, &cache);

    std::vector<Request> reqs;
    for (int i = 1; i <= 40; ++i)
    {
        Request r = MakeSharedPrefixReq(static_cast<RequestId>(i), 2000 + i * 100);
        r.arrival_time = 0.001 * (i - 1);
        r.max_new_tokens = 3;
        reqs.push_back(r);
    }

    auto result = loop.Run(std::move(reqs));
    cache.Trim();

    CHECK(result.completed.size() == 40);
    CHECK(cache.CachedBlocks() <= 8);
    CHECK(mgr.NumFreeBlocks() + static_cast<int>(cache.CachedBlocks()) == config.num_blocks);

    for (const auto& r : result.completed)
    {
        CHECK(r.state == RequestState::kDone);
    }
}

int main()
{
    TestSharedBlockEntersView();
    TestBlockAccountingUnderSoak();

    if (g_failures == 0)
    {
        std::printf("All prefix-sharing tests passed.\n");
        return 0;
    }
    std::printf("%d checks failed.\n", g_failures);
    return 1;
}
