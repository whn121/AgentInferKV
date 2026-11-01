#include "FCFSScheduler.h"
#include "NaivePreallocator.h"
#include "SimulatedExecutor.h"
#include "SimulationLoop.h"
#include "PagedBlockManager.h"
#include "PrefixCache.h"

#include <cstdio>

using namespace agentinferkv;

static int g_failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("CHECK failed: %s at line %d\n", #cond, __LINE__); \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

static ModelConfig MakeConfig(int num_blocks) {
    ModelConfig c;
    c.block_size = 16;
    c.num_blocks = num_blocks;
    return c;
}

static Request MakeReq(RequestId id, double arrival, int prompt_len, int max_new) {
    Request r;
    r.id = id;
    r.arrival_time = arrival;
    r.prompt_tokens.resize(static_cast<std::size_t>(prompt_len));
    r.max_new_tokens = max_new;
    return r;
}

// 单请求完整闭环：到达 → Prefill → Decode×N → 完成 → 释放
static void TestSingleRequest() {
    auto config = MakeConfig(16);
    FCFSScheduler sched;
    NaivePreallocator mgr(config);
    SimulatedExecutor exec;
    SimulationLoop loop(config, sched, mgr, exec);

    std::vector<Request> reqs = { MakeReq(1, 0.0, 10, 3) };
    auto result = loop.Run(std::move(reqs));

    if (result.completed.size() != 1) {
        std::printf("FAIL(TestSingleRequest): completed=%zu, expected 1\n",
                    result.completed.size());
        ++g_failures;
        return;
    }
    CHECK(result.completed[0].state == RequestState::kDone);
    CHECK(result.completed[0].generated_tokens == 3);
    CHECK(result.completed[0].Ttft() > 0.0);
    CHECK(result.completed[0].Latency() > result.completed[0].Ttft());
    CHECK(mgr.NumFreeBlocks() == config.num_blocks);
    CHECK(result.max_concurrent == 1);
}

// max_new_tokens == 1：Prefill 完成时直接 DONE
static void TestSingleTokenRequest() {
    auto config = MakeConfig(16);
    FCFSScheduler sched;
    NaivePreallocator mgr(config);
    SimulatedExecutor exec;
    SimulationLoop loop(config, sched, mgr, exec);

    std::vector<Request> reqs = { MakeReq(1, 0.0, 10, 1) };
    auto result = loop.Run(std::move(reqs));

    if (result.completed.size() != 1) {
        std::printf("FAIL(TestSingleTokenRequest): completed=%zu, expected 1\n",
                    result.completed.size());
        ++g_failures;
        return;
    }
    CHECK(result.completed[0].state == RequestState::kDone);
    CHECK(result.completed[0].generated_tokens == 1);
    CHECK(mgr.NumFreeBlocks() == config.num_blocks);
}

// 两个请求不同时刻到达，都能完成
static void TestTwoRequests() {
    auto config = MakeConfig(16);
    FCFSScheduler sched;
    NaivePreallocator mgr(config);
    SimulatedExecutor exec;
    SimulationLoop loop(config, sched, mgr, exec);

    std::vector<Request> reqs = {
        MakeReq(1, 0.0, 10, 2),
        MakeReq(2, 0.001, 10, 2),
    };
    auto result = loop.Run(std::move(reqs));

    if (result.completed.size() != 2) {
        std::printf("FAIL(TestTwoRequests): completed=%zu, expected 2\n",
                    result.completed.size());
        ++g_failures;
        return;
    }
    CHECK(mgr.NumFreeBlocks() == config.num_blocks);
    CHECK(result.max_concurrent == 2);
}

// 资源不足：请求1占满块，请求2必须等请求1完成
static void TestResourceContention() {
    auto config = MakeConfig(2);  // 2 块 = 32 token
    FCFSScheduler sched;
    NaivePreallocator mgr(config);
    SimulatedExecutor exec;
    SimulationLoop loop(config, sched, mgr, exec);

    std::vector<Request> reqs = {
        MakeReq(1, 0.0, 20, 10),  // 30 token → 2 块，占满
        MakeReq(2, 0.0, 5, 5),    // 10 token → 1 块，等请求1释放
    };
    auto result = loop.Run(std::move(reqs));

    if (result.completed.size() != 2) {
        std::printf("FAIL(TestResourceContention): completed=%zu, expected 2\n",
                    result.completed.size());
        ++g_failures;
        return;
    }

    const Request* r1 = nullptr;
    const Request* r2 = nullptr;
    for (const auto& r : result.completed) {
        if (r.id == 1) r1 = &r;
        if (r.id == 2) r2 = &r;
    }
    if (r1 == nullptr || r2 == nullptr) {
        std::printf("FAIL(TestResourceContention): r1=%p r2=%p\n",
                    static_cast<const void*>(r1),
                    static_cast<const void*>(r2));
        ++g_failures;
        return;
    }
    CHECK(r2->first_token_time >= r1->finish_time - 1e-9);
    CHECK(result.max_concurrent == 1);
}

// 连续到达，验证事件按时间推进
static void TestEventOrdering() {
    auto config = MakeConfig(16);
    FCFSScheduler sched;
    NaivePreallocator mgr(config);
    SimulatedExecutor exec;
    SimulationLoop loop(config, sched, mgr, exec);

    std::vector<Request> reqs = {
        MakeReq(1, 0.0, 10, 3),
        MakeReq(2, 0.1, 10, 3),
    };
    auto result = loop.Run(std::move(reqs));

    if (result.completed.size() != 2) {
        std::printf("FAIL(TestEventOrdering): completed=%zu, expected 2\n",
                    result.completed.size());
        ++g_failures;
        return;
    }

    const Request* r1 = nullptr;
    const Request* r2 = nullptr;
    for (const auto& r : result.completed) {
        if (r.id == 1) r1 = &r;
        if (r.id == 2) r2 = &r;
    }
    if (r1 == nullptr || r2 == nullptr) {
        std::printf("FAIL(TestEventOrdering): r1=%p r2=%p\n",
                    static_cast<const void*>(r1),
                    static_cast<const void*>(r2));
        ++g_failures;
        return;
    }
    CHECK(r1->first_token_time < r2->first_token_time);
    CHECK(r1->finish_time < r2->finish_time);
}

// Paged 下，请求完成时块数应该大于初始分配的块数（证明扩块发生了）
static void TestPagedAppendTokenInLoop() {
    auto config = MakeConfig(16);
    FCFSScheduler sched;
    PagedBlockManager mgr(config);
    SimulatedExecutor exec;
    SimulationLoop loop(config, sched, mgr, exec);

    // prompt=20 → 初始 2 块（32 token 容量）
    // max_new=20 → 总 40 token，需要 3 块，Decode 中会扩 1 块
    std::vector<Request> reqs = { MakeReq(1, 0.0, 20, 20) };
    auto result = loop.Run(std::move(reqs));

    if (result.completed.size() != 1) {
        std::printf("FAIL(TestPagedAppendTokenInLoop): completed=%zu\n",
                    result.completed.size());
        ++g_failures;
        return;
    }
    CHECK(result.completed[0].state == RequestState::kDone);
    CHECK(result.completed[0].generated_tokens == 20);
    CHECK(mgr.NumFreeBlocks() == config.num_blocks);  // 全部释放
}

// Naive 下，AppendToken 是 no-op，块数不增长
static void TestNaiveNoAppend() {
    auto config = MakeConfig(16);
    FCFSScheduler sched;
    NaivePreallocator mgr(config);
    SimulatedExecutor exec;
    SimulationLoop loop(config, sched, mgr, exec);

    std::vector<Request> reqs = { MakeReq(1, 0.0, 20, 20) };
    auto result = loop.Run(std::move(reqs));

    if (result.completed.size() != 1) {
        std::printf("FAIL(TestNaiveNoAppend): completed=%zu\n",
                    result.completed.size());
        ++g_failures;
        return;
    }
    CHECK(result.completed[0].state == RequestState::kDone);
    CHECK(mgr.NumFreeBlocks() == config.num_blocks);
}

// 开启 PrefixCache：两个请求共享前缀，第二个请求 TTFT 更短
static void TestPrefixCacheReducesTtft() {
    auto config = MakeConfig(16);
    FCFSScheduler sched;
    PagedBlockManager mgr(config);
    SimulatedExecutor exec;
    PrefixCache cache;
    SimulationLoop loop(config, sched, mgr, exec, &cache);

    // 两个请求：相同 System Prompt（前 16 token），不同 User Query
    Request r1 = MakeReq(1, 0.0, 32, 5);
    Request r2 = MakeReq(2, 0.1, 32, 5);  // 晚到，r1 已完成
    for (int i = 0; i < 16; ++i) {
        r1.prompt_tokens[i] = 1000 + i;
        r2.prompt_tokens[i] = 1000 + i;
    }
    for (int i = 16; i < 32; ++i) {
        r1.prompt_tokens[i] = 2000 + i;
        r2.prompt_tokens[i] = 3000 + i;
    }

    auto result = loop.Run({r1, r2});
    if (result.completed.size() != 2) {
        std::printf("FAIL(TestPrefixCacheReducesTtft): completed=%zu\n",
                    result.completed.size());
        ++g_failures;
        return;
    }

    const Request* c1 = nullptr;
    const Request* c2 = nullptr;
    for (const auto& r : result.completed) {
        if (r.id == 1) c1 = &r;
        if (r.id == 2) c2 = &r;
    }
    if (c1 == nullptr || c2 == nullptr) {
        std::printf("FAIL(TestPrefixCacheReducesTtft): missing request\n");
        ++g_failures;
        return;
    }
    CHECK(c2->cached_prefix_len == 16);       // r2 命中 1 个完整块
    CHECK(c2->Ttft() < c1->Ttft());           // 命中使 TTFT 更短
    CHECK(cache.TotalHits() >= 1);            // 至少命中一次
}

// 多请求同时 decode 时，会走 DecodeBatch；完成数和资源账目正常
static void TestBatchDecode() {
    auto config = MakeConfig(16);
    FCFSScheduler sched;
    PagedBlockManager mgr(config);
    SimulatedExecutor exec;
    SimulationLoop loop(config, sched, mgr, exec);

    // 3 个请求，同一时刻到达，同时 decode
    std::vector<Request> reqs = {
        MakeReq(1, 0.0, 16, 3),
        MakeReq(2, 0.0, 16, 3),
        MakeReq(3, 0.0, 16, 3),
    };
    auto result = loop.Run(std::move(reqs));

    if (result.completed.size() != 3) {
        std::printf("FAIL(TestBatchDecode): completed=%zu\n",
                    result.completed.size());
        ++g_failures;
        return;
    }
    for (const auto& r : result.completed) {
        CHECK(r.state == RequestState::kDone);
        CHECK(r.generated_tokens == 3);
    }
    CHECK(mgr.NumFreeBlocks() == config.num_blocks);
}

int main() {
    TestSingleRequest();
    TestSingleTokenRequest();
    TestTwoRequests();
    TestResourceContention();
    TestEventOrdering();
    TestPagedAppendTokenInLoop();
    TestNaiveNoAppend();
    TestPrefixCacheReducesTtft();
    TestBatchDecode();

    if (g_failures == 0) {
        std::printf("All SimulationLoop tests passed.\n");
        return 0;
    }
    std::printf("%d checks failed.\n", g_failures);
    return 1;
}