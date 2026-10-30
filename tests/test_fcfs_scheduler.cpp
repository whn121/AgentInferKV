#include "FCFSScheduler.h"
#include "NaivePreallocator.h"

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

static ModelConfig SmallConfig() {
    ModelConfig c;
    c.block_size = 16;
    c.num_blocks = 4;  // 只能容 64 token
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

// 两个请求都能装下：1 块 + 1 块 = 2 块，总 4 块
static void TestAllFit() {
    NaivePreallocator m(SmallConfig());
    FCFSScheduler s;

    std::vector<Request> waiting = {
        MakeReq(2, 0.02, 10, 5),  // 15 token -> 1 块
        MakeReq(1, 0.01, 10, 5),  // 15 token -> 1 块
    };
    std::vector<Request> running;

    BlockBudget budget = m.MakeBudget();
    auto out = s.Schedule(waiting, running, budget);

    CHECK(out.scheduled_ids.size() == 2);
    CHECK(out.scheduled_ids[0] == 1);
    CHECK(out.scheduled_ids[1] == 2);
    CHECK(budget.free_blocks == 2);  // 4 - 1 - 1 = 2
}

// 请求1 占 3 块后只剩 1 块，请求2 需要 2 块，失败，FCFS 停住
static void TestPartialFit() {
    NaivePreallocator m(SmallConfig());
    FCFSScheduler s;

    std::vector<Request> waiting = {
        MakeReq(1, 0.01, 40, 4),   // 44 token -> 3 块
        MakeReq(2, 0.02, 20, 8),   // 28 token -> 2 块
    };
    std::vector<Request> running;

    BlockBudget budget = m.MakeBudget();
    auto out = s.Schedule(waiting, running, budget);

    CHECK(out.scheduled_ids.size() == 1);
    CHECK(out.scheduled_ids[0] == 1);
    CHECK(budget.free_blocks == 1);  // 4 - 3 = 1，请求2 需要 2 块被拒
}

// 同一 arrival_time，按 id 升序
static void TestDeterministicTieBreak() {
    NaivePreallocator m(SmallConfig());
    FCFSScheduler s;

    std::vector<Request> waiting = {
        MakeReq(3, 0.01, 10, 0),  // 10 token -> 1 块
        MakeReq(1, 0.01, 10, 0),
        MakeReq(2, 0.01, 10, 0),
    };
    std::vector<Request> running;

    BlockBudget budget = m.MakeBudget();
    auto out = s.Schedule(waiting, running, budget);

    CHECK(out.scheduled_ids.size() == 3);
    CHECK(out.scheduled_ids[0] == 1);
    CHECK(out.scheduled_ids[1] == 2);
    CHECK(out.scheduled_ids[2] == 3);
}

// 预算不够时，Scheduler 不再返回“假承诺”
static void TestBudgetIsDebited() {
    NaivePreallocator m(SmallConfig());
    FCFSScheduler s;

    // 4 块，两个请求各要 3 块：只能上第一个
    std::vector<Request> waiting = {
        MakeReq(1, 0.01, 40, 4),  // 44 token -> 3 块
        MakeReq(2, 0.02, 40, 4),  // 44 token -> 3 块
    };
    std::vector<Request> running;

    BlockBudget budget = m.MakeBudget();
    auto out = s.Schedule(waiting, running, budget);

    CHECK(out.scheduled_ids.size() == 1);
    CHECK(out.scheduled_ids[0] == 1);
    CHECK(budget.free_blocks == 1);
}

int main() {
    TestAllFit();
    TestPartialFit();
    TestDeterministicTieBreak();
    TestBudgetIsDebited();

    if (g_failures == 0) {
        std::printf("All FCFSScheduler tests passed.\n");
        return 0;
    }
    std::printf("%d checks failed.\n", g_failures);
    return 1;
}