#include "FCFSScheduler.h"

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

// 模拟 Naive 的估算：按 TotalTokens 向上取整
static int NaiveNeed(const Request& r, int block_size) {
    int t = r.TotalTokens();
    if (t <= 0) return 0;
    return (t + block_size - 1) / block_size;
}

static Request MakeReq(RequestId id, double arrival, int prompt_len, int max_new) {
    Request r;
    r.id = id;
    r.arrival_time = arrival;
    r.prompt_tokens.resize(static_cast<std::size_t>(prompt_len));
    r.max_new_tokens = max_new;
    return r;
}

static BlockBudget MakeBudget(int free_blocks) {
    BlockBudget b;
    b.free_blocks = free_blocks;
    b.blocks_needed = [](const Request& r) { return NaiveNeed(r, 16); };
    return b;
}

static void TestAllFit() {
    FCFSScheduler s;
    std::vector<Request> waiting = {
        MakeReq(2, 0.02, 10, 5),  // 15 token -> 1 块
        MakeReq(1, 0.01, 10, 5),
    };
    std::vector<Request> running;

    BlockBudget budget = MakeBudget(4);
    auto out = s.Schedule(waiting, running, budget);

    CHECK(out.scheduled_ids.size() == 2);
    CHECK(out.scheduled_ids[0] == 1);
    CHECK(out.scheduled_ids[1] == 2);
    CHECK(budget.free_blocks == 2);
}

static void TestPartialFit() {
    FCFSScheduler s;
    std::vector<Request> waiting = {
        MakeReq(1, 0.01, 40, 4),   // 44 token -> 3 块
        MakeReq(2, 0.02, 20, 8),   // 28 token -> 2 块
    };
    std::vector<Request> running;

    BlockBudget budget = MakeBudget(4);
    auto out = s.Schedule(waiting, running, budget);

    CHECK(out.scheduled_ids.size() == 1);
    CHECK(out.scheduled_ids[0] == 1);
    CHECK(budget.free_blocks == 1);
}

static void TestDeterministicTieBreak() {
    FCFSScheduler s;
    std::vector<Request> waiting = {
        MakeReq(3, 0.01, 10, 0),
        MakeReq(1, 0.01, 10, 0),
        MakeReq(2, 0.01, 10, 0),
    };
    std::vector<Request> running;

    BlockBudget budget = MakeBudget(4);
    auto out = s.Schedule(waiting, running, budget);

    CHECK(out.scheduled_ids.size() == 3);
    CHECK(out.scheduled_ids[0] == 1);
    CHECK(out.scheduled_ids[1] == 2);
    CHECK(out.scheduled_ids[2] == 3);
}

static void TestBudgetIsDebited() {
    FCFSScheduler s;
    std::vector<Request> waiting = {
        MakeReq(1, 0.01, 40, 4),  // 3 块
        MakeReq(2, 0.02, 40, 4),  // 3 块
    };
    std::vector<Request> running;

    BlockBudget budget = MakeBudget(4);
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