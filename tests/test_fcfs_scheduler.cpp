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

static void TestAllFit() {
    NaivePreallocator m(SmallConfig());
    FCFSScheduler s;

    std::vector<Request> waiting = {
        MakeReq(2, 0.02, 10, 5),  // 1 块
        MakeReq(1, 0.01, 10, 5),  // 1 块
    };
    std::vector<Request> running;

    auto out = s.Schedule(waiting, running, m);
    // 按 arrival 排序后应为 1, 2
    CHECK(out.scheduled_ids.size() == 2);
    CHECK(out.scheduled_ids[0] == 1);
    CHECK(out.scheduled_ids[1] == 2);
}

static void TestPartialFit() {
    NaivePreallocator m(SmallConfig());
    FCFSScheduler s;

    // 三个请求，每个 1 块，总共只能容 4 块，但前两个都能过。
    // 第三个来之前先占两块，第三个也能过。
    // 用一个更极端的：请求 1 就要 3 块，请求 2 要 2 块
    std::vector<Request> waiting = {
        MakeReq(1, 0.01, 40, 4),   // 44 token -> 3 块
        MakeReq(2, 0.02, 20, 8),   // 28 token -> 2 块
    };
    std::vector<Request> running;

    auto out = s.Schedule(waiting, running, m);
    // 请求 1 占 3 块后只剩 1 块，请求 2 需要 2 块，不够。
    // FCFS 严格语义：请求 2 不插队。
    CHECK(out.scheduled_ids.size() == 1);
    CHECK(out.scheduled_ids[0] == 1);
}

static void TestDeterministicTieBreak() {
    NaivePreallocator m(SmallConfig());
    FCFSScheduler s;

    // 同一 arrival_time，按 id 升序
    std::vector<Request> waiting = {
        MakeReq(3, 0.01, 10, 0),
        MakeReq(1, 0.01, 10, 0),
        MakeReq(2, 0.01, 10, 0),
    };
    std::vector<Request> running;

    auto out = s.Schedule(waiting, running, m);
    CHECK(out.scheduled_ids.size() == 3);
    CHECK(out.scheduled_ids[0] == 1);
    CHECK(out.scheduled_ids[1] == 2);
    CHECK(out.scheduled_ids[2] == 3);
}

int main() {
    TestAllFit();
    TestPartialFit();
    TestDeterministicTieBreak();

    if (g_failures == 0) {
        std::printf("All FCFSScheduler tests passed.\n");
        return 0;
    }
    std::printf("%d checks failed.\n", g_failures);
    return 1;
}