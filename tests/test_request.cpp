#include "Interfaces.h"

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

// 正常路径：WAITING -> PREFILL -> DECODE -> DONE
static void TestHappyPath() {
    Request r;
    r.id = 1;
    r.arrival_time = 0.0;
    r.prompt_tokens = {1, 2, 3};
    r.max_new_tokens = 2;

    CHECK(r.state == RequestState::kWaiting);
    CHECK(!IsTerminal(r.state));
    CHECK(r.PromptLen() == 3);
    CHECK(r.TotalTokens() == 5);
    CHECK(r.Ttft() == -1.0);
    CHECK(r.Latency() == -1.0);

    // 手动推进状态，模拟 Scheduler 的调用顺序。
    r.state = RequestState::kPrefill;
    CHECK(!IsTerminal(r.state));

    // 首 token 产出
    r.state = RequestState::kDecode;
    r.generated_tokens = 1;
    r.first_token_time = 0.020;
    CHECK(r.Ttft() > 0.019 && r.Ttft() < 0.021);
    CHECK(r.Latency() == -1.0);

    // 生成完成
    r.state = RequestState::kDone;
    r.generated_tokens = 2;
    r.finish_time = 0.035;
    CHECK(IsTerminal(r.state));
    CHECK(r.Latency() > 0.034 && r.Latency() < 0.036);
}

// 非法转换必须被 CanTransition 拒绝
static void TestIllegalTransitions() {
    // 不能跳过 Prefill 直接 Decode
    CHECK(!CanTransition(RequestState::kWaiting, RequestState::kDecode));
    CHECK(!CanTransition(RequestState::kWaiting, RequestState::kDone));
    // 不能从 Prefill 直接 Done
    CHECK(!CanTransition(RequestState::kPrefill, RequestState::kDone));
    // 终态不能再转换
    CHECK(!CanTransition(RequestState::kDone, RequestState::kWaiting));
    CHECK(!CanTransition(RequestState::kDone, RequestState::kCancelled));
    CHECK(!CanTransition(RequestState::kCancelled, RequestState::kPrefill));
}

// 合法转换必须全部允许
static void TestLegalTransitions() {
    CHECK(CanTransition(RequestState::kWaiting, RequestState::kPrefill));
    CHECK(CanTransition(RequestState::kWaiting, RequestState::kCancelled));
    CHECK(CanTransition(RequestState::kPrefill, RequestState::kDecode));
    CHECK(CanTransition(RequestState::kPrefill, RequestState::kCancelled));
    CHECK(CanTransition(RequestState::kDecode, RequestState::kDone));
    CHECK(CanTransition(RequestState::kDecode, RequestState::kCancelled));
}

// 从未产出首 token 就取消，Ttft 保持 -1
static void TestCancelBeforeFirstToken() {
    Request r;
    r.arrival_time = 1.0;
    r.state = RequestState::kCancelled;
    r.finish_time = 1.5;
    CHECK(r.Ttft() == -1.0);
    CHECK(r.Latency() > 0.49 && r.Latency() < 0.51);
}

int main() {
    TestHappyPath();
    TestIllegalTransitions();
    TestLegalTransitions();
    TestCancelBeforeFirstToken();

    if (g_failures == 0) {
        std::printf("All Request tests passed.\n");
        return 0;
    }
    std::printf("%d checks failed.\n", g_failures);
    return 1;
}