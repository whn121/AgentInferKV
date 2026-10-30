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

// 正常路径：WAITING -> PREFILL -> DECODE -> DONE（max_new_tokens > 1）
static void TestHappyPath() {
    Request r;
    r.arrival_time = 0.0;
    r.max_new_tokens = 3;

    CHECK(r.state == RequestState::kWaiting);
    CHECK(r.StartPrefill(0.01));
    CHECK(r.state == RequestState::kPrefill);

    CHECK(r.FinishPrefill(0.02));
    CHECK(r.state == RequestState::kDecode);
    CHECK(r.generated_tokens == 1);
    CHECK(r.Ttft() > 0.019 && r.Ttft() < 0.021);

    CHECK(r.DecodeOneToken(0.03));
    CHECK(r.state == RequestState::kDecode);
    CHECK(r.generated_tokens == 2);

    CHECK(r.DecodeOneToken(0.04));
    CHECK(r.state == RequestState::kDone);
    CHECK(r.generated_tokens == 3);
    CHECK(r.Latency() > 0.039 && r.Latency() < 0.041);

    // 终态不能再推进
    CHECK(!r.DecodeOneToken(0.05));
    CHECK(!r.Cancel(0.05));
}

// max_new_tokens == 1：FinishPrefill 经 DECODE 直接转 DONE
static void TestMaxNewTokensOne() {
    Request r;
    r.arrival_time = 0.0;
    r.max_new_tokens = 1;

    CHECK(r.StartPrefill(0.01));
    CHECK(r.FinishPrefill(0.02));
    CHECK(r.state == RequestState::kDone);
    CHECK(r.generated_tokens == 1);
    CHECK(r.Ttft() > 0.019 && r.Ttft() < 0.021);
    CHECK(r.Latency() > 0.019 && r.Latency() < 0.021);
}

// 非法顺序：未 StartPrefill 就 FinishPrefill；未 FinishPrefill 就 Decode
static void TestIllegalSequence() {
    Request r;
    r.max_new_tokens = 3;

    CHECK(!r.FinishPrefill(0.01));   // 还在 WAITING
    CHECK(!r.DecodeOneToken(0.01));  // 还在 WAITING
    CHECK(r.state == RequestState::kWaiting);
    CHECK(r.first_token_time == -1.0);  // 非法调用不能写时间戳

    CHECK(r.StartPrefill(0.02));
    CHECK(!r.StartPrefill(0.03));    // 重复 StartPrefill
    CHECK(!r.DecodeOneToken(0.03));  // 还在 PREFILL
    CHECK(r.state == RequestState::kPrefill);
}

// Cancel 各阶段都能取消，且都记录 finish_time
static void TestCancel() {
    // WAITING 阶段取消
    Request r1;
    r1.max_new_tokens = 3;
    CHECK(r1.Cancel(0.01));
    CHECK(r1.state == RequestState::kCancelled);
    CHECK(r1.Ttft() == -1.0);   // 未产出首 token
    CHECK(!r1.StartPrefill(0.02));

    // PREFILL 阶段取消
    Request r2;
    r2.max_new_tokens = 3;
    r2.StartPrefill(0.01);
    CHECK(r2.Cancel(0.02));
    CHECK(r2.state == RequestState::kCancelled);
    CHECK(r2.Ttft() == -1.0);

    // DECODE 阶段取消
    Request r3;
    r3.max_new_tokens = 3;
    r3.StartPrefill(0.01);
    r3.FinishPrefill(0.02);
    CHECK(r3.Cancel(0.03));
    CHECK(r3.state == RequestState::kCancelled);
    CHECK(r3.Ttft() > 0.019 && r3.Ttft() < 0.021);
    CHECK(r3.Latency() > 0.029 && r3.Latency() < 0.031);
}

// 重复 Cancel 不改变状态，不覆盖 finish_time
static void TestDoubleCancel() {
    Request r;
    r.max_new_tokens = 3;
    CHECK(r.Cancel(0.01));
    CHECK(r.finish_time > 0.009 && r.finish_time < 0.011);

    CHECK(!r.Cancel(0.99));  // 终态拒绝
    CHECK(r.finish_time > 0.009 && r.finish_time < 0.011);  // 时间戳没被覆盖
}

int main() {
    TestHappyPath();
    TestMaxNewTokensOne();
    TestIllegalSequence();
    TestCancel();
    TestDoubleCancel();

    if (g_failures == 0) {
        std::printf("All Request tests passed.\n");
        return 0;
    }
    std::printf("%d checks failed.\n", g_failures);
    return 1;
}