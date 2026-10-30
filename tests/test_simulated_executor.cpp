#include "SimulatedExecutor.h"

#include <cmath>
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

// 浮点比较：允许 1e-9 的误差
static bool NearlyEqual(double a, double b) {
    return std::fabs(a - b) < 1e-9;
}

static Request MakeReq(int prompt_len, int max_new) {
    Request r;
    r.prompt_tokens.resize(static_cast<std::size_t>(prompt_len));
    r.max_new_tokens = max_new;
    return r;
}

// Prefill 耗时 = base + per_token * prompt_len
static void TestPrefillFormula() {
    ExecutorConfig cfg;
    cfg.base_prefill_s = 0.005;
    cfg.per_token_prefill_s = 0.0001;
    SimulatedExecutor ex(cfg);
    BlockTableView view;

    Request r0 = MakeReq(0, 1);
    CHECK(NearlyEqual(ex.Prefill(r0, view), 0.005));

    Request r100 = MakeReq(100, 1);
    CHECK(NearlyEqual(ex.Prefill(r100, view), 0.005 + 0.0001 * 100));

    Request r200 = MakeReq(200, 1);
    CHECK(NearlyEqual(ex.Prefill(r200, view), 0.005 + 0.0001 * 200));
}

// Decode 耗时固定，与 prompt 长度无关
static void TestDecodeIsConstant() {
    ExecutorConfig cfg;
    cfg.base_decode_s = 0.0005;
    cfg.per_token_decode_s = 0.00002;
    SimulatedExecutor ex(cfg);
    BlockTableView view;

    Request r10 = MakeReq(10, 5);
    Request r500 = MakeReq(500, 5);
    double d1 = ex.Decode(r10, view);
    double d2 = ex.Decode(r500, view);
    CHECK(NearlyEqual(d1, d2));
    CHECK(NearlyEqual(d1, 0.0005 + 0.00002));
}

// 耗时随 prompt 长度单调递增
static void TestPrefillMonotonic() {
    SimulatedExecutor ex;
    BlockTableView view;

    double prev = -1.0;
    for (int len = 0; len <= 1000; len += 100) {
        Request r = MakeReq(len, 1);
        double cur = ex.Prefill(r, view);
        CHECK(cur > prev);
        prev = cur;
    }
}

// 参数为 0 时也不应返回负值
static void TestNonNegative() {
    ExecutorConfig cfg;
    cfg.base_prefill_s = 0.0;
    cfg.per_token_prefill_s = 0.0;
    cfg.base_decode_s = 0.0;
    cfg.per_token_decode_s = 0.0;
    SimulatedExecutor ex(cfg);
    BlockTableView view;

    CHECK(ex.Prefill(MakeReq(100, 1), view) == 0.0);
    CHECK(ex.Decode(MakeReq(100, 1), view) == 0.0);
}

int main() {
    TestPrefillFormula();
    TestDecodeIsConstant();
    TestPrefillMonotonic();
    TestNonNegative();

    if (g_failures == 0) {
        std::printf("All SimulatedExecutor tests passed.\n");
        return 0;
    }
    std::printf("%d checks failed.\n", g_failures);
    return 1;
}