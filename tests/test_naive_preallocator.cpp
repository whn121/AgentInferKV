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
    c.num_blocks = 8;
    return c;
}

static Request MakeReq(RequestId id, int prompt_len, int max_new) {
    Request r;
    r.id = id;
    r.prompt_tokens.resize(static_cast<std::size_t>(prompt_len));
    r.max_new_tokens = max_new;
    return r;
}

static void TestBasicAllocate() {
    NaivePreallocator m(SmallConfig());
    CHECK(m.NumFreeBlocks() == 8);

    // prompt=32, max_new=8 -> total=40 -> ceil(40/16)=3 块
    Request r = MakeReq(1, 32, 8);
    CHECK(m.CanAllocate(r));
    auto view = m.Allocate(r);
    CHECK(view.NumBlocks() == 3);
    CHECK(m.NumFreeBlocks() == 5);

    // 同一请求不能重复分配
    CHECK(!m.CanAllocate(r));

    m.Free(1);
    CHECK(m.NumFreeBlocks() == 8);
}

static void TestExhaustion() {
    NaivePreallocator m(SmallConfig());

    // total=128 -> 8 块，正好吃完
    Request r1 = MakeReq(1, 100, 28);
    CHECK(m.CanAllocate(r1));
    m.Allocate(r1);
    CHECK(m.NumFreeBlocks() == 0);

    // 再来一个请求，没块了
    Request r2 = MakeReq(2, 10, 5);  // 需要 1 块
    CHECK(!m.CanAllocate(r2));
}

static void TestFreeUnknownIdIsNoOp() {
    NaivePreallocator m(SmallConfig());
    m.Free(999);
    CHECK(m.NumFreeBlocks() == 8);
}

static void TestDoubleFree() {
    NaivePreallocator m(SmallConfig());
    Request r = MakeReq(1, 16, 0);  // 需要 1 块
    m.Allocate(r);
    CHECK(m.NumFreeBlocks() == 7);
    m.Free(1);
    CHECK(m.NumFreeBlocks() == 8);
    m.Free(1);  // 第二次 Free 不能重复还块
    CHECK(m.NumFreeBlocks() == 8);
}

int main() {
    TestBasicAllocate();
    TestExhaustion();
    TestFreeUnknownIdIsNoOp();
    TestDoubleFree();

    if (g_failures == 0) {
        std::printf("All NaivePreallocator tests passed.\n");
        return 0;
    }
    std::printf("%d checks failed.\n", g_failures);
    return 1;
}