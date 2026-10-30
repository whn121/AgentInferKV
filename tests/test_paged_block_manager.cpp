#include "PagedBlockManager.h"

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

// Allocate 只按 prompt_len 分块，不预分配 max_new 部分
static void TestAllocateOnlyPrompt() {
    PagedBlockManager m(SmallConfig());
    Request r = MakeReq(1, 20, 100);  // prompt=20 → 2 块；max_new=100 不预分配
    CHECK(m.CanAllocate(r));
    auto view = m.Allocate(r);
    CHECK(view.NumBlocks() == 2);
    CHECK(m.NumFreeBlocks() == 6);
}

// AppendToken 在需要时扩块
static void TestAppendToken() {
    PagedBlockManager m(SmallConfig());
    Request r = MakeReq(1, 20, 50);
    m.Allocate(r);

    BlockTableView view;
    // 生成到 25 token，还在第 2 块内，不扩
    CHECK(m.AppendToken(1, 25, view));
    CHECK(view.NumBlocks() == 2);
    CHECK(m.NumFreeBlocks() == 6);

    // 生成到 33 token，需要第 3 块
    CHECK(m.AppendToken(1, 33, view));
    CHECK(view.NumBlocks() == 3);
    CHECK(m.NumFreeBlocks() == 5);

    // 生成到 40 token，还在第 3 块内
    CHECK(m.AppendToken(1, 40, view));
    CHECK(view.NumBlocks() == 3);
}

// 块不够时 AppendToken 返回 false，不部分分配
static void TestAppendTokenOutOfBlocks() {
    auto config = SmallConfig();
    config.num_blocks = 2;
    PagedBlockManager m(config);

    Request r = MakeReq(1, 20, 50);  // 2 块，占满
    m.Allocate(r);
    CHECK(m.NumFreeBlocks() == 0);

    BlockTableView view;
    CHECK(!m.AppendToken(1, 33, view));  // 需要第 3 块，没有
    CHECK(m.NumFreeBlocks() == 0);       // 没有部分分配
}

// 未知 id 的 AppendToken 返回 false
static void TestAppendTokenUnknownId() {
    PagedBlockManager m(SmallConfig());
    BlockTableView view;
    CHECK(!m.AppendToken(999, 33, view));
}

// Free 后块回收，TailUnusedCapacity 归零
static void TestFreeAndTail() {
    PagedBlockManager m(SmallConfig());
    Request r1 = MakeReq(1, 20, 50);  // 2 块，20 token，最后一块用 4/16，tail=12
    m.Allocate(r1);
    CHECK(m.TailUnusedCapacity() == 12);

    BlockTableView v;
    CHECK(m.AppendToken(1, 25, v));  // 25 token，tail=7
    CHECK(m.TailUnusedCapacity() == 7);

    m.Free(1);
    CHECK(m.NumFreeBlocks() == 8);
    CHECK(m.TailUnusedCapacity() == 0);
}

// 同样块预算下 Paged 能塞更多请求
static void TestMoreConcurrentThanNaive() {
    auto config = SmallConfig();  // 8 块
    PagedBlockManager paged(config);

    // 4 个请求，每个 prompt=20（2 块），max_new=100（Naive 需要 8 块）
    for (RequestId id = 1; id <= 4; ++id) {
        Request r = MakeReq(id, 20, 100);
        CHECK(paged.CanAllocate(r));
        CHECK(paged.Allocate(r).NumBlocks() == 2);
    }
    CHECK(paged.NumFreeBlocks() == 0);
    // Naive 下同样的 4 个请求：每个需 ceil(120/16)=8 块，只能上 1 个。
    // Paged 上 4 个，Naive 上 1 个。
}

static void TestSharedBlockRefCount() {
    PagedBlockManager m(SmallConfig());  // 8 块

    Request r = MakeReq(1, 16, 0);
    auto view = m.Allocate(r);  // ceil(16/16)=1 块
    CHECK(view.NumBlocks() == 1);
    CHECK(m.NumFreeBlocks() == 7);
    BlockId shared = view.PhysicalBlock(0);

    m.RetainSharedBlock(shared);   // ref: 1 -> 2
    m.Free(1);                      // ref: 2 -> 1
    CHECK(m.NumFreeBlocks() == 7);

    m.ReleaseSharedBlock(shared);   // ref: 1 -> 0
    CHECK(m.NumFreeBlocks() == 8);
}

int main() {
    TestAllocateOnlyPrompt();
    TestAppendToken();
    TestAppendTokenOutOfBlocks();
    TestAppendTokenUnknownId();
    TestFreeAndTail();
    TestMoreConcurrentThanNaive();
    TestSharedBlockRefCount();

    if (g_failures == 0) {
        std::printf("All PagedBlockManager tests passed.\n");
        return 0;
    }
    std::printf("%d checks failed.\n", g_failures);
    return 1;
}