#include "PagedBlockManager.h"

#include <atomic>
#include <cstdio>
#include <thread>
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

static ModelConfig SmallConfig()
{
    ModelConfig c;
    c.block_size = 16;
    c.num_blocks = 8;
    return c;
}

static Request MakeReq(RequestId id, int prompt_len, int max_new)
{
    Request r;
    r.id = id;
    r.prompt_tokens.resize(static_cast<std::size_t>(prompt_len));
    r.max_new_tokens = max_new;
    return r;
}

// ① 多线程反复 Retain/Release 同一个块：计数必须能回到持有者那一份，
//    既不能泄漏（回池少），也不能多还（回池多）。
static void TestConcurrentRetainRelease()
{
    PagedBlockManager m(SmallConfig());
    Request r = MakeReq(1, 16, 8);          // 1 块
    auto view = m.Allocate(r);
    CHECK(view.NumBlocks() == 1);
    const BlockId b = view.logical_to_physical[0];
    CHECK(m.NumFreeBlocks() == 7);

    constexpr int kThreads = 8;
    constexpr int kIters = 20000;
    std::atomic<int> retained{0};

    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t)
    {
        threads.emplace_back([&]
        {
            for (int i = 0; i < kIters; ++i)
            {
                // 请求本身持有一份，所以这个块永远不会归零，Retain 必须一直成功
                if (m.RetainIfAlive(b)) { ++retained; m.ReleaseSharedBlock(b); }
            }
        });
    }
    for (auto& th : threads) th.join();

    CHECK(retained.load() == kThreads * kIters);
    CHECK(m.IsBlockAlive(b));               // 请求那一份还在

    m.Free(1);
    CHECK(!m.IsBlockAlive(b));
    CHECK(m.NumFreeBlocks() == 8);          // 恰好回池一次
}

// ② 两个线程同时对同一个请求 Free：只能有一个真正回收。
static void TestConcurrentDoubleFree()
{
    PagedBlockManager m(SmallConfig());
    Request r = MakeReq(2, 48, 8);          // 3 块
    auto view = m.Allocate(r);
    CHECK(view.NumBlocks() == 3);
    CHECK(m.NumFreeBlocks() == 5);

    std::thread t1([&] { m.Free(2); });
    std::thread t2([&] { m.Free(2); });
    t1.join();
    t2.join();

    CHECK(m.NumFreeBlocks() == 8);          // 多还一次就会变成 11
    CHECK(!m.IsBlockAlive(view.logical_to_physical[0]));
}

// ③ 块已回池后，RetainIfAlive 必须拒绝，而不是把已回收的块又“救活”。
static void TestRetainAfterFree()
{
    PagedBlockManager m(SmallConfig());
    Request r = MakeReq(3, 16, 8);
    auto view = m.Allocate(r);
    const BlockId b = view.logical_to_physical[0];

    m.Free(3);
    CHECK(!m.RetainIfAlive(b));
    CHECK(!m.IsBlockAlive(b));
    CHECK(m.NumFreeBlocks() == 8);
}

// ④ 一个线程反复 Free、另一个线程反复 RetainIfAlive：无论交错顺序如何，
//    都不允许出现负计数或重复回池（回池总数必须恰好等于块数）。
static void TestRetainFreeRace()
{
    PagedBlockManager m(SmallConfig());
    Request r = MakeReq(4, 16, 8);
    auto view = m.Allocate(r);
    const BlockId b = view.logical_to_physical[0];

    std::thread freer([&] { m.Free(4); });
    std::thread retainer([&]
    {
        for (int i = 0; i < 2000; ++i)
        {
            if (m.RetainIfAlive(b)) m.ReleaseSharedBlock(b);
        }
    });
    freer.join();
    retainer.join();

    CHECK(m.NumFreeBlocks() == 8);
    CHECK(!m.IsBlockAlive(b));
}

int main()
{
    TestConcurrentRetainRelease();
    TestConcurrentDoubleFree();
    TestRetainAfterFree();
    TestRetainFreeRace();

    if (g_failures == 0) std::printf("all concurrent-release tests passed\n");
    return g_failures == 0 ? 0 : 1;
}
