#include "PrefixCache.h"

#include <cstdio>
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

static std::vector<TokenId> MakePrompt(std::initializer_list<TokenId> tokens) {
    return std::vector<TokenId>(tokens);
}

// 空缓存 Match 返回空
static void TestEmptyCache() {
    PrefixCache cache;
    auto prompt = MakePrompt({1, 2, 3, 4});
    auto result = cache.Match(prompt, 4);
    CHECK(result.matched_blocks.empty());
    CHECK(result.matched_tokens == 0);
}

// 插入一个 prompt，同 prompt 再次 Match 命中
static void TestInsertThenMatch() {
    PrefixCache cache;
    auto prompt = MakePrompt({1, 2, 3, 4});
    cache.Insert(prompt, {100}, 4);

    auto result = cache.Match(prompt, 4);
    CHECK(result.matched_blocks.size() == 1);
    CHECK(result.matched_blocks[0] == 100);
    CHECK(result.matched_tokens == 4);
}

// 修正后的共享前缀测试
static void TestSharedPrefixCorrect() {
    PrefixCache cache;
    auto p1 = MakePrompt({1, 2, 3, 4, 5, 6, 7, 8});
    cache.Insert(p1, {10, 11}, 4);

    auto p2 = MakePrompt({1, 2, 3, 4, 20, 21, 22, 23});
    cache.Insert(p2, {10, 20}, 4);  // 第 0 块复用 10，第 1 块新建 20

    // p1 Match 仍然命中 2 块
    auto r1 = cache.Match(p1, 4);
    CHECK(r1.matched_blocks.size() == 2);
    CHECK(r1.matched_blocks[0] == 10);
    CHECK(r1.matched_blocks[1] == 11);

    // p2 Match 命中 2 块（第 0 块共享，第 1 块是新节点但内容一致）
    auto r2 = cache.Match(p2, 4);
    CHECK(r2.matched_blocks.size() == 2);
    CHECK(r2.matched_blocks[0] == 10);
    CHECK(r2.matched_blocks[1] == 20);
}

// 不同前缀不命中
static void TestDifferentPrefix() {
    PrefixCache cache;
    auto p1 = MakePrompt({1, 2, 3, 4});
    cache.Insert(p1, {100}, 4);

    auto p2 = MakePrompt({9, 9, 9, 9});
    auto result = cache.Match(p2, 4);
    CHECK(result.matched_blocks.empty());
    CHECK(result.matched_tokens == 0);
}

// 不完整 block 不参与匹配（prompt 长度不是 block_size 整数倍）
static void TestPartialBlockNotMatched() {
    PrefixCache cache;
    // p1 长度 6，block_size=4，只有第 0 块完整
    auto p1 = MakePrompt({1, 2, 3, 4, 5, 6});
    cache.Insert(p1, {100}, 4);

    auto result = cache.Match(p1, 4);
    CHECK(result.matched_blocks.size() == 1);  // 只有第 0 块
    CHECK(result.matched_tokens == 4);          // 不是 6
}

// Release 减引用，多次 Release 后再 Match 仍能命中（节点不删）
static void TestRelease() {
    PrefixCache cache;
    auto p = MakePrompt({1, 2, 3, 4});
    cache.Insert(p, {100}, 4);
    cache.Release(p, 4);

    // 节点还在，Match 仍然命中（第一版不删节点）
    auto result = cache.Match(p, 4);
    CHECK(result.matched_blocks.size() == 1);
    CHECK(result.matched_blocks[0] == 100);
}

// 统计计数
static void TestStats() {
    PrefixCache cache;
    auto p1 = MakePrompt({1, 2, 3, 4});
    cache.Insert(p1, {100}, 4);

    CHECK(cache.TotalQueries() == 0);
    CHECK(cache.TotalHits() == 0);

    cache.Match(p1, 4);                       // 命中
    cache.Match(MakePrompt({9, 9, 9, 9}), 4); // 未命中

    CHECK(cache.TotalQueries() == 2);
    CHECK(cache.TotalHits() == 1);
}

int main() {
    TestEmptyCache();
    TestInsertThenMatch();
    TestSharedPrefixCorrect();
    TestDifferentPrefix();
    TestPartialBlockNotMatched();
    TestRelease();
    TestStats();

    if (g_failures == 0) {
        std::printf("All PrefixCache tests passed.\n");
        return 0;
    }
    std::printf("%d checks failed.\n", g_failures);
    return 1;
}