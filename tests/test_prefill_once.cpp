// 回归测试：每个请求的 Prefill 只能执行一次。
// 历史 bug：SimulationLoop 里 prefill_cost 和 next_event_time 各调了一次
// executor_.Prefill，模拟执行器无副作用所以测试发现不了，真实 executor 会泄漏 seq/KV。
#include "FCFSScheduler.h"
#include "Interfaces.h"
#include "PagedBlockManager.h"
#include "SimulationLoop.h"

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

namespace
{
    struct CountingExecutor : IInferenceExecutor
    {
        int prefill_calls = 0;
        int decode_calls = 0;
        int batch_calls = 0;
        int batch_requests = 0;

        double Prefill(const Request& req, const BlockTableView&) override
        {
            ++prefill_calls;
            return 0.001 + 0.0001 * req.PromptLen();
        }
        double Decode(const Request&, const BlockTableView&) override
        {
            ++decode_calls;
            return 0.001;
        }
        std::vector<double> DecodeBatch(const std::vector<const Request*>& reqs,
                                        const std::vector<BlockTableView>&) override
        {
            ++batch_calls;
            batch_requests += static_cast<int>(reqs.size());
            return std::vector<double>(reqs.size(), 0.001);
        }
        void Release(RequestId) override {}

        int ComputeSteps() const { return prefill_calls + decode_calls + batch_calls; }
    };

    Request MakeReq(RequestId id, double arrival, int prompt_len, int max_new)
    {
        Request r;
        r.id = id;
        r.arrival_time = arrival;
        r.prompt_tokens.resize(static_cast<std::size_t>(prompt_len));
        r.max_new_tokens = max_new;
        return r;
    }

    ModelConfig MakeConfig()
    {
        ModelConfig c;
        c.block_size = 16;
        c.num_blocks = 64;
        return c;
    }
}

// 单请求 max_new=n：compute 次数必须恰好是 n（1 次 Prefill + n-1 次 Decode）
static void TestSingleRequestStepCount()
{
    for (int max_new = 1; max_new <= 5; ++max_new)
    {
        ModelConfig config = MakeConfig();
        FCFSScheduler sched;
        PagedBlockManager mgr(config);
        CountingExecutor exec;
        SimulationLoop loop(config, sched, mgr, exec);

        auto result = loop.Run({MakeReq(1, 0.0, 10, max_new)});

        CHECK(result.completed.size() == 1);
        CHECK(exec.prefill_calls == 1);                 // 这条就是 bug 的回归断言
        CHECK(exec.ComputeSteps() == max_new);          // 每个生成 token 恰好一次计算
        if (!result.completed.empty())
        {
            CHECK(result.completed[0].generated_tokens == max_new);
        }
        CHECK(mgr.NumFreeBlocks() == config.num_blocks);
    }
}

// 多请求：Prefill 次数 == 请求数（与并发无关）
static void TestPrefillCountMatchesRequests()
{
    ModelConfig config = MakeConfig();
    FCFSScheduler sched;
    PagedBlockManager mgr(config);
    CountingExecutor exec;
    SimulationLoop loop(config, sched, mgr, exec);

    std::vector<Request> reqs;
    for (int i = 1; i <= 6; ++i)
    {
        reqs.push_back(MakeReq(static_cast<RequestId>(i), 0.01 * (i - 1), 20, i));
    }

    auto result = loop.Run(std::move(reqs));

    CHECK(result.completed.size() == 6);
    CHECK(exec.prefill_calls == 6);
    CHECK(exec.ComputeSteps() == 1 + 2 + 3 + 4 + 5 + 6);
    CHECK(mgr.NumFreeBlocks() == config.num_blocks);
}

int main()
{
    TestSingleRequestStepCount();
    TestPrefillCountMatchesRequests();

    if (g_failures == 0)
    {
        std::printf("All prefill-once tests passed.\n");
        return 0;
    }
    std::printf("%d checks failed.\n", g_failures);
    return 1;
}
