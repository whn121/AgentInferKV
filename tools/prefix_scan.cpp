// 前缀共享度扫描：公共前缀占比 -> 命中率 / 复用 token 数。
//
// 为什么要这个实验：只报"热缓存场景 99.8%"说明不了复用能力，
// 面试官会问"真实请求共享度低的时候呢"。这里把共享度做成可调参数，
// 给出命中率与省下的 prefill token 随共享度变化的曲线。
//
// 用法：./agentinferkv_prefix_scan [requests] [prompt_len]

#include "FCFSScheduler.h"
#include "PagedBlockManager.h"
#include "PrefixCache.h"
#include "SimulatedExecutor.h"
#include "SimulationLoop.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace agentinferkv;

namespace
{
    const int kBlockSize = 16;

    ModelConfig MakeConfig(int num_blocks)
    {
        ModelConfig c;
        c.block_size = kBlockSize;
        c.num_blocks = num_blocks;
        return c;
    }

    // 构造一批请求：前 shared_tokens 个 token 所有请求相同，后面各自不同。
    std::vector<Request> MakeTrace(int requests, int prompt_len, int shared_tokens)
    {
        std::vector<Request> reqs;
        reqs.reserve(static_cast<std::size_t>(requests));

        for (int i = 1; i <= requests; ++i)
        {
            Request r;
            r.id = static_cast<RequestId>(i);
            r.arrival_time = 0.5 * (i - 1);
            r.max_new_tokens = 8;
            r.prompt_tokens.resize(static_cast<std::size_t>(prompt_len));

            for (int k = 0; k < shared_tokens && k < prompt_len; ++k)
            {
                r.prompt_tokens[static_cast<std::size_t>(k)] = 100 + k;   // 公共前缀
            }
            for (int k = shared_tokens; k < prompt_len; ++k)
            {
                r.prompt_tokens[static_cast<std::size_t>(k)] = 10000 + i * 1000 + k;  // 各自不同
            }
            reqs.push_back(std::move(r));
        }
        return reqs;
    }

    struct Row
    {
        int ratio = 0;
        int queries = 0;
        int hits = 0;
        long long reused = 0;
        long long total_tokens = 0;
        int completed = 0;
    };

    Row RunOnce(int requests, int prompt_len, int ratio)
    {
        Row row;
        row.ratio = ratio;

        const int shared = (prompt_len * ratio / 100 / kBlockSize) * kBlockSize;  // 按块对齐
        const ModelConfig config = MakeConfig(2048);

        FCFSScheduler sched;
        PagedBlockManager mgr(config);
        PrefixCache cache(&mgr, 0);          // 不淘汰：只看复用上界
        SimulatedExecutor exec;

        SimulationLoop loop(config, sched, mgr, exec, &cache);
        auto result = loop.Run(MakeTrace(requests, prompt_len, shared));

        row.queries = cache.RetainQueries();
        row.hits = cache.RetainHits();
        row.reused = static_cast<long long>(cache.ReusedTokens());
        row.total_tokens = static_cast<long long>(requests) * prompt_len;
        row.completed = static_cast<int>(result.completed.size());
        return row;
    }
}  // namespace

int main(int argc, char** argv)
{
    const int requests = (argc > 1) ? std::atoi(argv[1]) : 100;
    const int prompt_len = (argc > 2) ? std::atoi(argv[2]) : 256;

    std::printf("requests=%d prompt_len=%d block_size=%d\n", requests, prompt_len, kBlockSize);
    std::printf("%-8s %-10s %-10s %-10s %-12s %-10s %-6s\n",
                "shared", "queries", "hits", "hit_rate", "reused_tok", "saved", "done");

    for (int ratio : {0, 10, 25, 50, 75, 90})
    {
        Row r = RunOnce(requests, prompt_len, ratio);
        const double hit_rate = r.queries > 0 ? 100.0 * r.hits / r.queries : 0.0;
        const double saved = r.total_tokens > 0 ? 100.0 * static_cast<double>(r.reused) / r.total_tokens : 0.0;
        std::printf("%-8s %-10d %-10d %-9.1f%% %-12lld %-9.1f%% %-6d\n",
                    (std::to_string(r.ratio) + "%").c_str(),
                    r.queries, r.hits, hit_rate, r.reused, saved, r.completed);
    }
    return 0;
}
