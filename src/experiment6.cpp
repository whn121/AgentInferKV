// L1 演示：自管 cell 池的执行器 —— 证明"块就是物理内存区间"和"复用真的省了写"。
//
// 对比两组：
//   A. 不开前缀缓存：每个请求都要写满 40 个 token 的 KV
//   B. 开前缀缓存：前 32 个 token 命中，只写没命中的部分
//
// 统计口径：每 token KV 字节数来自 ModelConfig（4B 模型约 80KB/token），
// 所以"写入字节数"是有物理意义的，不是随便乘一个数。
//
// 用法：./agentinferkv_cell_demo

#include "CellPool.h"
#include "FCFSScheduler.h"
#include "Interfaces.h"
#include "ModelConfig.h"
#include "PagedBlockManager.h"
#include "PrefixCache.h"
#include "SimulatedCellExecutor.h"
#include "SimulationLoop.h"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace agentinferkv;

namespace
{
    void RunOnce(const char* label, bool use_cache)
    {
        ModelConfig config;
        config.num_layers = 32;
        config.num_kv_heads = 4;
        config.head_dim = 160;
        config.dtype_bytes = 2;
        config.block_size = 16;
        config.num_blocks = 64;

        FCFSScheduler sched;
        PagedBlockManager mgr(config);
        PrefixCache cache(&mgr, 0);
        SimulatedCellExecutor exec(config);

        SimulationLoop loop(config, sched, mgr, exec, use_cache ? &cache : nullptr);

        // 20 个请求：前 32 token 完全相同（共享 system prompt），后 8 token 各自不同
        std::vector<Request> reqs;
        for (int i = 1; i <= 20; ++i)
        {
            Request r;
            r.id = static_cast<RequestId>(i);
            r.arrival_time = 0.02 * (i - 1);
            r.max_new_tokens = 8;
            r.prompt_tokens.resize(40);
            for (int k = 0; k < 32; ++k) r.prompt_tokens[static_cast<std::size_t>(k)] = k;      // 共享
            for (int k = 32; k < 40; ++k) r.prompt_tokens[static_cast<std::size_t>(k)] = 1000 + i; // 不同
            reqs.push_back(std::move(r));
        }

        auto result = loop.Run(std::move(reqs));

        const auto pool = exec.Pool();
        std::printf("%-10s kv=%zu written=%.2fMB hit=%d/%d\n",
                    label,
                    config.KvBytesPerToken(),
                    static_cast<double>(pool->WrittenBytes()) / (1024.0 * 1024.0),
                    cache.TotalHits(), cache.TotalQueries());
    }
}

int main(int argc, char** argv)
{
    if (argc > 1 && (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0))
    {
        std::printf("usage: %s\n  no args: run KV write-volume A/B (20 reqs, 32 shared)\n", argv[0]);
        return 0;
    }
    RunOnce("A", false);
    RunOnce("B", true);
    return 0;
}
