#include "FCFSScheduler.h"
#include "Interfaces.h"
#include "NaivePreallocator.h"
#include "PagedBlockManager.h"
#include "SimulatedExecutor.h"
#include "SimulationLoop.h"
#include "TraceReplayer.h"

#include <cstdio>
#include <vector>

using namespace agentinferkv;

static void PrintResult(const char* label,
                        const SimulationResult& result,
                        int num_blocks) {
    double sum_ttft = 0.0;
    int n = 0;
    for (const auto& r : result.completed) {
        if (r.state == RequestState::kDone && r.Ttft() > 0) {
            sum_ttft += r.Ttft();
            ++n;
        }
    }
    double avg_ttft = (n > 0) ? sum_ttft / n : 0.0;

    std::printf("=== %s ===\n", label);
    std::printf("  max_concurrent=%d completed=%zu avg_ttft=%.4f\n",
                result.max_concurrent, result.completed.size(), avg_ttft);
}

int main() {
    ModelConfig config;
    config.num_layers = 32;
    config.num_kv_heads = 32;
    config.head_dim = 128;
    config.dtype_bytes = 2;
    config.block_size = 16;
    config.num_blocks = 8;  // 小预算，放大 Naive vs Paged 差异

    // 6 个请求：prompt 相同 20，max_new 都是 30（总 50 token）
    // Naive 需要 ceil(50/16)=4 块/请求；Paged 初始 2 块，Decode 中扩到 4 块
    std::vector<TraceSpec> trace;
    for (int i = 1; i <= 6; ++i) {
        trace.push_back({static_cast<RequestId>(i),
                         0.001 * (i - 1),
                         20,
                         30});
    }

    // Naive 跑一遍
    {
        auto requests = BuildRequests(trace);
        FCFSScheduler sched;
        NaivePreallocator mgr(config);
        SimulatedExecutor exec;
        SimulationLoop loop(config, sched, mgr, exec);
        auto result = loop.Run(std::move(requests));
        PrintResult("Naive", result, config.num_blocks);
    }

    // Paged 跑一遍
    {
        auto requests = BuildRequests(trace);
        FCFSScheduler sched;
        PagedBlockManager mgr(config);
        SimulatedExecutor exec;
        SimulationLoop loop(config, sched, mgr, exec);
        auto result = loop.Run(std::move(requests));
        PrintResult("Paged", result, config.num_blocks);

        std::printf("  tail_unused_capacity=%d\n", mgr.TailUnusedCapacity());
    }

    return 0;
}