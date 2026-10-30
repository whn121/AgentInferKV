#include "FCFSScheduler.h"
#include "Interfaces.h"
#include "NaivePreallocator.h"
#include "SimulatedExecutor.h"
#include "SimulationLoop.h"
#include "TraceReplayer.h"

#include <cstdio>

using namespace agentinferkv;

int main() {
    ModelConfig config;
    config.num_layers = 32;
    config.num_kv_heads = 32;
    config.head_dim = 128;
    config.dtype_bytes = 2;
    config.block_size = 16;
    config.num_blocks = 64;

    std::vector<TraceSpec> trace = {
        {1, 0.00, 100, 5},
        {2, 0.01, 200, 5},
        {3, 0.02, 50,  5},
        {4, 0.03, 300, 10},
        {5, 0.04, 400, 20},
    };

    auto requests = BuildRequests(trace);

    FCFSScheduler sched;
    NaivePreallocator mgr(config);
    SimulatedExecutor exec;
    SimulationLoop loop(config, sched, mgr, exec);

    auto result = loop.Run(std::move(requests));

    std::printf("=== SimulationLoop result ===\n");
    std::printf("Total sim time: %.4f s\n", result.total_sim_time);
    std::printf("Max concurrent: %d\n", result.max_concurrent);
    std::printf("Completed: %zu\n", result.completed.size());
    std::printf("\n");

    for (const auto& r : result.completed) {
        std::printf("  id=%llu state=%s ttft=%.4f latency=%.4f gen=%d\n",
                    static_cast<unsigned long long>(r.id),
                    ToString(r.state),
                    r.Ttft(), r.Latency(), r.generated_tokens);
    }

    std::printf("\nFree blocks: %d / %d\n", mgr.NumFreeBlocks(), config.num_blocks);

    return 0;
}