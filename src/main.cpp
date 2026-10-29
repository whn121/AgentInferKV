#include "FCFSScheduler.h"
#include "Interfaces.h"
#include "NaivePreallocator.h"
#include "TraceReplayer.h"

#include <algorithm>
#include <cstdio>

using namespace agentinferkv;

int main() {
    // 用 LLaMA-2-7B 的配置，block_size 调小一点方便看日志。
    ModelConfig config;
    config.num_layers = 32;
    config.num_kv_heads = 32;
    config.head_dim = 128;
    config.dtype_bytes = 2;
    config.block_size = 16;
    config.num_blocks = 64;  // 池子小一点，能直接看到“块不够”

    // 5 条 trace：3 条早到 + 2 条晚到，故意让后两条争块。
    std::vector<TraceSpec> trace = {
        {1, 0.00, 100, 20},  // 需要 ceil(120/16)=8 块
        {2, 0.01, 200, 20},  // 需要 ceil(220/16)=14 块
        {3, 0.02, 50,  10},  // 需要 ceil(60/16)=4 块
        {4, 0.03, 300, 40},  // 需要 ceil(340/16)=22 块
        {5, 0.04, 400, 50},  // 需要 ceil(450/16)=29 块
    };

    auto requests = BuildRequests(trace);
    std::sort(requests.begin(), requests.end(),
              [](const Request& a, const Request& b) {
                  return a.arrival_time < b.arrival_time;
              });

    FCFSScheduler scheduler;
    NaivePreallocator block_mgr(config);

    std::vector<Request> waiting;
    std::vector<Request> running;

    std::printf("=== FCFS + NaivePreallocator trace replay ===\n");
    std::printf("Model: LLaMA-2-7B-like, block_size=%d, num_blocks=%d, "
                "KV bytes/token=%zu\n",
                config.block_size, config.num_blocks, config.KvBytesPerToken());

    for (auto& req : requests) {
        waiting.push_back(req);

        SchedulerOutput out = scheduler.Schedule(waiting, running, block_mgr);

        for (RequestId id : out.scheduled_ids) {
            // 从 waiting 里找出该请求，分配块，转移到 running。
            auto it = std::find_if(waiting.begin(), waiting.end(),
                                   [id](const Request& r) { return r.id == id; });
            if (it == waiting.end()) continue;

            BlockTableView view = block_mgr.Allocate(*it);
            if (view.NumBlocks() == 0) continue;  // 现在容易发生, 未来理论上不会发生

            std::printf("  [t=%.2f] admit  id=%llu prompt_len=%d "
                        "max_new=%d blocks=%d free_after=%d\n",
                        it->arrival_time,
                        static_cast<unsigned long long>(it->id),
                        it->PromptLen(),
                        it->max_new_tokens,
                        view.NumBlocks(),
                        block_mgr.NumFreeBlocks());

            running.push_back(*it);
            waiting.erase(it);
        }
    }

    std::printf("\n=== End state ===\n");
    std::printf("  running=%zu waiting=%zu free_blocks=%d\n",
                running.size(), waiting.size(), block_mgr.NumFreeBlocks());

    if (!waiting.empty()) {
        std::printf("  Rejected (no blocks):");
        for (const auto& r : waiting) {
            std::printf(" id=%llu", static_cast<unsigned long long>(r.id));
        }
        std::printf("\n");
    }

    return 0;
}