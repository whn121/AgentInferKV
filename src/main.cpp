#include "FCFSScheduler.h"
#include "Interfaces.h"
#include "PagedBlockManager.h"
#include "PrefixCache.h"
#include "SimulatedExecutor.h"
#include "SimulationLoop.h"

#include <cstdio>
#include <vector>

using namespace agentinferkv;

static Request MakeAgentReq(RequestId id, double arrival, int system_len, int user_len, int max_new) 
{
    Request r;
    r.id = id;
    r.arrival_time = arrival;
    r.max_new_tokens = max_new;
    r.prompt_tokens.resize(static_cast<std::size_t>(system_len + user_len));
    // 所有请求共享的 System Prompt
    for (int i = 0; i < system_len; ++i) 
    {
        r.prompt_tokens[static_cast<std::size_t>(i)] = 1000 + i;
    }
    // 每个请求不同的 User Query
    for (int i = 0; i < user_len; ++i) 
    {
        r.prompt_tokens[static_cast<std::size_t>(system_len + i)] = 2000 + static_cast<int>(id) * 100 + i;
    }
    return r;
}

static void RunOnce(const char* label, const ModelConfig& config, const ExecutorConfig& exec_cfg
                    , const std::vector<Request>& reqs, bool enable_prefix) 
{
    FCFSScheduler sched;
    PagedBlockManager mgr(config);
    SimulatedExecutor exec(exec_cfg);
    PrefixCache cache;
    SimulationLoop loop(config, sched, mgr, exec, enable_prefix ? &cache : nullptr);

    // Run 接收 vector 按值，拷一份
    std::vector<Request> copy = reqs;
    auto result = loop.Run(std::move(copy));

    double sum_ttft = 0.0;
    int n = 0;
    int total_cached = 0;
    int total_prefill_tokens = 0;
    for (const auto& r : result.completed) 
    {
        if (r.state == RequestState::kDone && r.Ttft() > 0) 
        {
            sum_ttft += r.Ttft();
            ++n;
            total_cached += r.cached_prefix_len;
            total_prefill_tokens += (r.PromptLen() - r.cached_prefix_len);
        }
    }
    double avg_ttft = (n > 0) ? sum_ttft / n : 0.0;

    std::printf("=== %s ===\n", label);
    std::printf("  completed=%zu avg_ttft=%.4f total_cached_tokens=%d total_prefill_tokens=%d hits=%d/%d\n"
                , result.completed.size(), avg_ttft, total_cached, total_prefill_tokens
                , cache.TotalHits(), cache.TotalQueries());
}

int main() 
{
    ModelConfig config;
    config.num_layers = 32;
    config.num_kv_heads = 32;
    config.head_dim = 128;
    config.dtype_bytes = 2;
    config.block_size = 16;
    config.num_blocks = 32;

    ExecutorConfig exec_cfg;
    exec_cfg.base_prefill_s = 0.001;          // 固定开销调小
    exec_cfg.per_token_prefill_s = 0.0001;    // 每 token 开销调大（0.1ms/token）

    // Agent 多轮：相同 System Prompt 32 token，不同 User Query 16 token
    std::vector<Request> reqs;
    for (int i = 1; i <= 5; ++i) 
    {
        reqs.push_back(MakeAgentReq(static_cast<RequestId>(i), 0.02 * (i - 1), 32, 16, 10));
    }

    RunOnce("PrefixCache OFF", config, exec_cfg, reqs, false);
    RunOnce("PrefixCache ON", config, exec_cfg, reqs, true);

    return 0;
}