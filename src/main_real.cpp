#include "FCFSScheduler.h"
#include "Interfaces.h"
#include "LlamaCppExecutor.h"
#include "PagedBlockManager.h"
#include "PrefixCache.h"
#include "SimulationLoop.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace agentinferkv;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <model.gguf> [--no-prefix] [--no-kv-reuse] [--sequential]\n", argv[0]);
        std::fprintf(stderr, "  --no-prefix    关闭 PrefixCache（控制层不做前缀匹配）\n");
        std::fprintf(stderr, "  --no-kv-reuse  控制层照常匹配，但执行层不复制 KV\n");
        std::fprintf(stderr, "  --sequential   请求串行到达，batch 组成固定，用于校验采样一致性\n");
        return 1;
    }
    std::string model_path = argv[1];

    bool enable_prefix = true;
    bool enable_kv_reuse = true;
    bool sequential = false;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-prefix") == 0) enable_prefix = false;
        if (std::strcmp(argv[i], "--no-kv-reuse") == 0) enable_kv_reuse = false;
        if (std::strcmp(argv[i], "--sequential") == 0) sequential = true;
    }

    ModelConfig config;
    config.num_layers = 32;
    config.num_kv_heads = 4;
    config.head_dim = 256;
    config.dtype_bytes = 2;
    config.block_size = 16;
    config.num_blocks = 64;

    FCFSScheduler sched;
    PagedBlockManager mgr(config);
    // n_seq_max 要给"被钉住的前缀源 seq"留余量
    LlamaCppExecutor exec(model_path, 99, 2048, 16);
    exec.SetChunkTokens(enable_kv_reuse ? config.block_size : 0);
    exec.SetMaxPinnedSeqs(3);

    PrefixCache cache(&mgr, 0);
    SimulationLoop loop(config, sched, mgr, exec, enable_prefix ? &cache : nullptr);

    // Agent 场景的典型特征：一长段共享 system prompt + 每轮不同的 user query。
    // 这里重复到约 80~100 token，才能让"前缀复用"真正省下可观的工作量。
    std::string system_prompt;
    for (int k = 0; k < 6; ++k) {
        system_prompt += "You are a careful assistant for retrieval augmented generation. ";
    }

    std::vector<Request> reqs;
    for (int i = 1; i <= 5; ++i) {
        std::string uq = "Question number " + std::to_string(i) +
                         ": What is the capital of country " +
                         std::to_string(i) + "?";
        Request r;
        r.id = static_cast<RequestId>(i);
        // sequential：请求不重叠，batch 组成固定，用来严格验证"复用后采样是否一致"
        r.arrival_time = (sequential ? 0.6 : 0.05) * (i - 1);
        r.max_new_tokens = 5;
        r.prompt_text = system_prompt + " " + uq;
        r.prompt_tokens = exec.TokenizeText(r.prompt_text);
        reqs.push_back(std::move(r));
    }

    std::printf("prompt_tokens(first request)=%zu\n", reqs.empty() ? 0u : reqs[0].prompt_tokens.size());

    auto result = loop.Run(std::move(reqs));

    double sum_ttft = 0.0;
    int n = 0;
    for (const auto& r : result.completed) {
        if (r.state == RequestState::kDone && r.Ttft() > 0) {
            sum_ttft += r.Ttft();
            ++n;
        }
    }
    double avg_ttft = (n > 0) ? sum_ttft / n : 0.0;

    int total_gen = 0;
    int max_gen = 0;
    int eos_early = 0;
    int total_cached = 0;
    for (const auto& r : result.completed) 
    {
        total_gen += r.generated_tokens;
        if (r.generated_tokens > max_gen) max_gen = r.generated_tokens;
        if (r.generated_tokens < 5) eos_early++;
        total_cached += r.cached_prefix_len;
    }

    std::printf("=== Real GPU execution ===\n");
    std::printf("  prefix_cache=%s kv_reuse=%s sequential=%s kv_reuse_supported=%s\n",
                enable_prefix ? "ON" : "OFF", enable_kv_reuse ? "ON" : "OFF",
                sequential ? "ON" : "OFF", exec.KvReuseSupported() ? "yes" : "no");
    std::printf("  completed=%zu avg_ttft=%.4f s max_concurrent=%d total_sim_time=%.4f s\n",
                result.completed.size(), avg_ttft,
                result.max_concurrent, result.total_sim_time);
    std::printf("  total_generated=%d max_generated=%d eos_early=%d/%zu\n",
                total_gen, max_gen, eos_early, result.completed.size());
    std::printf("  prefix: cached_tokens=%d hits=%d/%d | kv: reuse_hits=%d reused_tokens=%d pinned=%zu evicted=%d\n",
                total_cached, cache.TotalHits(), cache.TotalQueries(),
                exec.ReuseHits(), exec.ReusedPrefixTokens(),
                exec.PinnedSeqs(), exec.EvictedPinnedSeqs());

    // 每个请求生成 token 序列的 FNV 哈希：开启 KV 复用后 greedy 采样结果必须与关闭时完全一致
    std::printf("  token_digest:");
    for (RequestId id = 1; id <= 5; ++id)
    {
        const std::vector<int32_t>* ids = exec.GeneratedIds(id);
        std::uint64_t h = 1469598103934665603ull;
        if (ids != nullptr)
        {
            for (int32_t tok : *ids)
            {
                h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(tok));
                h *= 1099511628211ull;
            }
        }
        std::printf(" r%llu=%016llx", static_cast<unsigned long long>(id), static_cast<unsigned long long>(h));
    }
    std::printf("\n");
    return 0;
}
