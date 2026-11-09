#include "FCFSScheduler.h"
#include "Interfaces.h"
#include "LlamaCppExecutor.h"
#include "PagedBlockManager.h"
#include "PrefixCache.h"
#include "SimulationLoop.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace agentinferkv;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <model.gguf> [--no-prefix] [--no-kv-reuse] [--sequential]\n"
                             "       [--system-repeat=N] [--ctx=N]\n", argv[0]);
        std::fprintf(stderr, "  --no-prefix    关闭 PrefixCache（控制层不做前缀匹配）\n");
        std::fprintf(stderr, "  --kv-reuse     开启跨请求 KV 复用（实验性：长前缀下输出可能不一致）\n");
        std::fprintf(stderr, "  --sequential   请求串行到达，batch 组成固定，用于校验采样一致性\n");
        std::fprintf(stderr, "  --system-repeat=N  共享 system prompt 重复段数（默认 6），用来放大前缀长度\n");
        std::fprintf(stderr, "  --ctx=N        上下文长度（默认 2048）\n");
        std::fprintf(stderr, "  --num-requests=N  请求数（默认 5；trace 实验用 100）\n");
        std::fprintf(stderr, "  --prompt-mix=A/B/C  短/中/长 system prompt 百分比（默认 50/30/20）\n");
        std::fprintf(stderr, "  --seed=S       随机种子（默认 42，保证 trace 可复现）\n");
        std::fprintf(stderr, "  --max-new=N    每个请求最多生成 token（默认 5）\n");
        std::fprintf(stderr, "  --n-seq=N      并发 seq 上限（默认 16；统一 KV 下每 seq 可用 ctx/N）\n");
        std::fprintf(stderr, "  --blocks=N     KV 块池大小（默认 64，每块 16 token）\n");
        return 1;
    }
    std::string model_path = argv[1];

    bool enable_prefix = true;
    // 跨请求 KV 复用默认关闭：实验四证明 seq_cp 路径在长前缀下与完整 prefill
    // 输出不一致（README 实验四）。正确性优先，需要时显式 --kv-reuse 打开。
    bool enable_kv_reuse = false;
    bool sequential = false;
    int system_repeat = 6;
    int n_ctx = 2048;
    int max_pinned = 3;
    double arrival_gap = -1.0;   // <0 表示用默认值（sequential 0.6s / 否则 0.05s）
    bool verify_reuse = false;
    int reuse_margin = 0;
    int num_requests = 5;
    int mix_short = 50, mix_mid = 30, mix_long = 20;
    long seed = 42;
    int max_new = 5;
    int n_seq = 16;
    int num_blocks = 64;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-prefix") == 0) enable_prefix = false;
        if (std::strcmp(argv[i], "--no-kv-reuse") == 0) enable_kv_reuse = false;
        if (std::strcmp(argv[i], "--kv-reuse") == 0) enable_kv_reuse = true;
        if (std::strcmp(argv[i], "--sequential") == 0) sequential = true;
        if (std::strncmp(argv[i], "--system-repeat=", 16) == 0) system_repeat = std::atoi(argv[i] + 16);
        if (std::strncmp(argv[i], "--ctx=", 6) == 0) n_ctx = std::atoi(argv[i] + 6);
        if (std::strncmp(argv[i], "--max-pinned=", 13) == 0) max_pinned = std::atoi(argv[i] + 13);
        if (std::strncmp(argv[i], "--arrival-gap=", 14) == 0) arrival_gap = std::atof(argv[i] + 14);
        if (std::strcmp(argv[i], "--verify-reuse") == 0) verify_reuse = true;
        if (std::strncmp(argv[i], "--reuse-margin=", 15) == 0) reuse_margin = std::atoi(argv[i] + 15);
        if (std::strncmp(argv[i], "--num-requests=", 15) == 0) num_requests = std::atoi(argv[i] + 15);
        if (std::strncmp(argv[i], "--seed=", 7) == 0) seed = std::atol(argv[i] + 7);
        if (std::strncmp(argv[i], "--max-new=", 10) == 0) max_new = std::atoi(argv[i] + 10);
        if (std::strncmp(argv[i], "--n-seq=", 8) == 0) n_seq = std::atoi(argv[i] + 8);
        if (std::strncmp(argv[i], "--blocks=", 9) == 0) num_blocks = std::atoi(argv[i] + 9);
        if (std::strncmp(argv[i], "--prompt-mix=", 13) == 0)
        {
            std::sscanf(argv[i] + 13, "%d/%d/%d", &mix_short, &mix_mid, &mix_long);
        }
    }
    if (system_repeat < 1) system_repeat = 1;
    if (n_ctx < 256) n_ctx = 256;

    // 先加载模型，再按模型的真实形状构造 ModelConfig。
    // 之前的写法是"先拍一组默认值(32层/4头/256维)再建模型"，
    // 池容量、每 token KV 字节都会算错，面试一问"这些数字哪来的"就答不上。
    if (num_requests < 1) num_requests = 1;
    if (num_blocks < 4) num_blocks = 4;
    if (n_seq < 1) n_seq = 1;
    LlamaCppExecutor exec(model_path, 99, n_ctx, n_seq);
    exec.SetVerifyReuse(verify_reuse);
    exec.SetReuseMargin(reuse_margin);
    exec.SetMaxPinnedSeqs(static_cast<std::size_t>(max_pinned > 0 ? max_pinned : 1));

    ModelConfig config;
    config.num_layers = exec.NumLayers();
    config.num_kv_heads = exec.NumKvHeads();
    config.head_dim = exec.HeadDim();
    config.dtype_bytes = 2;
    config.block_size = 16;
    config.num_blocks = num_blocks;

    std::printf("model_shape: layers=%d kv_heads=%d head_dim=%d kv_bytes_per_token=%zu pool_tokens=%zu\n",
                config.num_layers, config.num_kv_heads, config.head_dim,
                config.KvBytesPerToken(), config.TotalTokenCapacity());

    exec.SetChunkTokens(enable_kv_reuse ? config.block_size : 0);

    FCFSScheduler sched;
    PagedBlockManager mgr(config);
    PrefixCache cache(&mgr, 0);
    SimulationLoop loop(config, sched, mgr, exec, enable_prefix ? &cache : nullptr);

    // Agent 场景的典型特征：一长段共享 system prompt + 每轮不同的 user query。
    // trace 用三种长度的 system prompt 混合，模拟"短/中/长上下文"的真实分布。
    const int repeat_table[3] = {8, 24, 48};   // 约 80 / 240 / 480 token
    const int pct_table[3] = {mix_short, mix_mid, mix_long};

    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::uniform_int_distribution<int> pick(0, 99);

    std::vector<Request> reqs;
    for (int i = 1; i <= num_requests; ++i) {
        // 按百分比抽长度档位（固定 seed → trace 可复现）
        const int roll = pick(rng);
        int bucket = 0;
        if (roll < pct_table[0]) bucket = 0;
        else if (roll < pct_table[0] + pct_table[1]) bucket = 1;
        else bucket = 2;

        std::string system_prompt;
        for (int k = 0; k < repeat_table[bucket]; ++k) {
            system_prompt += "You are a careful assistant for retrieval augmented generation. ";
        }

        // 为什么不用"某国首都是哪"这种问法：模型一句话就 EOS，请求在 prefill 阶段
        // 就结束、根本不产生 TTFT 样本。trace 实验要让每个请求稳定生成，
        // 所以要求"用二十个词描述"，保证 decode 步数可测。
        const std::string uq = "Describe country number " + std::to_string(i) +
                               " in about twenty words, in one sentence.";

        Request r;
        r.id = static_cast<RequestId>(i);
        const double gap = (arrival_gap >= 0.0)
            ? arrival_gap
            : (sequential ? 0.6 : 0.05);
        r.arrival_time = gap * (i - 1);
        r.max_new_tokens = max_new;
        r.prompt_text = system_prompt + " " + uq;
        r.prompt_tokens = exec.TokenizeText(r.prompt_text);
        reqs.push_back(std::move(r));
    }

    std::printf("trace: n=%d mix=%d/%d/%d seed=%ld max_new=%d arrival_gap=%.3f\n",
                num_requests, mix_short, mix_mid, mix_long, seed, max_new,
                arrival_gap >= 0.0 ? arrival_gap : (sequential ? 0.6 : 0.05));

    std::printf("prompt_tokens(first request)=%zu\n", reqs.empty() ? 0u : reqs[0].prompt_tokens.size());

    auto result = loop.Run(std::move(reqs));

    std::vector<double> ttfts;
    for (const auto& r : result.completed) {
        if (r.state == RequestState::kDone && r.Ttft() > 0) ttfts.push_back(r.Ttft());
    }
    std::sort(ttfts.begin(), ttfts.end());

    auto pct = [&ttfts](double q) -> double {
        if (ttfts.empty()) return 0.0;
        std::size_t idx = static_cast<std::size_t>(q * (ttfts.size() - 1));
        return ttfts[idx];
    };
    const double avg_ttft = ttfts.empty() ? 0.0
        : std::accumulate(ttfts.begin(), ttfts.end(), 0.0) / ttfts.size();

    int total_gen = 0;
    int max_gen = 0;
    int eos_early = 0;
    int total_cached = 0;
    for (const auto& r : result.completed) 
    {
        total_gen += r.generated_tokens;
        if (r.generated_tokens > max_gen) max_gen = r.generated_tokens;
        if (r.generated_tokens < max_new) eos_early++;
        total_cached += r.cached_prefix_len;
    }

    std::printf("=== Real GPU execution ===\n");
    std::printf("  prefix_cache=%s kv_reuse=%s sequential=%s kv_reuse_supported=%s\n",
                enable_prefix ? "ON" : "OFF", enable_kv_reuse ? "ON" : "OFF",
                sequential ? "ON" : "OFF", exec.KvReuseSupported() ? "yes" : "no");
    std::printf("  completed=%zu avg_ttft=%.4f s p50=%.4f p95=%.4f max=%.4f (n=%zu)\n",
                result.completed.size(), avg_ttft, pct(0.50), pct(0.95), pct(1.0), ttfts.size());
    std::printf("  max_concurrent=%d total_sim_time=%.4f s\n",
                result.max_concurrent, result.total_sim_time);
    std::printf("  total_generated=%d max_generated=%d eos_early=%d/%zu\n",
                total_gen, max_gen, eos_early, result.completed.size());
    std::printf("  prefix: cached_tokens=%d hits=%d/%d | kv: reuse_hits=%d reused_tokens=%d pinned=%zu evicted=%d\n",
                total_cached, cache.TotalHits(), cache.TotalQueries(),
                exec.ReuseHits(), exec.ReusedPrefixTokens(),
                exec.PinnedSeqs(), exec.EvictedPinnedSeqs());

    // 每个请求生成 token 序列的 FNV 哈希：开启 KV 复用后 greedy 采样结果必须与关闭时完全一致
    std::printf("  token_digest:");
    const int digest_n = (num_requests < 5) ? num_requests : 5;
    for (RequestId id = 1; id <= digest_n; ++id)
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
