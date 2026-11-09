// 标定工具：在真实 GPU 上测量 Prefill / Decode 耗时，供 SimulatedExecutor 拟合。
//
// 为什么要它：SimulatedExecutor 的 ExecutorConfig 一开始是"定性默认值"
// （base_prefill=5ms、per_token=10µs、decode=0.5ms…），没有任何实测依据。
// 用它跑真实模型，把 (prompt_tokens, prefill_ms) 和 (decode 每步 ms) 采出来，
// 再线性拟合 cost = a + b × tokens，写回 ExecutorConfig。
//
// 用法：
//   ./agentinferkv_calibrate <model.gguf> [reps]
// 输出（CSV，供 tools/fit_executor_config.py 使用）：
//   L,tokens,prefill_ms,decode_step_ms
//
// 关键实现决定：**每次测量都重建 LlamaCppExecutor**。
// 原因：LlamaCppExecutor 会把 prefill 过的 seq 登记为可复用前缀源并 pin 住，
// Release 时不释放它的 KV、也不归还 seq id；同一个 context 里连续测多个请求，
// 会走到 seq_id >= n_seq_max 的断言（GGML_ASSERT abort）。
// 标定要的是"干净 KV 上的单次 prefill 成本"，重建一次 ~1s，代价可接受。

#include "LlamaCppExecutor.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace agentinferkv;

static double Median(std::vector<double> v)
{
    if (v.empty()) return -1.0;
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: %s <model.gguf> [reps]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const int reps = (argc > 2) ? std::max(1, std::atoi(argv[2])) : 5;

    const std::string unit = "The quick brown fox jumps over the lazy dog. ";
    const int counts[] = {1, 4, 8, 16, 32};
    const int decode_steps = 16;

    std::printf("L,tokens,prefill_ms,decode_step_ms\n");

    for (int count : counts)
    {
        std::string text;
        for (int i = 0; i < count; ++i) text += unit;

        std::vector<double> prefills;
        std::vector<double> decodes;
        std::size_t tokens = 0;

        for (int r = 0; r < reps; ++r)
        {
            // 每次测量都从干净 context 开始（见文件头注释）
            LlamaCppExecutor exec(model_path, /*n_gpu_layers=*/99, /*n_ctx=*/4096, /*n_seq_max=*/1);
            exec.SetChunkTokens(0);   // 标定"干净 prefill"，不掺复用
            exec.SetMaxPinnedSeqs(0); // 不保留任何前缀源

            Request req;
            req.id = 1;                       // 每个 context 只跑一个请求
            req.prompt_text = text;
            req.prompt_tokens = exec.TokenizeText(text);
            req.max_new_tokens = decode_steps;
            tokens = req.prompt_tokens.size();

            const double p = exec.Prefill(req, {});
            if (p < 0.0)
            {
                std::fprintf(stderr, "prefill failed at L=%d rep=%d\n", count, r);
                continue;
            }

            double d_total = 0.0;
            int d_ok = 0;
            for (int s = 0; s < decode_steps; ++s)
            {
                const double d = exec.Decode(req, {});
                if (d < 0.0) break;
                d_total += d;
                ++d_ok;
            }

            prefills.push_back(p * 1000.0);
            if (d_ok > 0) decodes.push_back(d_total / d_ok * 1000.0);
        }

        std::printf("%d,%zu,%.4f,%.4f\n", count, tokens, Median(prefills), Median(decodes));
        std::fflush(stdout);
    }

    return 0;
}
