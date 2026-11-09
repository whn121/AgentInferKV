#pragma once

#include "Interfaces.h"

namespace agentinferkv
{
    // 模拟执行器的耗时参数。单位：秒。
    //
    // ── 这些系数不是拍脑袋的 ──
    // 由 tools/calibrate_executor.cpp 在真实 GPU 上实测后拟合而来：
    //   硬件：RTX 4060 Laptop 8GB（WSL2）
    //   方法：prompt 长度 11/41/81/161/321 token，每个长度 9 次取中位数，
    //         再对 (tokens, prefill_ms) 做最小二乘拟合
    //   拟合：tools/fit_executor_config.py
    //
    // 模型 A：Qwen2.5-0.5B-Instruct Q4_K_M（dense，默认值取自它）
    //   prefill = 5.63 ms + 3.09 µs/token      (R² = 0.33，固定开销主导)
    //   decode  = 0.63 ms/token                (与上下文长度基本无关)
    //
    // 模型 B：Qwen3.5-4B Q4_K_M（hybrid）
    //   prefill = 12.47 ms + 31.72 µs/token    (R² = 0.91，线性很好)
    //   decode  = 1.10 ms/token
    //   想用这组：ExecutorConfig{0.012466, 0.000031718, 0.001101, 0.0}
    //
    // 诚实边界：R²=0.33 说明 0.5B 上 prefill 主要是固定开销（kernel launch 等），
    // 线性项在噪声范围内；4B 上线性项才是主导。模拟器只用于比较**策略**，
    // 不代表任意硬件上的绝对延迟。
    struct ExecutorConfig
    {
        double base_prefill_s = 0.005634;         // 5.63 ms（0.5B 实测拟合）
        double per_token_prefill_s = 0.000003094; // 3.09 µs/token
        double base_decode_s = 0.000633;          // 0.63 ms/token
        double per_token_decode_s = 0.0;          // 实测 decode 与长度无关，故为 0
    };
    
    // 模拟执行器：用公式返回耗时，不提交 GPU 任务
    // 所有返回值仅用于驱动模拟时钟，不代表真实 GPU 性能
    class SimulatedExecutor : public IInferenceExecutor 
    {
        public:
        explicit SimulatedExecutor(ExecutorConfig config = {});

        // 返回 base_prefill + per_token_prefill * prompt_len。
        // view 参数当前未使用，保留是为了未来接真实 Executor 时签名不变。
        double Prefill(const Request& req, const BlockTableView& view) override;

        // 返回 base_decode + per_token_decode * 1（Decode 每步只生成 1 个 token）
        double Decode(const Request& req, const BlockTableView& view) override;

        std::vector<double> DecodeBatch(const std::vector<const Request*>& reqs,
                                    const std::vector<BlockTableView>& views) override;
            
        void Release(RequestId id) override;

    private:
        ExecutorConfig config_;

    };
}