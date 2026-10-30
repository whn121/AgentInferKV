#pragma once

#include "Interfaces.h"

namespace agentinferkv
{
    // 模拟执行器的耗时参数。
    // 单位：秒。默认值对应“长 prompt Prefill 明显更慢、Decode 每步接近固定”的定性模型
    struct ExecutorConfig
    {
        double base_prefill_s = 0.005;         // Prefill 固定开销
        double per_token_prefill_s = 0.00001;  // 每个 prompt token 的额外耗时
        double base_decode_s = 0.0005;         // Decode 一步的固定开销
        double per_token_decode_s = 0.00002;   // 每个生成 token 的额外耗时
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

    private:
        ExecutorConfig config_;

    };
}