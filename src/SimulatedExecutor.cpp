#include "SimulatedExecutor.h"

namespace agentinferkv 
{

    SimulatedExecutor::SimulatedExecutor(ExecutorConfig config) : config_(config) {}

    double SimulatedExecutor::Prefill(const Request& req, const BlockTableView& view) 
    {
        (void)view;
        int prefill_tokens = req.PromptLen() - req.cached_prefix_len;
        if (prefill_tokens < 0) prefill_tokens = 0;
        double cost = config_.base_prefill_s + config_.per_token_prefill_s * static_cast<double>(prefill_tokens);
        return cost > 0.0 ? cost : 0.0;
    }

    double SimulatedExecutor::Decode(const Request& req, const BlockTableView& view) 
    {
        (void)view;
        (void)req;  // Decode 每步只生成 1 个 token，和 prompt 长度无关
        double cost = config_.base_decode_s + config_.per_token_decode_s;
        return cost > 0.0 ? cost : 0.0;
    }

    std::vector<double> SimulatedExecutor::DecodeBatch(const std::vector<const Request*>& reqs,
            const std::vector<BlockTableView>& views) 
    {
        std::vector<double> costs(reqs.size(), 0.0);
        for (std::size_t i = 0; i < reqs.size(); ++i) 
        {
            costs[i] = Decode(*reqs[i], views[i]);
        }
        return costs;
    }

    void SimulatedExecutor::Release(RequestId id) 
    {
        (void)id;  // 模拟执行器无执行侧资源
    }
}   