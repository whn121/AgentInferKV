#include "SimulatedCellExecutor.h"

#include <algorithm>
#include <unordered_map>

namespace agentinferkv
{
    void SimulatedCellExecutor::WriteKv(const BlockTableView& view, int start_token, int tokens)
    {
        if (tokens <= 0) return;
        const int bs = config_.block_size;
        for (int t = start_token; t < start_token + tokens; ++t)
        {
            const int logical = t / bs;
            const int offset = t % bs;
            if (logical < 0 || static_cast<std::size_t>(logical) >= view.logical_to_physical.size())
            {
                continue;   // 块表没覆盖到的位置（理论上不该发生）
            }
            pool_->WriteTokens(view.logical_to_physical[static_cast<std::size_t>(logical)],
                               offset, 1, static_cast<std::uint8_t>(0xA0 + (logical & 0x0F)));
        }
    }

    void SimulatedCellExecutor::OnBlocksAssigned(RequestId id, const BlockTableView& view)
    {
        views_[id] = view;
    }

    double SimulatedCellExecutor::Prefill(const Request& req, const BlockTableView& view)
    {
        views_[req.id] = view;
        return PrefillWithView(req, static_cast<int>(req.PromptLen()));
    }

    double SimulatedCellExecutor::PrefillTokens(const Request& req,
                                                const std::vector<int32_t>& tokens)
    {
        // 块表在 OnBlocksAssigned / Prefill 时已经记下，这里只需要 token 数
        return PrefillWithView(req, static_cast<int>(tokens.size()));
    }

    double SimulatedCellExecutor::PrefillWithView(const Request& req, int total_tokens)
    {
        auto it = views_.find(req.id);
        const BlockTableView empty;
        const BlockTableView& view = (it == views_.end()) ? empty : it->second;

        const int cached = std::min(req.cached_prefix_len, total_tokens);

        // 只写"没命中"的那部分：命中前缀的 KV 已经在 cell 里，复用的价值就在这里。
        WriteKv(view, cached, total_tokens - cached);

        const int computed = total_tokens - cached;
        const double cost = exec_cfg_.base_prefill_s +
                            exec_cfg_.per_token_prefill_s * static_cast<double>(computed);
        return cost > 0.0 ? cost : 0.0;
    }

    double SimulatedCellExecutor::Decode(const Request& req, const BlockTableView& view)
    {
        views_[req.id] = view;
        // 这一步生成的 token 位置 = prompt + 已生成数 - 1（DecodeOneToken 已经 +1 过）
        const int pos = req.PromptLen() + req.generated_tokens - 1;
        WriteKv(view, pos, 1);
        return exec_cfg_.base_decode_s +
               exec_cfg_.per_token_decode_s * static_cast<double>(pos + 1);
    }

    std::vector<double> SimulatedCellExecutor::DecodeBatch(
        const std::vector<const Request*>& reqs,
        const std::vector<BlockTableView>& views)
    {
        std::vector<double> costs;
        costs.reserve(reqs.size());
        for (std::size_t i = 0; i < reqs.size(); ++i)
        {
            costs.push_back(Decode(*reqs[i], views[i]));
        }
        return costs;
    }

    void SimulatedCellExecutor::Release(RequestId id)
    {
        // 保留 views_ 里最后一份，方便实验结束后核对块表；真实执行器这里会回收 seq
        (void)id;
    }
}
