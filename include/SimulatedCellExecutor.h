#pragma once

#include "CellPool.h"
#include "Interfaces.h"
#include "ModelConfig.h"
#include "SimulatedExecutor.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace agentinferkv
{
    // 自管 cell 池的执行器（L1 演示）。
    //
    // 它和 SimulatedExecutor 的差别：
    //   SimulatedExecutor 只返回耗时，不碰任何"内存"；
    //   这个执行器会把 prompt（以及 decode 的每个 token）的 KV 真正写进 CellPool，
    //   因此可以统计"写了多少字节""命中前缀省了多少写"。
    //
    // 它仍然不跑 attention——计算部分还是用公式模拟。
    // 目的是证明"块 → cell 区间"的映射和复用是真的，不是纸上画的。
    class SimulatedCellExecutor : public IInferenceExecutor
    {
    public:
        SimulatedCellExecutor(const ModelConfig& config, ExecutorConfig exec_cfg = {})
            : config_(config),
              exec_cfg_(exec_cfg),
              pool_(std::make_shared<CellPool>(config.num_blocks, config.block_size,
                                               config.KvBytesPerToken()))
        {
        }

        double Prefill(const Request& req, const BlockTableView& view) override;
        double PrefillTokens(const Request& req, const std::vector<int32_t>& tokens) override;
        double Decode(const Request& req, const BlockTableView& view) override;
        std::vector<double> DecodeBatch(const std::vector<const Request*>& reqs,
                                        const std::vector<BlockTableView>& views) override;
        void Release(RequestId id) override;
        void OnBlocksAssigned(RequestId id, const BlockTableView& view) override;

        std::shared_ptr<CellPool> Pool() const { return pool_; }

    private:
        // 把 [start_token, start_token+tokens) 的 KV 写进块表对应的 cell
        void WriteKv(const BlockTableView& view, int start_token, int tokens);
        double PrefillWithView(const Request& req, int total_tokens);

        ModelConfig config_;
        ExecutorConfig exec_cfg_;
        std::shared_ptr<CellPool> pool_;
        std::unordered_map<RequestId, BlockTableView> views_;
    };
}
