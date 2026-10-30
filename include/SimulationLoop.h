#pragma once

#include "Interfaces.h"
#include "ModelConfig.h"
#include "PrefixCache.h"

#include <vector>

namespace agentinferkv
{
    // 一次 Run 的结果
    struct SimulationResult
    {
        std::vector<Request> completed;   // 已完成或取消的请求
        double total_sim_time = 0.0;      // 模拟时钟结束时刻
        int max_concurrent = 0;           // 峰值并发运行请求数
    };

    // 事件驱动的主循环
    // 串起 Scheduler + BlockSpaceManager + Executor，按模拟时间推进
    class SimulationLoop
    {
    public:
        SimulationLoop(ModelConfig config, IScheduler& scheduler, IBlockSpaceManager& block_mgr, IInferenceExecutor& executor
                        , PrefixCache* prefix_cache = nullptr);
        
        // 跑完整个trace, 返回结果 调用后 block_mgr 应该回到初始空闲状态
        SimulationResult Run(std::vector<Request> requests);

    private:
        ModelConfig config_;
        IScheduler& scheduler_;
        IBlockSpaceManager& block_mgr_;
        IInferenceExecutor& executor_;
        PrefixCache* prefix_cache_;
    };
}