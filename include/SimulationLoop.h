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
        int peak_blocks_in_use = 0;       // 峰值占用物理块数（碎片对比用）
    };

    // 事件驱动的主循环
    // 串起 Scheduler + BlockSpaceManager + Executor，按模拟时间推进
    //
    // 块回收约定：
    //   - 未启用 PrefixCache 时，Run 返回后 block_mgr 回到初始空闲状态；
    //   - 启用 PrefixCache 时，缓存主动持有的块会留在池外（max_cached_blocks == 0），
    //     或最多保留 max_cached_blocks 个（LRU 淘汰开启）。这是“用空间换复用”的代价。
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