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
        int preempted = 0;                // recompute 抢占次数（块不足时重排队的次数）
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
        
        // 抢占策略：
        //   Self   —— 块不够时抢占"自己"（释放自己的块、重排队），实现简单、不会毁别人进度；
        //   Victim —— 选一个正在跑的后到请求（LIFO）作为 victim，释放它的块给当前请求，
        //             victim 重新排队。更接近 vLLM 的 recompute preemption。
        enum class PreemptPolicy { Self, Victim };
        void SetPreemptPolicy(PreemptPolicy p) { preempt_policy_ = p; }

        // 解码批模式：
        //   Continuous（默认）—— 每个请求只等自己的解码耗时（连续批处理的效果）；
        //   Sync              —— 整批等最慢的那个（静态 batch 的效果，用来做对照）。
        void SetDecodeSyncBatch(bool on) { decode_sync_batch_ = on; }

        // 跑完整个trace, 返回结果 调用后 block_mgr 应该回到初始空闲状态
        SimulationResult Run(std::vector<Request> requests);

    private:
        ModelConfig config_;
        IScheduler& scheduler_;
        IBlockSpaceManager& block_mgr_;
        IInferenceExecutor& executor_;
        PrefixCache* prefix_cache_;
        PreemptPolicy preempt_policy_ = PreemptPolicy::Self;
        bool decode_sync_batch_ = false;
    };
}