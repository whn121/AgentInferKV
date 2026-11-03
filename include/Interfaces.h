#pragma once

#include "ModelConfig.h"

#include <cstdint>
#include <string>
#include <vector>
#include <functional>


namespace agentinferkv
{
    using RequestId = std::uint64_t;
    using BlockId = std::int32_t;
    using TokenId = std::int32_t;

    // 合法转换只有四条路：
    //   WAITING → PREFILL → DECODE → DONE
    //   任意非终态 → CANCELLED
    // 终态（DONE / CANCELLED）不能再转换；具体判定见 Request.cpp 的 CanTransition。
    enum class RequestState
    {
        kWaiting,     // 已到达，等待调度准入
        kPrefill,     // 正在执行 prefill
        kDecode,      // 正在逐 token decode
        kDone,        // 正常完成
        kCancelled,   // 被取消（取消/超时/资源不足终止等）
    };

    const char* ToString(RequestState state);
    bool IsTerminal(RequestState state);

    // 合法状态转换的唯一判定入口
    bool CanTransition(RequestState from , RequestState to);

    struct Request
    {
        RequestId id = 0;                            // 全局唯一
        double arrival_time = 0.0;                   // 模拟时钟下的到达时间
        std::vector<TokenId> prompt_tokens;          // 已tokenize的输入
        int max_new_tokens = 0;                      // 最多生成多少token

        RequestState state = RequestState::kWaiting;
        int generated_tokens = 0;                    // 已生成的token数
        std::string prompt_text;     // 真实文本LlamaCppExecutor 用它 tokenize
        int cached_prefix_len = 0;   // 命中的前缀 token 数, Prefill 时跳过

        double first_token_time = -1.0;              // 首个 token 产出时刻，-1 表示未产出
        double finish_time = -1.0;                   // DONE/CANCELLED 时刻，-1 表示未结束

        // prompt 长度（token 数）
        int PromptLen() const { return static_cast<int>(prompt_tokens.size()); }
        // prompt + max_new_tokens 用于预估总块需求
        int TotalTokens() const { return PromptLen() + max_new_tokens; }

        // 首 token 延迟，未产出返回 -1
        double Ttft() const;

        // 从到达到结束的总延迟，未结束返回 -1
        double Latency() const;

        // 简洁,用于日志
        std::string DebugString() const;

        // 状态推进。全部走 CanTransition 校验，非法转换返回 false，状态不变。
        bool StartPrefill(double now);
        bool FinishPrefill(double now);
        bool DecodeOneToken(double now);
        bool Cancel(double now);
        // 收到 EOS 提前结束。
        bool FinishByEos(double now);
    };
    
    // 逻辑块 -> 物理块 的只读映射
    // BlockSpaceManager 分配时生成，Scheduler/Executor 只读不改
    struct BlockTableView
    {
        std::vector<BlockId> logical_to_physical;

        int NumBlocks() const { return static_cast<int>(logical_to_physical.size()); }

        // 第 logical_idx 个逻辑块对应的物理块 ID
        BlockId PhysicalBlock(int logical_idx) const
        {
            return logical_to_physical.at(static_cast<std::size_t> (logical_idx));
        }
    };

    // Scheduler 用的预算副本
    // blocks_needed 由调用方注入：Naive 按 TotalTokens 估，Paged 按 PromptLen 估
    // 只影响副本，不影响真实池。真实池只由 Allocate/Free 扣/还
    struct BlockBudget 
    {
        int free_blocks = 0;
        std::function<int(const Request&)> blocks_needed;

        // 扣减这个请求需要的块；不够返回 false，不扣。
        bool TryReserveForRequest(const Request& req) 
        {
            int need = blocks_needed ? blocks_needed(req) : 0;
            if (need < 0 || free_blocks < need) return false;
            free_blocks -= need;
            return true;
        }
    };

    // Scheduler 每步的输出：本步允许进入运行的请求 ID
    // 不包含物理块，物理块由 BlockSpaceManager 独立分配
    struct SchedulerOutput
    {
        std::vector<RequestId> scheduled_ids;
    };
    
    // 物理块生命周期的唯一, 管理所有物理块的分配, 释放, 引用计数
    class IBlockSpaceManager
    {
    public:
        virtual ~IBlockSpaceManager() = default;

        // Scheduler 准入前调用：当前空闲块是否够这个请求用
        // 返回 true 不代表已分配，只是“能分”
        // prefix_blocks 默认空，表示无前缀命中
        virtual bool CanAllocate(const Request& req, const std::vector<BlockId>& prefix_blocks = {}) const = 0;

        // 真正分配块, 返回请求的块视图
        // 调用前必须确保 CanAllocate 为 true，否则返回空视图
        // 分配块：view = prefix_blocks + 后缀新块
        // prefix_blocks 的引用计数已由调用方 Retain 过，这里不再加
        virtual BlockTableView Allocate(const Request& req, const std::vector<BlockId>& prefix_blocks = {}) = 0;

         // 块是否仍在分配中(ref_count >= 1)用于过滤 PrefixCache 的 stale 引用。
        virtual bool IsBlockAlive(BlockId id) const = 0;

        // 请求结束(完成或取消)时调用，释放该请求独占的块
        // 共享块只减引用计数，不减到 0 不回池
        virtual void Free(RequestId id) = 0;

        // 请求需要扩到 new_total_tokens 个 token 的容量
        // Naive 实现是 no-op（已经按 max_len 分满），Paged 按需补块
        // 成功时把最新的块视图写入 out_view；失败返回 false，out_view 不变
        virtual bool AppendToken(RequestId id, int new_total_tokens, BlockTableView& out_view) = 0;

        // 增加一个块的引用计数。共享前缀时调用。
        // 块必须已分配（ref_count >= 1），否则是 bug。
        virtual void RetainSharedBlock(BlockId id) = 0;

        // 减少一个块的引用计数。和 RetainSharedBlock 成对使用。
        // 减到 0 时回池。
        virtual void ReleaseSharedBlock(BlockId id) = 0;

        // 返回“如果准入这个请求，预估需要多少块”
        // Naive 用 prompt + max_new，Paged 用 prompt
        // Scheduler 的预算扣减依赖它，保证“估算”和“实际分配”口径一致
        virtual int EstimateBlocks(const Request& req) const = 0;

        // 当前空闲物理块数，用于观测和背压判断
        virtual int NumFreeBlocks() const = 0;
    };

    // 调度策略: 决定哪些waiting请求,按什么顺序进入running
    // 只看请求元信息和可用块数，不接触物理块，不做模型计算
    class IScheduler
    {
    public:
        virtual ~IScheduler() = default;

        // budget 是可变引用：Scheduler 按顺序模拟扣减
        // 返回的 id 列表保证在扣减语义下都能分配
        virtual SchedulerOutput Schedule(const std::vector<Request>& waiting,
        const std::vector<Request>& running, BlockBudget& budget) = 0;
    };

    // 执行器：负责一次 Prefill 或一次 Decode 的计算边界。
    // 模拟阶段返回模拟耗时，接真实引擎时只换实现，Scheduler 不动
    class IInferenceExecutor
    {
    public:
        virtual ~IInferenceExecutor() = default;

        // 对一个请求做一次 Prefill, 返回耗时
        // view 用来定位请求kv块
        virtual double Prefill(const Request& req, const BlockTableView& view) = 0;

        // 对请求做一步 Decode (生成一个token), 返回耗时
        virtual double Decode(const Request& req, const BlockTableView& view) = 0;

        // 批量 Decode, 多个请求的 decode 一步合并成一次执行
        // 返回的 costs[i] 是第 i 个请求这一步的耗时
        // 单个 Decode 可以视为 batch=1 的特例
        virtual std::vector<double> DecodeBatch(const std::vector<const Request*>& reqs,
                                            const std::vector<BlockTableView>& views) = 0;

        // 请求完成/取消时释放执行侧资源(KV seq、采样状态等)
        // SimulatedExecutor 是空实现,LlamaCppExecutor 释放 seq_id 和 KV cache
        virtual void Release(RequestId id) = 0;

        // 该请求是否已采样到 EOS。默认 false，SimulatedExecutor 不实现。
        virtual bool IsEos(RequestId id) const { (void)id; return false; }
    };

}