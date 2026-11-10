// 实验 1：同一 trace、同一物理块预算，只替换 BlockSpaceManager，
// 对比 Naive（按 max_len 预分配）与 Paged（按 prompt 分配、Decode 按需扩块）。
//
// 这个实验回答两个问题：
//   1) Paged 能不能提高“初始准入并发”；
//   2) 准入并发提高之后，完成率会不会因为 Decode 扩块失败而下降。
// 第 2 点是早期 README 漏掉的，也是“并发上限提升 2 倍”这个结论的真正边界。
//
// 为了让块预算成为唯一约束，所有请求在同一时刻到达（arrival = 0）。
// 如果请求按 10ms 间隔到达，系统是被到达率限制的，块预算根本不起作用。
#include "FCFSScheduler.h"
#include "NaivePreallocator.h"
#include "PagedBlockManager.h"
#include "SimulatedExecutor.h"
#include "SimulationLoop.h"
#include "TraceReplayer.h"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace agentinferkv;
static agentinferkv::SimulationLoop::PreemptPolicy g_policy =
    agentinferkv::SimulationLoop::PreemptPolicy::Self;
static bool g_sync_batch = false;


namespace
{
    struct Row
    {
        const char* strategy = "";
        int max_concurrent = 0;
        int peak_blocks = 0;
        int done = 0;
        int cancelled = 0;
        double avg_ttft = 0.0;
        double makespan = 0.0;
        int preempted = 0;
    };

    ModelConfig MakeConfig(int num_blocks)
    {
        ModelConfig c;
        c.num_layers = 32;
        c.num_kv_heads = 32;
        c.head_dim = 128;
        c.dtype_bytes = 2;
        c.block_size = 16;
        c.num_blocks = num_blocks;
        return c;
    }

    template <typename Manager>
    Row RunStrategy(const char* name, int num_blocks, const ExecutorConfig& exec_cfg,
                    const std::vector<TraceSpec>& trace)
    {
        ModelConfig cfg = MakeConfig(num_blocks);
        FCFSScheduler sched;
        Manager mgr(cfg);
        SimulatedExecutor exec(exec_cfg);
        SimulationLoop loop(cfg, sched, mgr, exec);
        loop.SetPreemptPolicy(g_policy);
        loop.SetDecodeSyncBatch(g_sync_batch);

        SimulationResult result = loop.Run(BuildRequests(trace));

        Row row;
        row.strategy = name;
        row.max_concurrent = result.max_concurrent;
        row.peak_blocks = result.peak_blocks_in_use;
        row.makespan = result.total_sim_time;
        row.preempted = result.preempted;

        double sum_ttft = 0.0;
        int n_ttft = 0;
        for (const auto& r : result.completed)
        {
            if (r.state == RequestState::kDone)
            {
                ++row.done;
                if (r.Ttft() > 0.0)
                {
                    sum_ttft += r.Ttft();
                    ++n_ttft;
                }
            }
            else
            {
                ++row.cancelled;
            }
        }
        row.avg_ttft = (n_ttft > 0) ? sum_ttft / n_ttft : 0.0;
        return row;
    }

    void PrintRow(const Row& row)
    {
        std::printf("| %-7s | %14d | %12d | %4d/%-2d | %9d | %9.2f | %10.4f |\n",
                    row.strategy,
                    row.max_concurrent,
                    row.peak_blocks,
                    row.done, row.done + row.cancelled,
                    row.preempted,
                    row.avg_ttft * 1000.0,
                    row.makespan);
    }

    void RunSuite(int num_blocks, const ExecutorConfig& exec_cfg, const std::vector<TraceSpec>& trace)
    {
        std::printf("\n-- blocks=%d (容量 %d token) --\n", num_blocks, num_blocks * 16);
        std::printf("| strategy | max_concurrent | peak_blocks | done | preempted | avg_ttft(ms) | makespan(s) |\n");
        std::printf("|----------|----------------|-------------|------|-----------|--------------|-------------|\n");

        PrintRow(RunStrategy<NaivePreallocator>("Naive", num_blocks, exec_cfg, trace));
        PrintRow(RunStrategy<PagedBlockManager>("Paged", num_blocks, exec_cfg, trace));
    }
}

int main(int argc, char** argv)
{
    // 用 SimulatedExecutor 的默认系数：它们来自真实 GPU 实测拟合
    // （RTX 4060 + Qwen2.5-0.5B，见 include/SimulatedExecutor.h 与 tools/calibrate_executor.cpp）。
    // 早期这里手写过 0.001/0.0001/0.0005/0.00002 一组"定性"系数，
    // 结果和真实测量对不上，等于两套数字打架——已删除。
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--sync-batch") == 0) g_sync_batch = true;
        if (std::strncmp(argv[i], "--preempt=", 10) == 0)
        {
            const char* v = argv[i] + 10;
            g_policy = (std::strcmp(v, "victim") == 0)
                ? agentinferkv::SimulationLoop::PreemptPolicy::Victim
                : agentinferkv::SimulationLoop::PreemptPolicy::Self;
        }
    }
    std::printf("preempt_policy=%s decode_batch=%s\n",
        g_policy == agentinferkv::SimulationLoop::PreemptPolicy::Victim ? "victim" : "self",
        g_sync_batch ? "sync" : "continuous");

    ExecutorConfig exec_cfg;

    // 6 个请求同时到达：prompt 20 + 最多生成 30 → 总 50 token
    // Naive 每请求预分配 ceil(50/16)=4 块；Paged 先分 ceil(20/16)=2 块，Decode 再扩
    std::vector<TraceSpec> trace;
    for (int i = 1; i <= 6; ++i)
    {
        TraceSpec spec;
        spec.id = static_cast<RequestId>(i);
        spec.arrival_time = 0.0;
        spec.prompt_len = 20;
        spec.max_new_tokens = 30;
        trace.push_back(spec);
    }

    std::printf("== 实验1: 同 trace 下 Naive vs Paged ==\n");
    std::printf("trace: 6 requests 同时到达, prompt=20, max_new=30, block_size=16\n");

    RunSuite(8, exec_cfg, trace);
    RunSuite(16, exec_cfg, trace);

    std::printf("\n读法: \n");
    std::printf("  - max_concurrent 是峰值并发请求数, peak_blocks 是峰值物理块占用；\n");
    std::printf("  - Paged 用更少的块换到更高的准入并发，这是 PagedAttention 的收益；\n");
    std::printf("  - 但块池被占满后 Decode 扩块会失败，本版本用“取消”代替抢占，\n");
    std::printf("    所以 done 会下降。要把 done 拉回 100%%，需要 preemption / swap, \n");
    std::printf("    这是本项目明确没做的部分。\n");
    return 0;
}
