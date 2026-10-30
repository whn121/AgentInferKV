#include "SimulationLoop.h"

#include <algorithm>
#include <limits>


namespace agentinferkv
{
    namespace
    {
        // running 里每个请求的运行时状态。
        // next_event_time 是“下一个事件时刻”： 
        // PREFILL 中：Prefill 完成时刻
        // DECODE 中：下一步 Decode 完成时刻
        struct RunningRequest
        {
            Request req;
            BlockTableView view;
            double next_event_time;
        };

        constexpr double kEps = 1e-9; // 1的-9次方
    }

    SimulationLoop::SimulationLoop(ModelConfig config, IScheduler& scheduler, IBlockSpaceManager& block_mgr, IInferenceExecutor& executor)
         : config_ (config), scheduler_ (scheduler), block_mgr_ (block_mgr), executor_ (executor) {}

    SimulationResult SimulationLoop::Run (std::vector<Request> requests)
    {
        // 按照到达时间排序, 同一时刻按id
        std::sort(requests.begin(), requests.end(), [](const Request& a, const Request& b) 
            {
                if (a.arrival_time != b.arrival_time) 
                {
                    return a.arrival_time < b.arrival_time;
                }
                return a.id < b.id;
            });

        std::size_t next_arrival_idx = 0;
        std::vector<Request> waiting;
        std::vector<RunningRequest> running;
        std::vector<Request> completed;
        double sim_time = 0;
        int max_concurrent = 0;

        while (next_arrival_idx < requests.size() || !waiting.empty() || !running.empty())
        {
            // 第一步, 找下一个事件时间
            double next_time = std::numeric_limits<double>::infinity(); // 麦克斯奥特曼
            if (next_arrival_idx < requests.size())
            {
                next_time = std::min(next_time, requests[next_arrival_idx].arrival_time);
            }
            for (const auto& r : running)
            {
                next_time = std::min(next_time, r.next_event_time);
            }

            // 没有事件但有 waiting 中的请求：说明它们永远上不了（块不足）
            // 标记为取消，避免死循环
            if (next_time == std::numeric_limits<double>::infinity())
            {
                for (auto& r : waiting) 
                {
                    r.Cancel(sim_time);
                    completed.push_back(std::move(r));
                }
                waiting.clear();
                break;
            }

            // 第二步, 推进时间
            if (next_time > sim_time)
            {
                sim_time = next_time;
            }

            // 第三步, 处理到达事件 (可能同一时刻多个)
            while (next_arrival_idx < requests.size() && requests[next_arrival_idx].arrival_time <= sim_time + kEps)
            {
                waiting.push_back(requests[next_arrival_idx]);
                next_arrival_idx++;
            }

            // 第四步, 处理完成事件 (可能同一时刻多个)
            for (auto it = running.begin(); it != running.end(); )
            {
                if (it->next_event_time > sim_time + kEps) 
                {
                    ++it;
                    continue;
                }

                if (it->req.state == RequestState::kPrefill)
                {
                    it->req.FinishPrefill(sim_time);
                    if (it->req.state == RequestState::kDone)
                    {
                        // max_new_tokens == 1 的情况，Prefill 产出的首 token 就是最后一个
                        completed.push_back(it->req);
                        block_mgr_.Free(it->req.id);
                        it = running.erase(it);
                    }
                    else
                    {
                        it->next_event_time = sim_time + executor_.Decode(it->req, it->view);
                        ++it;
                    }
                }
                else if (it->req.state == RequestState::kDecode)
                {
                    it->req.DecodeOneToken(sim_time);
                    if (it->req.state == RequestState::kDone)
                    {
                        completed.push_back(it->req);
                        block_mgr_.Free(it->req.id);
                        it = running.erase(it);
                    }
                    else
                    {
                        // Paged需要在序列增长到新块边界时扩块; Naive 是 no-op (空操作)
                        int new_toal = it->req.PromptLen() + it->req.generated_tokens;
                        BlockTableView new_view;
                        if (!block_mgr_.AppendToken(it->req.id, new_toal, new_view))
                        {
                            // 扩块失败,块耗尽 第一版直接取消，标记为 OOM
                            it->req.Cancel(sim_time);
                            block_mgr_.Free(it->req.id);
                            completed.push_back(it->req);
                            it = running.erase(it);
                            continue;
                        }
                        it->view = new_view;  // 同步最新映射
                        it->next_event_time = sim_time + executor_.Decode(it->req, it->view);
                        ++it;
                    }
                }
                else
                {
                    ++it; // 理论上不会到这里
                }
            }

            // 第五步, 调用 (用 BlockBudget 模拟扣减)
            // 预算副本由 Loop 从当前空闲块数和 block_size 拼出来
            // 不依赖 IBlockSpaceManager 提供 MakeBudget，避免每个实现重复拼装
            BlockBudget budget;
            budget.free_blocks = block_mgr_.NumFreeBlocks();
            budget.blocks_needed = [this](const Request& r) 
            {
                return block_mgr_.EstimateBlocks(r);
            };
            std::vector<Request> running_snapshot;
            running_snapshot.reserve(running.size());
            for (const auto& r : running)
            {
                running_snapshot.push_back(r.req);
            }
            SchedulerOutput out = scheduler_.Schedule(waiting, running_snapshot, budget);

            // 第六步, 对每个准入的请求分块, 启动prefill
            for (RequestId id : out.scheduled_ids)
            {
                auto it = std::find_if (waiting.begin(), waiting.end(), [id](const Request& r){ return r.id == id; });
                if (it == waiting.end()) continue;

                BlockTableView view = block_mgr_.Allocate(*it);
                if (view.NumBlocks() == 0) continue;  // Scheduler 说能分，实际分不到

                it->StartPrefill(sim_time);
                
                RunningRequest rr;
                rr.req = *it;
                rr.view = view;
                rr.next_event_time = sim_time + executor_.Prefill(*it, view);
                running.push_back(std::move(rr));
                waiting.erase(it);
            }

            // 更新峰值并发
            int cur = static_cast<int>(running.size());
            if (cur > max_concurrent)
            {
                max_concurrent = cur;
            }
        }

        SimulationResult result;
        result.completed = std::move(completed);
        result.total_sim_time = sim_time;
        result.max_concurrent = max_concurrent;
        return result;
    }
}