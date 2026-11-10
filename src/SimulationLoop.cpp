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
            bool preempted = false;   // 被本轮抢占，循环结束后统一移除
        };

        constexpr double kEps = 1e-9; // 1的-9次方
    // 单个请求最多被抢占几次：防活锁（块永远不够时不能无限重排队）
    constexpr int kMaxPreemptPerRequest = 2;
    }

    SimulationLoop::SimulationLoop(ModelConfig config, IScheduler& scheduler, IBlockSpaceManager& block_mgr
                                    , IInferenceExecutor& executor, PrefixCache* prefix_cache)
         : config_ (config), scheduler_ (scheduler), block_mgr_ (block_mgr), executor_ (executor)
            ,  prefix_cache_(prefix_cache) {}

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
        std::vector<Request> requeue;                 // 被抢占后重新排队的请求
        std::unordered_map<RequestId, int> preempt_count;
        int preempted_total = 0;
        double sim_time = 0;
        int max_concurrent = 0;
        int peak_blocks = 0;

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
            // 4a 先处理 PREFILL 完成
            {
                std::vector<std::size_t> to_remove;
                for (std::size_t i = 0; i < running.size(); ++i) 
                {
                    auto& r = running[i];
                    if (r.next_event_time > sim_time + kEps) continue;
                    if (r.req.state != RequestState::kPrefill) continue;

                    r.req.FinishPrefill(sim_time);
                    if (prefix_cache_ != nullptr) 
                    {
                        auto new_blocks = prefix_cache_->Insert(r.req.prompt_tokens, r.view.logical_to_physical,
                            config_.block_size);
                        for (BlockId bid : new_blocks) 
                        {
                            block_mgr_.RetainSharedBlock(bid);
                        }
                    }
                    if (r.req.state == RequestState::kDone) 
                    {
                        executor_.Release(r.req.id);
                        if (prefix_cache_ != nullptr) 
                        {
                            prefix_cache_->Release(r.req.prompt_tokens, config_.block_size);
                        }
                        completed.push_back(r.req);
                        block_mgr_.Free(r.req.id);
                        to_remove.push_back(i);
                    } 
                    else 
                    {
                        r.next_event_time = sim_time + executor_.Decode(r.req, r.view);
                    }
                }
                for (auto it = to_remove.rbegin(); it != to_remove.rend(); ++it) 
                {
                    running.erase(running.begin() + *it);
                }
            }

            // 4b 收集所有该 DECODE 的请求，批量执行
            {
                std::vector<std::size_t> to_remove;
                std::vector<std::size_t> decode_ready;

                for (std::size_t i = 0; i < running.size(); ++i) 
                {
                    auto& r = running[i];
                    if (r.next_event_time > sim_time + kEps) continue;
                    if (r.req.state != RequestState::kDecode) continue;

                    r.req.DecodeOneToken(sim_time);
                    if (r.req.state == RequestState::kDone) 
                    {
                        executor_.Release(r.req.id);
                        if (prefix_cache_ != nullptr) 
                        {
                            prefix_cache_->Release(r.req.prompt_tokens, config_.block_size);
                        }
                        completed.push_back(r.req);
                        block_mgr_.Free(r.req.id);
                        to_remove.push_back(i);
                        continue;
                    }

                    int new_total = r.req.PromptLen() + r.req.generated_tokens;
                    BlockTableView new_view;
                    if (r.preempted) continue;

                    // Victim 策略：优先抢"后到的、还没处理的"请求（LIFO），
                    // 只选下标 > i 的，保证它不会已经进了本轮 decode_ready。
                    bool resolved = false;
                    if (preempt_policy_ == PreemptPolicy::Victim)
                    {
                        std::size_t victim = running.size();
                        for (std::size_t k = i + 1; k < running.size(); ++k)
                        {
                            // 只能抢已经进入 decode 的请求：
                            // prefill 阶段的请求还没做过 PrefixCache::Insert，
                            // 对它调 Release 会错误地减掉别人的引用计数。
                            if (running[k].preempted) continue;
                            if (running[k].req.state != RequestState::kDecode) continue;
                            if (victim == running.size() || running[k].req.id > running[victim].req.id)
                            {
                                victim = k;
                            }
                        }

                        if (victim != running.size())
                        {
                            auto& v = running[victim];
                            block_mgr_.Free(v.req.id);
                            if (prefix_cache_ != nullptr)
                            {
                                prefix_cache_->Release(v.req.prompt_tokens, config_.block_size);
                            }
                            executor_.Release(v.req.id);
                            v.preempted = true;

                            Request again = v.req;
                            again.state = RequestState::kWaiting;
                            again.generated_tokens = 0;
                            again.first_token_time = -1.0;
                            again.finish_time = -1.0;
                            again.arrival_time = sim_time + kEps;
                            if (++preempt_count[again.id] <= kMaxPreemptPerRequest)
                            {
                                requeue.push_back(again);
                                ++preempted_total;
                            }
                            else
                            {
                                again.state = RequestState::kCancelled;
                                completed.push_back(again);
                            }

                            resolved = block_mgr_.AppendToken(r.req.id, new_total, new_view);
                            if (resolved)
                            {
                                r.view = new_view;
                                decode_ready.push_back(i);
                                continue;
                            }
                        }
                    }

                    if (!resolved && !block_mgr_.AppendToken(r.req.id, new_total, new_view)) 
                    {
                        // ── Recompute 抢占（简化版：抢占"自己"）──
                        // 块不够时不再直接取消，而是释放自己的块、丢掉已生成 token、
                        // 重新排队，等有空间时从 prompt 重新 prefill。
                        // 为什么不踢别人：FCFS 下"自己让路 + 推迟到达时间"就足以让
                        // 其他请求推进，且不会把别人的进度也毁掉（见 README 抢占小节）。
                        executor_.Release(r.req.id);
                        if (prefix_cache_ != nullptr) 
                        {
                            prefix_cache_->Release(r.req.prompt_tokens, config_.block_size);
                        }
                        block_mgr_.Free(r.req.id);

                        Request again = r.req;
                        again.state = RequestState::kWaiting;
                        again.generated_tokens = 0;
                        again.first_token_time = -1.0;
                        again.finish_time = -1.0;
                        // 推迟到达时间 → 排到队尾，避免"立刻又被选中、反复抢占自己"
                        again.arrival_time = sim_time + kEps;

                        if (++preempt_count[again.id] <= kMaxPreemptPerRequest)
                        {
                            requeue.push_back(again);
                            ++preempted_total;
                        }
                        else
                        {
                            // 抢占次数用尽仍然不够 → 判定为该请求无法完成
                            again.state = RequestState::kCancelled;
                            completed.push_back(again);
                        }
                        to_remove.push_back(i);
                        continue;
                    }
                    r.view = new_view;
                    decode_ready.push_back(i);
                }

                // 一次性批量执行
                if (!decode_ready.empty()) 
                {
                    std::vector<const Request*> batch_reqs;
                    std::vector<BlockTableView> batch_views;
                    batch_reqs.reserve(decode_ready.size());
                    batch_views.reserve(decode_ready.size());
                    for (std::size_t i : decode_ready) 
                    {
                        batch_reqs.push_back(&running[i].req);
                        batch_views.push_back(running[i].view);
                    }
                    std::vector<double> costs = executor_.DecodeBatch(batch_reqs, batch_views);

                    // 同步 batch 对照：整批等最慢的那个。
                    // 连续批处理下每个请求只等自己的耗时（costs[j] 原样使用）。
                    if (decode_sync_batch_ && !costs.empty())
                    {
                        double worst = 0.0;
                        for (double c : costs) worst = std::max(worst, c);
                        for (double& c : costs) c = worst;
                    }
                    for (std::size_t j = 0; j < decode_ready.size(); ++j) 
                    {
                        std::size_t idx = decode_ready[j];
                        if (costs[j] < 0.0) 
                        {
                            // 执行失败: 取消
                            running[idx].req.Cancel(sim_time);
                            executor_.Release(running[idx].req.id);
                            if (prefix_cache_ != nullptr) 
                            {
                                prefix_cache_->Release(running[idx].req.prompt_tokens, config_.block_size);
                            }
                            completed.push_back(running[idx].req);
                            block_mgr_.Free(running[idx].req.id);
                            to_remove.push_back(idx);
                        } 
                        else
                        {
                            if (executor_.IsEos(running[idx].req.id))
                            {
                                running[idx].req.FinishByEos(sim_time);
                            }
                            if (running[idx].req.state == RequestState::kDone)
                            {
                                executor_.Release(running[idx].req.id);
                                if (prefix_cache_ != nullptr)
                                {
                                    prefix_cache_->Release(
                                        running[idx].req.prompt_tokens,
                                        config_.block_size);
                                }
                                completed.push_back(running[idx].req);
                                block_mgr_.Free(running[idx].req.id);
                                to_remove.push_back(idx);
                            }
                            else
                            {
                                running[idx].next_event_time = sim_time + costs[j];
                            }
                        }
                    }
                }

                // 把被 Victim 抢占的项加进待移除（它们已经被 Free/Release 过）
                for (std::size_t k = 0; k < running.size(); ++k)
                {
                    if (running[k].preempted) to_remove.push_back(k);
                }

                for (auto it = to_remove.rbegin(); it != to_remove.rend(); ++it) 
                {
                    running.erase(running.begin() + *it);
                }

                // 被抢占的请求回到等待队列（下一轮调度）
                for (auto& q : requeue) waiting.push_back(std::move(q));
                requeue.clear();
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

                // 查前缀
                std::vector<BlockId> prefix_blocks;
                if (prefix_cache_ != nullptr) 
                {
                    auto raw = prefix_cache_->Match(it->prompt_tokens, config_.block_size);
                    for (BlockId bid : raw.matched_blocks) 
                    {
                        if (!block_mgr_.IsBlockAlive(bid)) break;  // stale 引用, 路径断
                        prefix_blocks.push_back(bid);
                    }
                    // Retain 共享块的引用
                    for (BlockId bid : prefix_blocks) 
                    {
                        block_mgr_.RetainSharedBlock(bid);
                    }
                    it->cached_prefix_len = static_cast<int>(prefix_blocks.size()) * config_.block_size;
                }

                BlockTableView view = block_mgr_.Allocate(*it, prefix_blocks);
                // Scheduler 说能分，实际分不到
                if (view.NumBlocks() == 0)
                {
                    for (BlockId bid : prefix_blocks)
                    {
                        block_mgr_.ReleaseSharedBlock(bid);
                    }
                    continue;
                }

                it->StartPrefill(sim_time);
                
                RunningRequest rr;
                rr.req = *it;
                rr.view = view;
                double prefill_cost = executor_.Prefill(*it, view);
                if (prefill_cost < 0.0) 
                {
                    // 执行失败 取消请求，释放资源
                    // 这条路径上 Insert 还没执行过 所以不能调 PrefixCache::Release
                    // 否则会把别的请求的节点引用计数减掉
                    it->Cancel(sim_time);
                    executor_.Release(it->id);
                    completed.push_back(*it);
                    block_mgr_.Free(it->id);
                    waiting.erase(it);
                    continue;
                }
                rr.next_event_time = sim_time + prefill_cost;
                running.push_back(std::move(rr));
                waiting.erase(it);
            }

            // 更新峰值并发
            int cur = static_cast<int>(running.size());
            if (cur > max_concurrent)
            {
                max_concurrent = cur;
            }

            // 峰值块占用 Paged 与 Naive 的碎片差异直接体现在这里
            const int used_blocks = config_.num_blocks - block_mgr_.NumFreeBlocks();
            if (used_blocks > peak_blocks)
            {
                peak_blocks = used_blocks;
            }
        }

        SimulationResult result;
        result.completed = std::move(completed);
        result.total_sim_time = sim_time;
        result.preempted = preempted_total;
        result.max_concurrent = max_concurrent;
        result.peak_blocks_in_use = peak_blocks;
        return result;
    }
}