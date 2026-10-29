#include "FCFSScheduler.h"

#include <algorithm> // 存放算法

namespace agentinferkv
{
    SchedulerOutput FCFSScheduler::Schedule(const std::vector<Request>& waiting, 
        const std::vector<Request>& running, const IBlockSpaceManager& block_manager)
    {
        (void)running; // 第一版不做抢占, running 里的请求一直占着块, 不重新调度

        // 按arrival_time 升序; 同一时刻按id升序, 保存确定性
        std::vector<const Request*> ordered;
        ordered.reserve(waiting.size());
        for (const auto& r : waiting)
        {
            ordered.push_back(&r);
        }
        std::sort(ordered.begin(), ordered.end(), [](const Request* a, Request* b)
        {
            if (a->arrival_time != b->arrival_time)
            {
                return a->arrival_time < b->arrival_time;
            }
            return a->id < b->id;
        });

        SchedulerOutput out;
        for (const Request* r : ordered)
        {
            if (!block_manager.CanAllocate(*r))
            {
                // FCFS 语义: 前面的请求分配不到, 后面的也就别插队了
                break;
            }
            out.scheduled_ids.push_back(r->id);
        }
        return out;
    }
}