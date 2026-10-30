#pragma once 

#include "Interfaces.h"

namespace agentinferkv
{
    // 对标vLLM v1默认 FCFS 调度
    // 只看arrival_time, 不看prompt长度, 不看前缀重复
    class FCFSScheduler : public IScheduler
    {
    public:
        SchedulerOutput Schedule(const std::vector<Request>& waiting, const std::vector<Request>& running,
            BlockBudget& budget) override;
    };
    
} 
