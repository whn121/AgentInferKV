#pragma once

#include "Interfaces.h"

#include <vector>

namespace agentinferkv
{
    // 一条 trace 记录 对应nano-vLLM trace的四个核心字段
    struct TraceSpec
    {
        RequestId id;
        double arrival_time;
        int prompt_len;
        int max_new_tokens;
    };

    // 把 trace 转成 Request 列表。
    // prompt_tokens 用 1000 + i 生成占位 token，只保证长度正确；
    // 以后做 PrefixCache 时才会真正关心 token 内容。
    std::vector<Request> BuildRequests(const std::vector<TraceSpec>& specs);
}