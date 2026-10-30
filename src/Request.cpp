#include "Interfaces.h"

#include <sstream>

namespace agentinferkv
{
    const char* ToString(RequestState state)
    {
        switch (state)
        {
            case RequestState::kWaiting: return "WAITING";
            case RequestState::kPrefill: return "PREFILL";
            case RequestState::kDecode:    return "DECODE";
            case RequestState::kDone:      return "DONE";
            case RequestState::kCancelled: return "CANCELLED";
        }
        return "UNKNOWN";
    }

    bool IsTerminal(RequestState state)
    {
        return state == RequestState::kDone || state == RequestState::kCancelled;
    }

    bool CanTransition(RequestState from, RequestState to)
    {
        if (IsTerminal(from)) return false;
        switch (from) 
        {
            case RequestState::kWaiting: return to == RequestState::kPrefill || to == RequestState::kCancelled;
            case RequestState::kPrefill: return to == RequestState::kDecode || to == RequestState::kCancelled;
            case RequestState::kDecode: return to == RequestState::kDone || to == RequestState::kCancelled;
            default: return false;
        }
    }

    double Request::Ttft() const
    {
        if (first_token_time < 0.0) return -1.0;
        return first_token_time - arrival_time;
    }

    double Request::Latency() const
    {
        if (finish_time < 0.0) return -1.0;
        return finish_time - arrival_time;
    }

    std::string Request::DebugString() const
    {
        std::ostringstream oss;
        oss << "Request{id=" << id
        << ", state=" << ToString(state)
        << ", prompt_len=" << PromptLen()
        << ", generated=" << generated_tokens << "/" << max_new_tokens
        << ", ttft=" << Ttft()
        << ", latency=" << Latency() << "}";
        return oss.str();
    }

    bool Request::StartPrefill (double now)
    {
        (void) now; // 开始时刻暂不记录，TTFT 从首 token 时间算
        if (!CanTransition(state, RequestState::kPrefill)) return false;
        state = RequestState::kPrefill;
        return true;
    }

    bool Request::FinishPrefill (double now)
    {
        if (!CanTransition(state, RequestState::kDecode)) return false;
        state = RequestState::kDecode;
        generated_tokens = 1; // Prefill 产出首 token
        first_token_time = now;

        // max_new_tokens == 1 时，首 token 就是最后一个 token。
        // 经 DECODE 中间态再转 DONE，保持状态机只允许合法转换。
        if (generated_tokens >= max_new_tokens) 
        {
            state = RequestState::kDone;
            finish_time = now;
        }
        return true;
    }

    bool Request::DecodeOneToken (double now)
    {
        if (state != RequestState::kDecode) return false;
        ++generated_tokens;
        if (generated_tokens >= max_new_tokens)
        {
            state = RequestState::kDone;
            finish_time = now;
        }
        return true;
    }

    bool Request::Cancel (double now)
    {
        // 先检查, 在写finish_time, 顺序反了会在违法Cannel时污染时间戳
        if (!CanTransition(state, RequestState::kCancelled)) return false;
        state = RequestState::kCancelled;
        finish_time = now;
        return true;
    }
}