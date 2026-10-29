#include "TraceReplayer.h"

namespace agentinferkv
{
    std::vector<Request> BuildRequests (const std::vector<TraceSpec>& specs)
    {
        std::vector<Request> out;
        out.reserve (specs.size());
        for (const auto& s : specs)
        {
            Request r;
            r.id = s.id;
            r.arrival_time = s.arrival_time;
            r.prompt_tokens.resize (static_cast<std::size_t> (s.prompt_len));
            for (int i = 0; i < s.prompt_len; ++i)
            {
                r.prompt_tokens[static_cast<std::size_t> (i)] = 1000 + static_cast<TokenId> (i);
            }
            r.max_new_tokens = s.max_new_tokens;
            out.push_back (std::move (r));
        }
        return out;
    }
}