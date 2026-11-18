#include "FCFSScheduler.h"
#include "Interfaces.h"
#include "PagedBlockManager.h"

#include <cassert>
#include <cstdio>

using namespace agentinferkv;

static Request MakeReq(RequestId id, int prompt_len)
{
    Request r;
    r.id = id;
    r.arrival_time = 0.0;
    r.max_new_tokens = 8;
    r.prompt_tokens.assign(static_cast<std::size_t>(prompt_len), 1);
    return r;
}

int main()
{
    ModelConfig cfg;
    cfg.block_size = 16;
    cfg.num_blocks = 4;

    PagedBlockManager mgr(cfg);

    Request r1 = MakeReq(1, 16);
    mgr.Allocate(r1, {});
    assert(mgr.NeedsBlockForNextToken(1));

    Request r2 = MakeReq(2, 48);
    FCFSScheduler sched;

    BlockBudget b_with;
    b_with.free_blocks = mgr.NumFreeBlocks() - 1;
    b_with.blocks_needed = [&mgr](const Request& r) { return mgr.EstimateBlocks(r); };
    auto out_with = sched.Schedule({r2}, {}, b_with);
    assert(out_with.scheduled_ids.empty());

    BlockBudget b_without;
    b_without.free_blocks = mgr.NumFreeBlocks();
    b_without.blocks_needed = [&mgr](const Request& r) { return mgr.EstimateBlocks(r); };
    auto out_without = sched.Schedule({r2}, {}, b_without);
    assert(out_without.scheduled_ids.size() == 1);

    Request r3 = MakeReq(3, 20);
    mgr.Allocate(r3, {});
    assert(!mgr.NeedsBlockForNextToken(3));

    std::printf("test_running_reservation: OK\n");
    return 0;
}
