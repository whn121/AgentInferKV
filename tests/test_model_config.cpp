#include "ModelConfig.h"

#include <cstdio>

using namespace agentinferkv;

static int g_failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("CHECK failed: %s at line %d\n", #cond, __LINE__); \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

int main() {
    // 用 LLaMA-2-7B 的真实配置：32 层，32 KV heads，head_dim 128，fp16
    // 单 token K+V = 2 * 32 * 32 * 128 * 2 = 524288 字节 = 512 KB
    ModelConfig llama2_7b;
    llama2_7b.num_layers = 32;
    llama2_7b.num_kv_heads = 32;
    llama2_7b.head_dim = 128;
    llama2_7b.dtype_bytes = 2;

    CHECK(llama2_7b.KvBytesPerToken() == 512 * 1024);

    // 4096 token 的单请求：512KB * 4096 = 2 GB
    CHECK(llama2_7b.RequestKvBytes(4096) == 2ull * 1024 * 1024 * 1024);

    // 块数向上取整
    llama2_7b.block_size = 16;
    CHECK(llama2_7b.BlocksForTokens(0) == 0);
    CHECK(llama2_7b.BlocksForTokens(1) == 1);
    CHECK(llama2_7b.BlocksForTokens(16) == 1);
    CHECK(llama2_7b.BlocksForTokens(17) == 2);
    CHECK(llama2_7b.BlocksForTokens(32) == 2);
    CHECK(llama2_7b.BlocksForTokens(33) == 3);

    // 总容量
    llama2_7b.num_blocks = 1024;
    CHECK(llama2_7b.TotalTokenCapacity() == 1024ull * 16);

    if (g_failures == 0) {
        std::printf("All ModelConfig tests passed.\n");
        return 0;
    }
    std::printf("%d checks failed.\n", g_failures);
    return 1;
}