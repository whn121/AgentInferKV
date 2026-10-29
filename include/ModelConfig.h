#pragma once

#include <cstddef>
#include <cstdint>

namespace agentinferkv
{
    // 模型与KVCahe的物理配置
    // 显存公式参照vLLM的KVCache形状 : 
    // [num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim]
    // 所有计算都从这里出发, 不允许在别处写死block数量
    struct ModelConfig
    {
        int num_layers = 32;   // Transformer层数
        int num_kv_heads = 8;  // GQA时KVhead数, MHA时等于num_heads
        int head_dim = 128;    // 每个head的维度
        int dtype_bytes = 2;   // fp16 = 2, fp32 = 4, int8 = 1;

        int block_size = 16;   // 每个物理block容纳的token数
        int num_blocks = 1024; // 显存中物理block总数

        // 单个token的k+v字节数
        // 2表示k和v各一份
        std::size_t KvBytesPerToken() const
        {
            return 2ull * static_cast<std::size_t>(num_layers) * 
            static_cast<std::size_t>(num_kv_heads) *
            static_cast<std::size_t>(head_dim) * 
            static_cast<std::size_t>(dtype_bytes);
        }

        // 一个物理block的容量(token数)
        int TokenPerBlock() const { return block_size; }

        // 显存池可容纳的总token数
        std::size_t TotalTokenCapacity() const
        {
            return static_cast<std::size_t>(block_size) * 
            static_cast<std::size_t>(num_blocks);
        }

        // 给定序列长度, 单请求的KV字节数
        std::size_t RequestKvBytes(int seq_len) const
        {
            return KvBytesPerToken() * static_cast<std::size_t>(seq_len);
        }

        // 给定token,向上取block数
        int BlocksForTokens(int num_tokens) const
        {
            if (num_tokens <= 0) return 0;
            return (num_tokens + block_size - 1) / block_size;
        }
    };
}