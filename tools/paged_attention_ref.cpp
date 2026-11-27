// 分页 KV 的 attention 参考实现：证明 "block table 参与 attention 访存" 到底是怎么发生的。
//
// 目的（面试可讲）：
//   1. 连续 KV：K/V 按 [seq][dim] 存，attention 直接按位置取。
//   2. 分页 KV：K/V 按 [物理块][块内偏移][dim] 存，逻辑位置 p 要先查 block_table[p/bs] 得到物理块，
//      再用 p%bs 取块内偏移。这就是 PagedAttention 的 gather 过程。
//   3. 两者结果必须一致（误差 1e-6 内），说明分页只是"寻址方式变了"，不影响数学。
//
// 编译：随仓库构建，产物 agentinferkv_paged_attn_ref

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace
{
    constexpr int kSeqLen = 64;
    constexpr int kBlockSize = 16;
    constexpr int kNumLogicalBlocks = kSeqLen / kBlockSize;   // 4
    constexpr int kNumPhysBlocks = 8;                          // 故意比逻辑块多，模拟"物理块乱序"
    constexpr int kHeadDim = 8;

    // 连续布局：K[pos][d]
    struct ContiguousKv
    {
        std::vector<std::vector<float>> k;   // [seq][dim]
        std::vector<std::vector<float>> v;
    };

    // 分页布局：K[phys_block][offset][d]
    struct PagedKv
    {
        std::vector<std::vector<std::vector<float>>> k;  // [phys][bs][dim]
        std::vector<std::vector<std::vector<float>>> v;
    };

    float Dot(const std::vector<float>& a, const std::vector<float>& b)
    {
        float s = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
        return s;
    }

    // 对所有位置做 softmax 加权求和
    std::vector<float> Attention(const std::vector<float>& q,
                                 const std::vector<std::vector<float>>& keys,
                                 const std::vector<std::vector<float>>& values)
    {
        const std::size_t n = keys.size();
        std::vector<float> logits(n);
        float max_logit = -1e30f;
        for (std::size_t i = 0; i < n; ++i)
        {
            logits[i] = Dot(q, keys[i]) / std::sqrt(static_cast<float>(kHeadDim));
            if (logits[i] > max_logit) max_logit = logits[i];
        }

        std::vector<float> out(kHeadDim, 0.0f);
        float denom = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
        {
            const float w = std::exp(logits[i] - max_logit);
            denom += w;
            for (int d = 0; d < kHeadDim; ++d) out[d] += w * values[i][d];
        }
        for (int d = 0; d < kHeadDim; ++d) out[d] /= denom;
        return out;
    }
}  // namespace

int main()
{
    std::mt19937 rng(20261127);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 1) 生成一份"逻辑"KV
    ContiguousKv cont;
    cont.k.assign(kSeqLen, std::vector<float>(kHeadDim));
    cont.v.assign(kSeqLen, std::vector<float>(kHeadDim));
    for (int p = 0; p < kSeqLen; ++p)
        for (int d = 0; d < kHeadDim; ++d)
        {
            cont.k[p][d] = dist(rng);
            cont.v[p][d] = dist(rng);
        }

    // 2) 造一张乱序的 block table：逻辑块 i -> 物理块 table[i]
    std::vector<int> free_phys(kNumPhysBlocks);
    for (int i = 0; i < kNumPhysBlocks; ++i) free_phys[i] = i;
    std::shuffle(free_phys.begin(), free_phys.end(), rng);
    std::vector<int> block_table(free_phys.begin(), free_phys.begin() + kNumLogicalBlocks);

    // 3) 按 block table 铺到物理块里
    PagedKv paged;
    paged.k.assign(kNumPhysBlocks,
                   std::vector<std::vector<float>>(kBlockSize, std::vector<float>(kHeadDim)));
    paged.v.assign(kNumPhysBlocks,
                   std::vector<std::vector<float>>(kBlockSize, std::vector<float>(kHeadDim)));

    for (int p = 0; p < kSeqLen; ++p)
    {
        const int logical = p / kBlockSize;
        const int offset = p % kBlockSize;
        const int phys = block_table[static_cast<std::size_t>(logical)];
        for (int d = 0; d < kHeadDim; ++d)
        {
            paged.k[static_cast<std::size_t>(phys)][static_cast<std::size_t>(offset)][static_cast<std::size_t>(d)] = cont.k[p][d];
            paged.v[static_cast<std::size_t>(phys)][static_cast<std::size_t>(offset)][static_cast<std::size_t>(d)] = cont.v[p][d];
        }
    }

    // 4) 连续版本直接算
    std::vector<float> q(kHeadDim);
    for (int d = 0; d < kHeadDim; ++d) q[d] = dist(rng);
    const std::vector<float> out_cont = Attention(q, cont.k, cont.v);

    // 5) 分页版本：按 block table gather 出逻辑顺序，再算
    std::vector<std::vector<float>> gathered_k(kSeqLen);
    std::vector<std::vector<float>> gathered_v(kSeqLen);
    for (int p = 0; p < kSeqLen; ++p)
    {
        const int logical = p / kBlockSize;
        const int offset = p % kBlockSize;
        const int phys = block_table[static_cast<std::size_t>(logical)];
        gathered_k[static_cast<std::size_t>(p)] =
            paged.k[static_cast<std::size_t>(phys)][static_cast<std::size_t>(offset)];
        gathered_v[static_cast<std::size_t>(p)] =
            paged.v[static_cast<std::size_t>(phys)][static_cast<std::size_t>(offset)];
    }
    const std::vector<float> out_paged = Attention(q, gathered_k, gathered_v);

    // 6) 比较
    float max_diff = 0.0f;
    for (int d = 0; d < kHeadDim; ++d)
        max_diff = std::fmax(max_diff, std::fabs(out_cont[static_cast<std::size_t>(d)] -
                                                 out_paged[static_cast<std::size_t>(d)]));

    std::printf("block_table (logical -> physical): ");
    for (int i = 0; i < kNumLogicalBlocks; ++i) std::printf("%d ", block_table[static_cast<std::size_t>(i)]);
    std::printf("\n");
    std::printf("seq=%d block_size=%d head_dim=%d\n", kSeqLen, kBlockSize, kHeadDim);
    std::printf("连续 KV 输出[0..3]: %.6f %.6f %.6f %.6f\n",
                out_cont[0], out_cont[1], out_cont[2], out_cont[3]);
    std::printf("分页 KV 输出[0..3]: %.6f %.6f %.6f %.6f\n",
                out_paged[0], out_paged[1], out_paged[2], out_paged[3]);
    std::printf("max_abs_diff = %.9f\n", max_diff);
    std::printf("%s\n", max_diff < 1e-6f ? "PASS: 分页寻址与连续 KV 等价" : "FAIL");

    return max_diff < 1e-6f ? 0 : 1;
}
