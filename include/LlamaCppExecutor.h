#pragma once

#include "Interfaces.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_sampler;

namespace agentinferkv
{
    // 真实 GPU 执行器, 用 llama.cpp 的 C API 驱动推理
    // 每个请求分配独立的 llama_seq_id, Prefill/Decode 各自调 llama_decode
    // 计时用 std::chrono, BlockTableView 参数被忽略——块账目在控制层
    // llama.cpp 内部有自己的 KV cache 管理
    class LlamaCppExecutor : public IInferenceExecutor
    {
    public:
        explicit LlamaCppExecutor(const std::string& model_path, int n_gpu_layers = 99, int n_ctx = 2048
                                    , int n_seq_max = 16);
        ~LlamaCppExecutor() override;
        double Prefill(const Request& req, const BlockTableView& view) override;
        double Decode(const Request& req, const BlockTableView& view) override;
        std::vector<double> DecodeBatch(const std::vector<const Request*>& reqs,
            const std::vector<BlockTableView>& views) override;
        void Release(RequestId id) override;

        bool IsEos(RequestId id) const override;

        // 用模型真实 tokenizer 把文本转 token 序列。
        // 供 main_real 填 prompt_tokens，保证 PrefixCache 匹配和真实 Prefill 用同一套 token。
        std::vector<int32_t> TokenizeText(const std::string& text) const;

        // ── 跨请求 KV 复用 ──
        // Prefill 成功后把该请求的 seq 登记为"前缀源"（按 chunk_tokens 对齐，
        // 默认 16，与 PrefixCache 的 block_size 一致）。后续请求命中相同前缀时，
        // 用 llama_memory_seq_cp 把 KV 复制过来，跳过那一段 Prefill。
        // 登记过的 seq 会被"钉住"：Release 不释放它的 KV，直到被淘汰。
        void SetChunkTokens(int chunk_tokens)
        {
            // 模型不支持时强制保持 0，避免调用方把复用又打开
            chunk_tokens_ = kv_reuse_supported_ ? chunk_tokens : 0;
        }
        void SetMaxPinnedSeqs(std::size_t n) { max_pinned_ = n; }

        int ReusedPrefixTokens() const { return reused_prefix_tokens_; }
        int ReuseHits() const { return reuse_hits_; }
        std::size_t PinnedSeqs() const { return pinned_.size(); }
        int EvictedPinnedSeqs() const { return evicted_pinned_; }

        // 当前模型是否支持"按位置部分复制 KV"。
        // hybrid / recurrent 模型（Mamba、RWKV、Jamba、Qwen3.5 等）的循环状态是
        // 对整段历史的摘要，不能按 position 区间截取，所以这类模型上关闭复用。
        bool KvReuseSupported() const { return kv_reuse_supported_; }

        // ── 模型真实形状 ──
        // ModelConfig 里的层数/头数/head_dim 必须来自模型本身，不能拍默认值，
        // 否则"每 token KV 字节""池子能放多少 token"全是错的。
        // main_real 用这三个值覆盖 ModelConfig。
        int NumLayers() const { return num_layers_; }
        int NumKvHeads() const { return num_kv_heads_; }
        int HeadDim() const { return head_dim_; }

        // 每个请求生成过的 token id。用来验证"开了 KV 复用之后 greedy 采样结果不变"。
        // Release 不清这里，方便跑完后对比。
        const std::vector<int32_t>* GeneratedIds(RequestId id) const
        {
            auto it = generated_ids_.find(id);
            return it == generated_ids_.end() ? nullptr : &it->second;
        }

    private:
        // 一个被钉住的 seq：它的 KV 覆盖 [0, n_tokens)，其中部分 block 对齐前缀已登记
        struct PinnedSeq
        {
            int seq_id = -1;
            int n_tokens = 0;
            std::vector<int> lengths;     // 已登记的前缀长度
            std::uint64_t stamp = 0;      // 最近一次被命中/登记的序号，用于淘汰
        };

        llama_model* model_ = nullptr;
        llama_context* ctx_ = nullptr;
        llama_sampler* sampler_ = nullptr;

        int next_seq_id_ = 0;
        int n_seq_max_ = 0;  
        std::unordered_map<RequestId, int> seq_map_; // RequestId -> llama_seq_id
        std::unordered_map<RequestId, int> n_past_; // RequestId -> 已处理 token 数
        std::unordered_map<RequestId, int32_t> pending_tokens_;  // 待喂入下一次 decode 的 token
        std::unordered_set<RequestId> eos_set_;   
        std::vector<int> free_seq_ids_; // 复用的 seq_id 池

        // 跨请求 KV 复用的状态
        bool kv_reuse_supported_ = false;

        // 模型真实形状（构造时从 llama_model 读，供 ModelConfig 对齐）
        int num_layers_ = 0;
        int num_kv_heads_ = 0;
        int head_dim_ = 0;
        int chunk_tokens_ = 16;
        std::size_t max_pinned_ = 3;
        std::uint64_t touch_clock_ = 0;
        int reused_prefix_tokens_ = 0;
        int reuse_hits_ = 0;
        int evicted_pinned_ = 0;
        std::vector<PinnedSeq> pinned_;
        // key = hash(前 len 个 token) -> (seq_id, len)
        std::unordered_map<std::uint64_t, std::pair<int, int>> prefix_index_;
        // seq_id -> 正在使用它的请求；有活跃请求的 seq 不能被淘汰
        std::unordered_map<int, RequestId> seq_owner_;

        // RequestId -> 生成的 token 序列（只用于结果校验，Release 不清）
        std::unordered_map<RequestId, std::vector<int32_t>> generated_ids_;

        std::vector<int> Tokenize(const std::string& text) const;
        int AcquireSeqId();

        static std::uint64_t PrefixKey(const std::vector<int32_t>& tokens, int len);
        PinnedSeq* FindPinned(int seq_id);
        const PinnedSeq* FindPinned(int seq_id) const;
        void RegisterPrefixes(int seq_id, const std::vector<int32_t>& tokens);
        void EvictPinnedIfNeeded();
    };
}
