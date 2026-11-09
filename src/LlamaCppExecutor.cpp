#include "LlamaCppExecutor.h"

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace agentinferkv 
{

    LlamaCppExecutor::LlamaCppExecutor(const std::string& model_path, int n_gpu_layers, int n_ctx, int n_seq_max)
         : n_seq_max_ (n_seq_max), n_ctx_ (n_ctx)
    {
        llama_backend_init();

        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = n_gpu_layers;
        model_ = llama_model_load_from_file(model_path.c_str(), mparams);
        if (model_ == nullptr) 
        {
            throw std::runtime_error("LlamaCppExecutor: failed to load model: " + model_path);
        }

        // 读出模型真实形状：ModelConfig 必须与它对齐，否则容量估算全错。
        num_layers_ = static_cast<int>(llama_model_n_layer(model_));
        num_kv_heads_ = static_cast<int>(llama_model_n_head_kv(model_));
        const int n_embd = static_cast<int>(llama_model_n_embd(model_));
        const int n_head = static_cast<int>(llama_model_n_head(model_));
        head_dim_ = (n_head > 0) ? (n_embd / n_head) : 0;
        std::fprintf(stderr,
            "[LlamaCppExecutor] model shape: layers=%d kv_heads=%d head_dim=%d (n_embd=%d n_head=%d)\n",
            num_layers_, num_kv_heads_, head_dim_, n_embd, n_head);

        // 判断这个模型能不能做"按位置部分复制 KV"。
        // hybrid / recurrent 模型的循环状态是整段历史的摘要，不能按 position 截取，
        // 硬做会得到错误的 KV（实测 Qwen3.5-4B 上 greedy 采样结果会变）。
        const bool is_hybrid = llama_model_is_hybrid(model_);
        const bool is_recurrent = llama_model_is_recurrent(model_);
        kv_reuse_supported_ = !is_hybrid && !is_recurrent;
        if (!kv_reuse_supported_)
        {
            chunk_tokens_ = 0;   // 主动禁用前缀登记与复用
            std::fprintf(stderr,
                "[LlamaCppExecutor] KV reuse disabled: model is %s; "
                "recurrent/hybrid state cannot be sliced by position.\n",
                is_hybrid ? "hybrid" : "recurrent");
        }

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = n_ctx;
        cparams.n_batch = 512;
        cparams.n_seq_max = n_seq_max;
        // 关键：统一 KV 缓冲。这样所有 seq 落在同一个 stream，
        // llama_memory_seq_cp 走"共享 cell"路径（打标记，不拷数据），
        // 跨请求 KV 复用才成立；否则 n_stream == n_seq_max，跨 stream
        // 复制只支持整块，p1 取中间值会直接 GGML_ASSERT abort。
        cparams.kv_unified = kv_reuse_supported_;
        ctx_ = llama_init_from_model(model_, cparams);
        if (ctx_ == nullptr) 
        {
            llama_model_free(model_);
            throw std::runtime_error("LlamaCppExecutor: failed to create context");
        }

        sampler_ = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());
    }

    LlamaCppExecutor::~LlamaCppExecutor() 
    {
        if (sampler_ != nullptr) llama_sampler_free(sampler_);
        if (verify_ctx_ != nullptr) llama_free(verify_ctx_);
        if (ctx_ != nullptr) llama_free(ctx_);
        if (model_ != nullptr) llama_model_free(model_);
        llama_backend_free();
    }

    int LlamaCppExecutor::AcquireSeqId() 
    {
        if (!free_seq_ids_.empty()) 
        {
            int id = free_seq_ids_.back();
            free_seq_ids_.pop_back();
            return id;
        }   
        if (next_seq_id_ >= n_seq_max_) 
        {
            return -1;   // 容量耗尽
        }
        return next_seq_id_++;
    }

    std::uint64_t LlamaCppExecutor::PrefixKey(const std::vector<int32_t>& tokens, int len)
    {
        // FNV-1a：对前 len 个 token 做哈希，再把 len 混进去，避免"同前缀不同长度"撞车
        std::uint64_t h = 1469598103934665603ull;
        for (int i = 0; i < len; ++i)
        {
            h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(tokens[static_cast<std::size_t>(i)]));
            h *= 1099511628211ull;
        }
        h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(len));
        h *= 1099511628211ull;
        return h;
    }

    LlamaCppExecutor::PinnedSeq* LlamaCppExecutor::FindPinned(int seq_id)
    {
        for (auto& item : pinned_)
        {
            if (item.seq_id == seq_id) return &item;
        }
        return nullptr;
    }

    const LlamaCppExecutor::PinnedSeq* LlamaCppExecutor::FindPinned(int seq_id) const
    {
        for (const auto& item : pinned_)
        {
            if (item.seq_id == seq_id) return &item;
        }
        return nullptr;
    }

    void LlamaCppExecutor::RegisterPrefixes(int seq_id, const std::vector<int32_t>& tokens)
    {
        const int n = static_cast<int>(tokens.size());
        if (chunk_tokens_ <= 0 || n < chunk_tokens_) return;

        PinnedSeq* entry = FindPinned(seq_id);
        if (entry == nullptr)
        {
            PinnedSeq fresh;
            fresh.seq_id = seq_id;
            pinned_.push_back(std::move(fresh));
            entry = &pinned_.back();
        }

        entry->n_tokens = std::max(entry->n_tokens, n);
        entry->stamp = ++touch_clock_;

        for (int len = chunk_tokens_; len <= n; len += chunk_tokens_)
        {
            const std::uint64_t key = PrefixKey(tokens, len);
            if (prefix_index_.find(key) != prefix_index_.end()) continue;
            prefix_index_.emplace(key, std::make_pair(seq_id, len));
            entry->lengths.push_back(len);
        }

        EvictPinnedIfNeeded();
    }

    void LlamaCppExecutor::EvictPinnedIfNeeded()
    {
        while (pinned_.size() > max_pinned_)
        {
            // 只淘汰"没有活跃请求在用"的最久未命中项；全都在跑就先不淘汰
            std::size_t victim = pinned_.size();
            std::uint64_t oldest = ~0ull;
            for (std::size_t i = 0; i < pinned_.size(); ++i)
            {
                if (seq_owner_.find(pinned_[i].seq_id) != seq_owner_.end()) continue;
                if (pinned_[i].stamp < oldest)
                {
                    oldest = pinned_[i].stamp;
                    victim = i;
                }
            }
            if (victim == pinned_.size()) return;

            const int seq_id = pinned_[victim].seq_id;
            for (auto it = prefix_index_.begin(); it != prefix_index_.end(); )
            {
                if (it->second.first == seq_id) it = prefix_index_.erase(it);
                else ++it;
            }

            llama_memory_seq_rm(llama_get_memory(ctx_), static_cast<llama_seq_id>(seq_id), -1, -1);
            free_seq_ids_.push_back(seq_id);
            pinned_.erase(pinned_.begin() + static_cast<std::ptrdiff_t>(victim));
            ++evicted_pinned_;
        }
    }

    std::vector<int> LlamaCppExecutor::Tokenize(const std::string& text) const 
    {
        const llama_vocab* vocab = llama_model_get_vocab(model_);
        int n_max = static_cast<int>(text.size()) + 16;
        std::vector<llama_token> tokens(static_cast<std::size_t>(n_max));
        int n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(), n_max, true, false);
        if (n < 0) 
        {
            tokens.resize(static_cast<std::size_t>(-n));
            n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(), -n, true, false);
        }
        if (n <= 0) 
        {
            return {0};
        }
        tokens.resize(static_cast<std::size_t>(n));
        std::vector<int> out(tokens.begin(), tokens.end());
        return out;
    }

    double LlamaCppExecutor::Prefill(const Request& req, const BlockTableView& view) 
    {
        (void)view;

        // 防御 同一 id 重复进入 Prefill 时先回收旧 seq, 否则会泄漏 KV
        auto existing = seq_map_.find(req.id);
        if (existing != seq_map_.end())
        {
            const int old_seq = existing->second;
            seq_owner_.erase(old_seq);
            // 旧 seq 如果被前缀缓存钉住，不能回收它的 KV，只断开请求映射
            if (FindPinned(old_seq) == nullptr)
            {
                llama_memory_seq_rm(llama_get_memory(ctx_), static_cast<llama_seq_id>(old_seq), -1, -1);
                free_seq_ids_.push_back(old_seq);
            }
            seq_map_.erase(existing);
        }

        int seq_id = AcquireSeqId();
        if (seq_id < 0) 
        {
            std::fprintf(stderr, "[LlamaCppExecutor] Prefill: seq capacity exhausted (n_seq_max=%d)\n", n_seq_max_);
            return -1.0;   // 用负值表示失败
        }
        seq_map_[req.id] = seq_id;

        // 用真实 tokenizer 或 prompt_text
        std::string text = req.prompt_text;
        if (text.empty()) 
        {
            text = std::string(static_cast<std::size_t>(req.PromptLen()) * 2, 'a');
        }
        std::vector<int32_t> tokens = TokenizeText(text);
        if (tokens.empty()) tokens.push_back(0);

        // ── 跨请求 KV 复用：命中前缀就先把 KV 复制过来，只 Prefill 后缀 ──
        seq_owner_[seq_id] = req.id;
        int reuse_len = 0;
        if (chunk_tokens_ > 0 && req.cached_prefix_len > 0)
        {
            // 至少留 1 个 token 给真实 decode，否则这一次拿不到 logits。
            // reuse_margin_ 是"不重用的尾部 token 数"，用来规避 seq_cp 在边界
            // cell 归属上的冲突（实验见 README 实验四）。
            int cached = req.cached_prefix_len - reuse_margin_;
            if (cached < 0) cached = 0;
            cached -= cached % (chunk_tokens_ > 0 ? chunk_tokens_ : 1);   // 对齐到 chunk
            const int want = std::min(cached, static_cast<int>(tokens.size()) - 1);
            if (want > 0)
            {
                auto hit = prefix_index_.find(PrefixKey(tokens, want));
                if (hit != prefix_index_.end() && hit->second.second == want)
                {
                    llama_memory_seq_cp(llama_get_memory(ctx_),
                                        static_cast<llama_seq_id>(hit->second.first),
                                        static_cast<llama_seq_id>(seq_id), 0, want);
                    reuse_len = want;
                    reused_prefix_tokens_ += want;
                    ++reuse_hits_;
                    if (PinnedSeq* src = FindPinned(hit->second.first))
                    {
                        src->stamp = ++touch_clock_;
                    }
                }
            }
        }

        const int32_t n_new = static_cast<int32_t>(tokens.size()) - reuse_len;

        // ── 分块 prefill（chunked prefill）──
        // llama_context 的 n_batch 上限是 512（见构造函数），一次塞超过 n_batch 个 token
        // 会 GGML_ASSERT abort。这里按 512 分块逐块 decode，只有最后一块的最后一个
        // token 需要 logits（用于采样首 token）。
        static constexpr int32_t kPrefillChunkTokens = 512;
        int rc = 0;
        double elapsed_s = 0.0;
        int32_t last_logits_index = 0;   // 最后一块里 logits 的 batch 内下标
        bool alloc_failed = false;

        for (int32_t off = 0; off < n_new && rc == 0; off += kPrefillChunkTokens)
        {
            const int32_t m = std::min<int32_t>(kPrefillChunkTokens, n_new - off);

            // 关键修复：用 llama_batch_init 而不是 llama_batch_get_one
            // llama_batch_get_one 固定 seq_id=0，会覆盖其他请求的 KV
            llama_batch batch = llama_batch_init(m, 0, 1);
            if (batch.token == nullptr)
            {
                alloc_failed = true;
                rc = -2;
                break;
            }

            for (int32_t i = 0; i < m; ++i)
            {
                const int32_t pos = reuse_len + off + i;
                batch.token[i] = tokens[static_cast<std::size_t>(pos)];
                batch.pos[i] = static_cast<llama_pos>(pos);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = static_cast<llama_seq_id>(seq_id);
                batch.logits[i] = (off + i == n_new - 1) ? 1 : 0;
            }
            batch.n_tokens = m;

            const auto t0 = std::chrono::steady_clock::now();
            rc = llama_decode(ctx_, batch);
            const auto t1 = std::chrono::steady_clock::now();
            elapsed_s += std::chrono::duration<double>(t1 - t0).count();

            last_logits_index = m - 1;
            llama_batch_free(batch);
        }

        if (alloc_failed)
        {
            // 分配失败必须返回负值，否则调用方会当成“瞬间完成”，completed 计数不可信
            seq_owner_.erase(seq_id);
            free_seq_ids_.push_back(seq_id);
            seq_map_.erase(req.id);
            return -1.0;
        }

        // ── 复用一致性诊断 ──
        // 在独立 context 里对同一段 prompt 做完整 prefill，比较最后位置的 logits。
        // 必须在任何后续 decode 之前把复用路径的 logits 拷出来（会被覆盖）。
        if (rc == 0 && verify_reuse_ && reuse_len > 0 && n_new > 0)
        {
            const int n_all = static_cast<int>(tokens.size());
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_));

            std::vector<float> reuse_logits(static_cast<std::size_t>(n_vocab));
            const float* got = llama_get_logits_ith(ctx_, last_logits_index);
            if (got != nullptr)
            {
                std::copy(got, got + n_vocab, reuse_logits.begin());

                if (verify_ctx_ == nullptr)
                {
                    llama_context_params vp = llama_context_default_params();
                    vp.n_ctx = n_ctx_;
                    vp.n_batch = 512;
                    vp.n_seq_max = 1;
                    vp.kv_unified = true;
                    verify_ctx_ = llama_init_from_model(model_, vp);
                }

                if (verify_ctx_ != nullptr)
                {
                    llama_batch rb = llama_batch_init(n_all, 0, 1);
                    for (int i = 0; i < n_all; ++i)
                    {
                        rb.token[i] = tokens[static_cast<std::size_t>(i)];
                        rb.pos[i] = i;
                        rb.n_seq_id[i] = 1;
                        rb.seq_id[i][0] = 0;
                        rb.logits[i] = (i == n_all - 1) ? 1 : 0;
                    }
                    rb.n_tokens = n_all;

                    if (llama_decode(verify_ctx_, rb) == 0)
                    {
                        const float* ref = llama_get_logits_ith(verify_ctx_, n_all - 1);
                        if (ref != nullptr)
                        {
                            double maxdiff = 0.0;
                            int arg_reuse = 0;
                            int arg_ref = 0;
                            for (int v = 0; v < n_vocab; ++v)
                            {
                                const double d = std::fabs(static_cast<double>(reuse_logits[static_cast<std::size_t>(v)])
                                                           - static_cast<double>(ref[v]));
                                if (d > maxdiff) maxdiff = d;
                                if (reuse_logits[static_cast<std::size_t>(v)] > reuse_logits[static_cast<std::size_t>(arg_reuse)]) arg_reuse = v;
                                if (ref[v] > ref[arg_ref]) arg_ref = v;
                            }
                            last_verify_max_diff_ = maxdiff;
                            std::fprintf(stderr,
                                "[verify] id=%llu reuse_len=%d maxdiff=%.6g argmax_reuse=%d argmax_ref=%d %s\n",
                                static_cast<unsigned long long>(req.id), reuse_len, maxdiff,
                                arg_reuse, arg_ref, (arg_reuse == arg_ref) ? "SAME_ARG" : "DIFF_ARG");
                        }
                    }
                    llama_batch_free(rb);
                    llama_memory_seq_rm(llama_get_memory(verify_ctx_), 0, -1, -1);
                }
            }
        }

        if (rc != 0) 
        {
            std::fprintf(stderr, "[LlamaCppExecutor] Prefill failed id=%llu rc=%d\n"
                    , static_cast<unsigned long long>(req.id), rc);
            // 清理 seq：归还 id 池，并把这个请求的映射一并去掉，否则 Release 会二次归还
            seq_owner_.erase(seq_id);
            free_seq_ids_.push_back(seq_id);
            seq_map_.erase(req.id);
            n_past_[req.id] = 0;
            return -1.0;
        }

        // 从最后一个位置采样首 token
        llama_token first = llama_sampler_sample(sampler_, ctx_, last_logits_index);
        pending_tokens_[req.id] = first;
        generated_ids_[req.id].push_back(static_cast<int32_t>(first));

        const llama_vocab* vocab = llama_model_get_vocab(model_);
        if (llama_vocab_is_eog(vocab, first)) 
        {
            eos_set_.insert(req.id);
        }

        n_past_[req.id] = static_cast<int>(tokens.size());

        // 把本次 prompt 的 block 对齐前缀登记为可复用源，供后续请求 seq_cp
        RegisterPrefixes(seq_id, tokens);

        return elapsed_s;
    }

    double LlamaCppExecutor::Decode(const Request& req, const BlockTableView& view) 
    {
        auto costs = DecodeBatch({&req}, {view});
        return costs.empty() ? 0.0 : costs[0];
    }

    std::vector<double> LlamaCppExecutor::DecodeBatch(const std::vector<const Request*>& reqs,
            const std::vector<BlockTableView>& views) 
    {
        (void)views;
        std::vector<double> costs(reqs.size(), 0.0);
        if (reqs.empty()) return costs;

        const std::size_t n = reqs.size();

        // 1. 检查每个请求都有 pending token 和 seq_id
        std::vector<llama_token> input_tokens(n);
        for (std::size_t i = 0; i < n; ++i) 
        {
            auto pt = pending_tokens_.find(reqs[i]->id);
            if (pt == pending_tokens_.end()) return costs;
            input_tokens[i] = pt->second;
        }

        // 2. 构造 batch
        llama_batch batch = llama_batch_init(static_cast<int32_t>(n), 0, 1);
        if (batch.token == nullptr) return costs;

        for (std::size_t i = 0; i < n; ++i) 
        {
            auto sit = seq_map_.find(reqs[i]->id);
            if (sit == seq_map_.end()) 
            {
                llama_batch_free(batch);
                // 这个请求没有 seq，标记所有为失败
                for (std::size_t k = 0; k < n; ++k) costs[k] = -1.0;
                return costs;
            }
            int seq_id = sit->second;
            batch.token[i] = input_tokens[i];
            batch.pos[i] = n_past_[reqs[i]->id];
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = static_cast<llama_seq_id>(seq_id);
            batch.logits[i] = 1;  // 每个位置都需要输出 logits
        }
        batch.n_tokens = static_cast<int32_t>(n);

        // 3. 一次 decode
        auto t0 = std::chrono::steady_clock::now();
        int rc = llama_decode(ctx_, batch);
        auto t1 = std::chrono::steady_clock::now();
        double total = std::chrono::duration<double>(t1 - t0).count();

        if (rc != 0) 
        {
            llama_batch_free(batch);
            for (std::size_t k = 0; k < n; ++k) costs[k] = -1.0;
            return costs;
        }

        // 4. 每个位置采样下一个 token, 更新 n_past
        for (std::size_t i = 0; i < n; ++i) 
        {
            llama_token next = llama_sampler_sample(sampler_, ctx_, static_cast<int32_t>(i));
            pending_tokens_[reqs[i]->id] = next;
            generated_ids_[reqs[i]->id].push_back(static_cast<int32_t>(next));

            const llama_vocab* vocab = llama_model_get_vocab(model_);
            if (llama_vocab_is_eog(vocab, next)) 
            {
            eos_set_.insert(reqs[i]->id);
            }

            n_past_[reqs[i]->id]++;
            costs[i] = total;  // 每个请求都等整个 batch
        }

        llama_batch_free(batch);
        return costs;
    }

    void LlamaCppExecutor::Release(RequestId id) 
    {
        auto it = seq_map_.find(id);
        if (it == seq_map_.end()) return;
        int seq_id = it->second;

        seq_map_.erase(it);
        n_past_.erase(id);
        pending_tokens_.erase(id);
        eos_set_.erase(id);
        seq_owner_.erase(seq_id);

        // 这个 seq 的前缀已经登记为可复用源：保留它的 KV，
        // 让后面的请求能用 llama_memory_seq_cp 直接复制，真正的回收交给淘汰逻辑
        if (FindPinned(seq_id) != nullptr) return;

        // 没被登记的，清掉 KV 并把 seq_id 归还池
        llama_memory_seq_rm(llama_get_memory(ctx_), static_cast<llama_seq_id>(seq_id), -1, -1);
        free_seq_ids_.push_back(seq_id);
    }

    std::vector<int32_t> LlamaCppExecutor::TokenizeText(const std::string& text) const 
    {
        const llama_vocab* vocab = llama_model_get_vocab(model_);
        int n_max = static_cast<int>(text.size()) + 16;
        std::vector<llama_token> tokens(static_cast<std::size_t>(n_max));
        int n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(), n_max, true, false);
        if (n < 0) 
        {
            tokens.resize(static_cast<std::size_t>(-n));
            n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(), -n, true, false);
        }
        if (n <= 0) return {0};
        tokens.resize(static_cast<std::size_t>(n));
        std::vector<int32_t> out(tokens.begin(), tokens.end());
        return out;
    }

    bool LlamaCppExecutor::IsEos(RequestId id) const 
    {
        return eos_set_.count(id) > 0;
    }
}  