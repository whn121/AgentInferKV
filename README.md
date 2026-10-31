# AgentInferKV

C++17 实现的 LLM 推理调度与 KV Cache 管理控制平面原型。参照 vLLM V1 的 Scheduler /
BlockSpaceManager 分层和 SGLang 的前缀复用思路，实现 Continuous Batching 调度、
Paged KV Cache 块管理、按 Block 对齐的前缀缓存。

执行层通过 `IInferenceExecutor` 接口与控制层解耦：模拟执行器用于控制层逻辑验证，llama.cpp
CUDA 后端用于真实 GPU 执行验证。控制层代码在两种后端下完全复用。

**项目定位**：控制平面原型，不是完整推理引擎。没有 CUDA kernel、没有真实 KV 张量存储、
没有跨请求 GPU KV 共享。

## 当前能力

| 模块 | 内容 |
|---|---|
| **Scheduler** | FCFS（对标 vLLM V1 默认调度），waiting/running 队列、准入控制、BlockBudget 模拟扣减 |
| **BlockManager** | Naive（按 max_len 预分配，基线）和 Paged（按 prompt 分配，Decode 按需扩块） |
| **PrefixCache** | 按 Block 对齐的前缀 Trie，前缀匹配 + 引用计数共享，只共享完整块 |
| **SimulationLoop** | 事件驱动主循环，按模拟时钟推进，处理到达 / Prefill 完成 / Decode 完成 |
| **Executor** | Simulated（公式耗时）和 LlamaCpp（真实 GPU 适配，支持批量 Decode） |

## 不做什么

- **不是完整推理引擎**：没有 CUDA kernel、attention 计算、真实 KV 张量存储。
- **控制平面与执行平面是半解耦的**：`PagedBlockManager` 的 `BlockId` 和 llama.cpp 内部
  KV cache 是两套系统，不是完全控制。
- **不支持跨请求 GPU KV 共享**：llama.cpp 的 seq 级 KV cache 需要 `llama_memory_seq_cp`
  才能跨请求复制，涉及源 seq 存活、块对齐、引用计数，第一版不做。**PrefixCache 只在
  模拟层生效，真实执行层未实现前缀复用。**
- **不支持抢占（preemption）/ KV swap**：块不足时用背压和取消替代。
- **不支持多模型 / 多 tokenizer**：PrefixCache 只有一棵树，没有 `model_id` 维度。
- **不做 LRU 淘汰**：PrefixCache 节点 `ref_count` 到 0 也不删，缓存持有块引用直到进程结束。
- **Prefill 不批量，不支持 Chunked Prefill**：Decode 支持 batch 合并，Prefill 仍逐请求
  执行；长 prompt 超过 `n_batch=512` 时会失败，第一版未做拆分。
- **真实执行层没有 EOS 判断和 token 解码**：`LlamaCppExecutor` 完成 Prefill/Decode 和
  采样，但没有判断生成是否遇到 EOS、没有把 token 解码成文本、没有保存生成结果。
- **未做复杂故障注入**：测试覆盖核心路径，未做故障恢复、并发压力、长时间稳定性验证。

## 构建与测试

### 纯模拟版（不依赖 llama.cpp）

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j
    ctest --test-dir build --output-on-failure
    ./build/agentinferkv_demo

### 真实化版（链接 llama.cpp CUDA 后端，可选）

需要先编译 llama.cpp 的 CUDA 版本（`-DGGML_CUDA=ON`），默认路径 `~/src/llama.cpp`，
可用 `-DLLAMA_ROOT=/your/path` 覆盖：

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_LLAMA_CPP=ON
    cmake --build build -j 4
    ./build/agentinferkv_demo_real ~/models/qwen3-5-4B-Q4_K_M.gguf

## 实验结果

### 实验 1：Paged vs Naive 并发上限（模拟 trace）

同一 8 块预算、6 个请求、prompt=20 + max_new=30、block_size=16：

| 策略 | max_concurrent | 说明 |
|---|---|---|
| Naive | 2 | 每请求按 50 token 分 4 块，8 块只能容纳 2 个 |
| Paged | 4 | 每请求先按 20 token 分 2 块，早期容纳 4 个 |

**Paged 在相同 block 预算下并发请求上限提升 2 倍。** 对比的是"同样物理块预算下能容纳
多少逻辑序列"——块和显存字节一一对应（LLaMA-2-7B 单 token 512KB，一个 block 16 token
= 8MB），所以这个结论在真实 GPU 上也成立。

**注意**：初始准入并发提高不等于持续完成吞吐量提高。Paged 让更多请求早期进入运行，但
后续 Decode 扩块时如果资源不足，仍会触发取消。**本实验衡量的是"准入上限"，不是"完成
吞吐"。**

### 实验 2：PrefixCache 对 Prefill 工作量的影响（模拟）

Agent 多轮场景：System Prompt 32 token + User Query 16 token，5 个请求，相同 System Prompt：

| 策略 | avg_ttft | total_prefill_tokens | cache hits |
|---|---|---|---|
| OFF | 0.0058 s | 240 | 0/0 |
| ON | 0.0032 s | 112 | 4/5 |

**TTFT 降低约 45%，Prefill token 减少 53.3%。** 模拟工作量模型下的结果，不是真实 GPU
性能。模拟耗时由 `ExecutorConfig` 的线性公式计算，TTFT 的相对差异来自前缀复用减少了
Prefill 的 token 数。

### 实验 3：真实 GPU 适配（架构验证，非性能优化）

执行层从 `SimulatedExecutor` 替换为 `LlamaCppExecutor`，用 llama.cpp CUDA 后端驱动真实
GPU 执行，控制层代码零改动。**目的是验证"控制层能驱动真实 GPU 执行"这个架构假设。**

**环境**：RTX 4060 Laptop 8GB + WSL2 + CUDA 12.x，Qwen3.5-4B Q4_K_M (2.5GB)。
**负载**：5 个 Agent 多轮请求，相同 System Prompt，50ms 到达间隔。

**结果**（5 次运行，每次 5 个请求全部完成）：

| 运行 | avg TTFT | max_concurrent |
|---|---|---|
| run1（冷启动） | 35.7 ms | 3 |
| run2 | 24.2 ms | 2 |
| run3 | 24.0 ms | 3 |
| run4 | 25.6 ms | 2 |
| run5 | 24.0 ms | 2 |

**稳定值约 24–26ms，第一次运行约 36ms（GPU 冷启动、CUDA context 初始化）。** 这是真实的
Prefill + 首 token 采样耗时，不包含任何 PrefixCache 优化。

**数据波动说明**：WSL2 环境下有 Windows 显示驱动预留显存（约 800MiB）和系统后台负载，
真实 GPU 数据天然有波动。第一次运行明显偏高是冷启动效应。

**适配过程中修复的关键问题**：

1. **seq_id 一致性**：`llama_batch_get_one` 固定使用 sequence ID 0，多请求场景下会导致
   不同请求的 Prefill KV 互相覆盖。改用 `llama_batch_init` 手动为每个请求分配独立
   `seq_id` 和 `pos`。
2. **真实 tokenizer 对齐**：`main_real.cpp` 用 `LlamaCppExecutor::TokenizeText` 填
   `prompt_tokens`，保证 PrefixCache 匹配和真实 Prefill 用同一套 token。
3. **执行失败传播**：`Prefill` 和 `DecodeBatch` 失败时返回负耗时，`SimulationLoop` 检查
   到负值就取消请求、释放资源。**修复前失败返回 0，会被误认为"瞬间完成"，`completed`
   计数不可信。**
4. **seq 容量限制**：`AcquireSeqId` 在 `next_seq_id_ >= n_seq_max_` 时返回 -1，避免
   seq 越界。

**范围限制**：

- 真实执行层不支持跨请求 KV 共享（PrefixCache 只在模拟层生效）；
- 真实 GPU 上的 PrefixCache TTFT 降低未验证；
- Prefill 仍逐请求执行，未批量，不支持 Chunked Prefill；
- 没有 EOS 判断和 token 解码，`LlamaCppExecutor` 是执行适配实验，不是可直接提供服务的
  推理引擎；
- 模拟时钟不等于真实在线 Serving 的端到端 TTFT——请求按模拟时间进入 `SimulationLoop`，
  真实 llama.cpp 耗时作为事件推进时间。

## 设计边界（面试重点）

### 1. PagedBlockManager 是控制平面，不是数据平面

`PagedBlockManager` 实现 PagedAttention 的**块管理控制逻辑**：`BlockId` 的分配、逻辑→物理
映射、引用计数、按需扩块。对应 vLLM 的 `BlockSpaceManager`。**不是**真实 KV 张量存储。

真实化后，`BlockId` 依然不映射到 llama.cpp 内部的物理显存块——llama.cpp 有自己的 KV
cache 管理，`PagedBlockManager` 的块账目和它是两套独立系统。**这是半解耦，不是完全控制。**

### 2. SimulatedExecutor 的指标是模拟值

模拟实验的 TTFT、吞吐是公式计算值，**不是真实 GPU 性能**。真实 GPU 的数值必须用
`LlamaCppExecutor` 跑，且要明确标注环境。

### 3. PrefixCache 只共享完整块，不做 CoW

共享单位是 block，不是 token。最后一个不完整块不共享，请求在共享前缀后的 Decode 只写
自己新分配的块。这样共享块永远只读，不需要 Copy-on-Write。

### 4. PrefixCache 持有块引用，没有 LRU 淘汰

请求结束只减请求的引用，缓存的引用保留到进程结束。**用空间换复用。** 真实引擎需要 LRU
淘汰策略，第一版没做，是范围控制。

### 5. Scheduler 的 EstimateBlocks 是保守估算

`EstimateBlocks` 不知道前缀命中，按完整 prompt 估算。**可能过早拒绝可命中的请求，但不
会错误准入。**

### 6. Decode 批量，Prefill 不批量

`DecodeBatch` 把多个请求的 token 合并成一次 `llama_decode`；Prefill 仍逐请求执行。**原因：**
Prefill 序列长度差异大（几十到几千 token），合并会让一个短请求等长请求完成；Decode 每步
只 1 个 token，合并的并行收益远大于延迟代价。

### 7. 执行层是适配，不是深度定制

`LlamaCppExecutor` 是 llama.cpp C API 的适配层：把控制层的 `Request` 映射到 llama.cpp 的
`seq_id`、`llama_batch`、`n_past`。**llama.cpp 内部的 KV cache 管理、batch 调度语义以
官方文档为准。** 本项目的价值在控制层——Scheduler 的分层、BlockManager 的引用计数、
PrefixCache 的前缀匹配——不在优化 llama.cpp。

## 目录

```
include/    头文件
  Interfaces.h          IBlockSpaceManager / IScheduler / IInferenceExecutor / Request
  ModelConfig.h         KV Cache 显存公式
  FCFSScheduler.h       对标 vLLM V1 FCFS
  NaivePreallocator.h   按 max_len 预分配（基线）
  PagedBlockManager.h   按 prompt 分配，按需扩块
  PrefixCache.h         按 Block 对齐的前缀 Trie
  SimulatedExecutor.h   公式耗时模拟
  LlamaCppExecutor.h    llama.cpp CUDA 后端适配（执行层）
  SimulationLoop.h      事件驱动主循环
src/        实现
tests/      CTest 单元测试（8 个可执行文件）
```

## 测试清单

以下是控制层已有的单元测试（CTest 注册，全部通过）。**测试覆盖核心路径，未做复杂故障注入。**

| 测试 | 内容 |
|---|---|
| test_request | Request 状态机、TTFT/Latency 计算、非法转换拒绝 |
| test_model_config | KV Cache 显存公式、块数计算边界 |
| test_naive_preallocator | 分配、释放、重复释放、引用计数、扩块 no-op |
| test_fcfs_scheduler | 准入顺序、预算扣减、tie-break |
| test_simulated_executor | Prefill 线性、Decode 固定、参数为 0 的边界 |
| test_paged_block_manager | 按 prompt 分配、按需扩块、扩块失败、共享引用 |
| test_prefix_cache | 前缀匹配、插入、释放、部分块不匹配、统计计数 |
| test_simulation_loop | 端到端闭环、资源竞争、PrefixCache 降低 TTFT（模拟） |

## 已知短板

- **WSL2 下有显存开销**：Windows 显示驱动预留约 800MiB，llama.cpp 实际可用约 7.3GB。
- **真实执行层不支持跨请求 KV 共享**：PrefixCache 在真实 GPU 上不生效，需要
  `llama_memory_seq_cp` 才能实现，第一版未做。
- **真实执行层没有 EOS 判断和 token 解码**：`LlamaCppExecutor` 完成 Prefill/Decode 和
  采样，但没有生成完整文本输出。
- **真实执行层不支持 Chunked Prefill**：长 prompt 超过 `n_batch=512` 时会失败。
- **模拟时钟不等于真实在线 Serving 的 TTFT**：请求按模拟时间进入 `SimulationLoop`，
  真实 llama.cpp 耗时作为事件推进时间。这能研究准入和调度，但不是真实在线服务的端到端
  TTFT 测量。
- **PrefixCache 的 `prompt_text` 截断是字符近似**：命中前缀后按字符比例截断文本，和
  token 边界不对齐。真实实现应按 token 边界截断。
- **batch 合并的耗时统一用 total**：每个请求都等整个 batch 完成，单个请求的模拟延迟会
  略高，但总吞吐提升。
- **没有端到端集成**：AgentRuntime 网络层和 AgentInferKV 是接口级衔接，还没做端到端部署。
- **测试未做故障注入**：未覆盖并发竞争、进程崩溃、长时间运行稳定性。