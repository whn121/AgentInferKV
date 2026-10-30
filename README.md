# AgentInferKV

C++17 实现的推理调度与 KV Cache 管理模拟系统。参照 vLLM V1 的 Scheduler / BlockSpaceManager
分层和 SGLang 的 RadixAttention，实现 Continuous Batching 调度、Paged KV Cache 块管理、
Agent 场景的前缀缓存复用。执行层用 `IInferenceExecutor` 接口隔离，当前为公式模拟，不提交
GPU 任务。

## 当前能力

| 模块 | 内容 |
|---|---|
| **Scheduler** | FCFS（对标 vLLM V1 默认调度），waiting/running 队列、准入控制、预算模拟扣减 |
| **BlockManager** | Naive（按 max_len 预分配，基线）和 Paged（按 prompt 分配，Decode 按需扩块） |
| **PrefixCache** | 按 block 对齐的 Radix Tree，前缀匹配 + 引用计数共享，只共享完整块 |
| **SimulationLoop** | 事件驱动主循环，按模拟时钟推进，处理到达 / Prefill 完成 / Decode 完成 |
| **Executor** | `SimulatedExecutor`：Prefill 耗时随 prompt 长度线性增长，Decode 每步固定 |

## 不做什么

- **不实现真实 GPU 推理**：没有 CUDA kernel、没有 attention 计算、没有真实 KV 张量。
- **不持久化 GPU KV Cache**：`BlockId` 没有对应真实的 `torch::Tensor`。
- **不支持抢占（preemption）/ KV swap**：块不足时用背压和取消替代。
- **不支持多模型 / 多 tokenizer**：PrefixCache 只有一棵树，没有 `model_id` 维度。
- **不做 LRU 淘汰**：PrefixCache 节点 `ref_count` 到 0 也不删，缓存持有块引用直到进程结束。

## 构建与测试

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
    cmake --build build -j
    ctest --test-dir build --output-on-failure
    ./build/agentinferkv_demo

## 实验结果（模拟工作量模型）

**Paged vs Naive（8 块预算，6 个请求，prompt=20 + max_new=30，block_size=16）：**

| 策略 | max_concurrent | 说明 |
|---|---|---|
| Naive | 2 | 每请求按 50 token 分 4 块，8 块只能容纳 2 个 |
| Paged | 4 | 每请求先按 20 token 分 2 块，早期容纳 4 个 |

**PrefixCache（Agent 多轮，System Prompt 32 token + User Query 16 token，5 个请求）：**

| 策略 | avg_ttft | total_prefill_tokens | cache hits |
|---|---|---|---|
| OFF | 0.0058 s | 240 | 0/0 |
| ON | 0.0032 s | 112 | 4/5 |

**所有性能数字为模拟工作量模型下的结果，不是真实 GPU 推理性能。** Prefill/Decode
耗时由 `ExecutorConfig` 的线性公式计算，TTFT 的“相对差异”来自调度顺序和前缀复用，
不来自 GPU 算子速度。

## 设计边界（面试重点）

- `PagedBlockManager` 实现的是 PagedAttention 的**块管理控制逻辑**（`BlockId` 分配、映射、
  引用计数、按需扩块），对应 vLLM 的 `BlockSpaceManager`。**不是**真实 KV 张量存储。
- PrefixCache 只共享“完整填满的 block”，不做 Copy-on-Write。最后一块不共享。
- PrefixCache 持有块引用，请求结束只减请求的引用，缓存引用保留到进程结束。**用空间换复用，
  没有 LRU 淘汰。**
- Scheduler 的 `EstimateBlocks` 不知道前缀命中，按完整 prompt 保守估算。**可能过早拒绝
  可命中的请求，但不会错误准入。**

## 目录

    include/    头文件：Interfaces、ModelConfig、Scheduler、BlockManager、PrefixCache、Loop、Executor
    src/        实现
    tests/      CTest 单元测试（9 个测试可执行文件）
    benchmark/  （暂无，benchmark 在 demo 中）