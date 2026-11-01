#!/usr/bin/env bash
# 一键复现：编译 -> 单元测试 -> 实验1 -> 实验2 -> 实验3（可选，需要 GPU + llama.cpp）
set -e
cd "$(dirname "$0")"

MODEL="${HOME}/models/qwen3-5-4B-Q4_K_M.gguf"
BUILD_DIR="build"

echo "=== Build ==="
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    if [ -d "$HOME/src/llama.cpp" ]; then
        cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DUSE_LLAMA_CPP=ON
    else
        cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
    fi
else
    # 已有 build/ 时不再重新配置，但要提醒：改过 CMakeLists 或切换过 USE_LLAMA_CPP
    # 之后必须手动 rm -rf build 重配，否则会用到过期目标/过期二进制
    echo "复用已有 $BUILD_DIR（如需重新配置请先 rm -rf $BUILD_DIR）"
fi
cmake --build "$BUILD_DIR" -j 4

echo ""
echo "=== Unit tests ==="
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo ""
echo "=== Experiment 1: Naive vs Paged (simulated) ==="
"./$BUILD_DIR/agentinferkv_experiment1"

echo ""
echo "=== Experiment 2: PrefixCache + LRU (simulated) ==="
"./$BUILD_DIR/agentinferkv_demo"

echo ""
echo "=== Experiment 3: Real GPU (if available) ==="
# 必须同时满足三个条件才跑实验3，尤其是第三个：
# build/ 里可能残留上一次（未开 USE_LLAMA_CPP）编出来的旧 demo_real，
# 直接执行会跑出与 README 不一致的数字。
LLAMA_ON=no
if grep -q "USE_LLAMA_CPP:BOOL=ON" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
    LLAMA_ON=yes
fi

if [ "$LLAMA_ON" = "yes" ] && [ -x "$BUILD_DIR/agentinferkv_demo_real" ] && [ -f "$MODEL" ]; then
  for i in 1 2 3; do
    echo "--- run $i ---"
    # 程序自己的输出走 stdout，直接打印；llama.cpp 的加载日志走 stderr，
    # 只挑出执行器自己的诊断行（例如 KV reuse disabled 的原因），其余丢掉
    "./$BUILD_DIR/agentinferkv_demo_real" "$MODEL" 2>"$BUILD_DIR/exp3.stderr.$i"
    grep "\[LlamaCppExecutor\]" "$BUILD_DIR/exp3.stderr.$i" || true
  done
else
  echo "Skipped: 需要同时满足 ——"
  echo "  1) build 用 -DUSE_LLAMA_CPP=ON 配置过（当前: $LLAMA_ON）"
  echo "  2) $BUILD_DIR/agentinferkv_demo_real 存在"
  echo "  3) 模型存在: $MODEL"
  echo "  重新配置: rm -rf $BUILD_DIR && cmake -S . -B $BUILD_DIR -DCMAKE_BUILD_TYPE=Release -DUSE_LLAMA_CPP=ON"
fi
