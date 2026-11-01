#!/usr/bin/env bash
# 一键复现：编译 -> 单元测试 -> 实验1 -> 实验2 -> 实验3（可选，需要 GPU）
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
if [ -f "$BUILD_DIR/agentinferkv_demo_real" ] && [ -f "$MODEL" ]; then
  for i in 1 2 3; do
    echo "--- run $i ---"
    "./$BUILD_DIR/agentinferkv_demo_real" "$MODEL" 2>/dev/null
  done
else
  echo "Skipped: agentinferkv_demo_real 或模型不存在"
  echo "  需要: cmake -DUSE_LLAMA_CPP=ON 编译，且 $MODEL 存在"
fi
