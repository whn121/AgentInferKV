#!/usr/bin/env bash
# soak：重复跑模拟/映射实验，检查块账目是否漂移（free+cached == num_blocks）。
# 用法：bench/soak.sh [轮数] [build 目录]
set -u
ROUNDS=${1:-30}
BUILD_DIR=${2:-build}
DEMO="$BUILD_DIR/agentinferkv_demo"
CELL="$BUILD_DIR/agentinferkv_cell_demo"
fail=0

if [ ! -x "$DEMO" ]; then
  echo "missing $DEMO (先 cmake --build $BUILD_DIR)"
  exit 2
fi

for i in $(seq 1 "$ROUNDS"); do
  out=$("$DEMO" 2>/dev/null) || { echo "round $i: demo exit $?"; fail=1; }
  if echo "$out" | grep -q MISMATCH; then
    echo "round $i: MISMATCH"
    echo "$out" | grep -E "blocks:|metrics:"
    fail=1
  fi
  if [ -x "$CELL" ]; then
    cout=$("$CELL" 2>/dev/null) || { echo "round $i: cell_demo exit $?"; fail=1; }
    if echo "$cout" | grep -qi "mismatch\|leak"; then
      echo "round $i: cell_demo mismatch/leak"
      echo "$cout" | tail -5
      fail=1
    fi
  fi
done

if [ "$fail" -eq 0 ]; then
  echo "soak OK: $ROUNDS rounds, no MISMATCH"
else
  echo "soak FAILED"
fi
exit "$fail"
