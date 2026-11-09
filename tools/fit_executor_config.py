#!/usr/bin/env python3
"""把 agentinferkv_calibrate 的 CSV 拟合成 SimulatedExecutor 的参数。

用法：
    ./build_real/agentinferkv_calibrate model.gguf 9 > calib.csv
    python3 tools/fit_executor_config.py calib.csv

输出：base_prefill_ms、per_token_prefill_us、decode_ms_per_token + R²。
把结果填进 include/SimulatedExecutor.h 的 ExecutorConfig。
"""

import sys


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)  # header: L,tokens,prefill_ms,decode_step_ms
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 4:
                continue
            rows.append((int(parts[1]), float(parts[2]), float(parts[3])))
    return rows


def ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return a, b, r2


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    rows = load(sys.argv[1])
    if not rows:
        print("no data")
        return 1

    xs = [r[0] for r in rows]
    ys_prefill = [r[1] for r in rows]
    ys_decode = [r[2] for r in rows]

    a, b, r2 = ols(xs, ys_prefill)
    decode = sum(ys_decode) / len(ys_decode)

    print(f"points: {len(rows)}  tokens: {xs[0]}..{xs[-1]}")
    print(f"prefill  = {a:.4f} ms + {b * 1000:.4f} us/token   (R^2 = {r2:.3f})")
    print(f"decode   = {decode:.4f} ms/token                  (std across lengths)")
    print()
    print("填入 include/SimulatedExecutor.h：")
    print(f"    base_prefill_s        = {a / 1000:.6f};   // {a:.2f} ms")
    print(f"    per_token_prefill_s   = {b / 1000:.9f};   // {b * 1000:.2f} us/token")
    print(f"    base_decode_s         = {decode / 1000:.6f};   // {decode:.2f} ms")
    print(f"    per_token_decode_s    = 0.0;")
    return 0


if __name__ == "__main__":
    sys.exit(main())
