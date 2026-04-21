# EvalScope Mixed Stability Perf Report

- Generated At: 2026-04-21T17:35:30
- Model: Qwen3-VL-30B-A3B-Instruct
- URL: http://120.48.75.178:4842/v1/chat/completions
- Target Duration: 0.0333h
- Output Root: /home/houxianzhou/vlm_eval_workspace/eval_runners/mixed_stability_smoke_outputs

## Combined

| 项目 | 初值/时长 | 末值/请求数 | 结果 |
| :--- | :--- | :--- | :--- |
| 长周期 | 0h01m43s | 540 | 100.00% |
| TTFT 衰减 | 0.11s | 0.12s | 4.84% |
| TPS 衰减 | 184.4314 | 184.6736 | -0.13% |

## Text Lane

| 项目 | 初值/时长 | 末值/请求数 | 结果 |
| :--- | :--- | :--- | :--- |
| 长周期 | 0h01m42s | 360 | 100.00% |
| TTFT 衰减 | 0.12s | 0.13s | 4.35% |
| TPS 衰减 | 125.4361 | 125.4361 | 0.00% |

## VL Lane

| 项目 | 初值/时长 | 末值/请求数 | 结果 |
| :--- | :--- | :--- | :--- |
| 长周期 | 0h01m43s | 180 | 100.00% |
| TTFT 衰减 | 0.08s | 0.09s | 6.31% |
| TPS 衰减 | 60.7080 | 60.9516 | -0.40% |

## 说明

- Combined 为两路请求明细合并后的整体稳定性结果。
- TPS 按输出 token throughput 计算，即窗口内成功请求 completion_tokens 总和 / 窗口时长。
- Text/VL 单路结果保留，方便定位是哪一路先退化。
