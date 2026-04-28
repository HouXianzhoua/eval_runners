# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This repo contains runner scripts for EvalScope performance benchmarking. It launches mixed-workload benchmarks (multimodal/VL + text-only) against OpenAI-compatible API endpoints.

## Project layout

- `run_evalscope_mixed_perf.py` — One-shot concurrent perf test using `evalscope perf` subprocesses. Runs a fixed number of requests for each workload, then generates a combined report.
- `run_evalscope_mixed_stability_perf.py` — Long-duration stability test using EvalScope's internal Python APIs in-process (async). Runs continuously for the target duration, analyzes time-windowed metrics, and generates stability reports with fluctuation percentages.
- `tokenizer_cache/` — Cached HuggingFace tokenizer files (Qwen3-VL), not part of the runner logic.
- No `pyproject.toml`, `setup.cfg`, or `requirements.txt` — this repo has no installable package. The `.venv/` is already configured and the scripts import `evalscope` from a sibling repo.

## Dependencies (evalscope)

Both scripts import from `evalscope`, which is expected to live in a sibling directory. The scripts auto-discover the evalscope repo by checking:

1. `SCRIPT_DIR / evalscope` (a subdirectory of this repo)
2. `SCRIPT_DIR.parent / evalscope` (a sibling directory)
3. A `.venv` inside the discovered evalscope repo

The stability script (`run_evalscope_mixed_stability_perf.py`) uses EvalScope internals directly: `evalscope.perf.arguments.Arguments`, `evalscope.perf.http_client.AioHttpClient`, `evalscope.perf.plugin.ApiRegistry`/`DatasetRegistry`, and `evalscope.perf.utils.db_util`. It inserts the evalscope venv's `site-packages` and repo root into `sys.path` at runtime.

The perf script (`run_evalscope_mixed_perf.py`) spawns `evalscope perf` as subprocesses and does not import the evalscope package.

## Running the scripts

Both scripts require `--model`, `--url`, and `--tokenizer-path`. They activate the project's `.venv/` (or the evalscope venv) implicitly — just use the python from `.venv/Scripts/python.exe`.

**One-shot mixed perf:**

```bash
.venv/Scripts/python run_evalscope_mixed_perf.py \
  --model "Qwen/Qwen3-VL-30B-A3B-Instruct" \
  --url "http://localhost:8000/v1" \
  --tokenizer-path "Qwen/Qwen3-VL-30B-A3B-Instruct" \
  --parallel 16
```

`--parallel` must be a multiple of 8 (split 3:5 VL:text). Use `--enable-warmup`/`--no-enable-warmup` to control warmup.

**Stability test (long-duration):**

```bash
.venv/Scripts/python run_evalscope_mixed_stability_perf.py \
  --model "Qwen/Qwen3-VL-30B-A3B-Instruct" \
  --url "http://localhost:8000/v1" \
  --tokenizer-path "Qwen/Qwen3-VL-30B-A3B-Instruct" \
  --duration-minutes 120 \
  --window-minutes 30 \
  --parallel 8 \
  --rate 1.0
```

Key stability options: `--duration-minutes` (default 720), `--window-minutes` (default 60), `--rate` (req/s, use -1 for closed-loop max throughput), `--warmup-requests` (default 30, set 0 to skip).

## Concurrency split convention

Both scripts enforce a 3:5 ratio for VL:text concurrency. `--parallel` must be a multiple of 8. VL gets `parallel * 3 // 8`, text gets `parallel * 5 // 8`. Per-lane overrides (`--vl-parallel`, `--text-parallel`) take effect when `--parallel` is not set.

## No tests

This repo has no test suite. All validation is done manually by running the scripts against a live endpoint and inspecting the generated reports (markdown + TSV + JSON in the output directory).

## Output directories (gitignored)

- `mixed_perf_outputs/` — default output for the one-shot perf script
- `mixed_stability_outputs/` — default output for the stability script
- Any `*_output/` or `*_outputs/` directory
