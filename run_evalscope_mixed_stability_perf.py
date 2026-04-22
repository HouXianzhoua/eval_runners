#!/usr/bin/env python3
"""Run a long-duration mixed EvalScope perf benchmark and generate stability reports.

This script preserves the request types and subprocess launch pattern from
`run_evalscope_mixed_perf.py`:

- one multimodal lane using `random_vl`
- one text-only lane using `random`

The difference is that the benchmark runs by wall-clock duration instead of
fixed request totals. Each lane is executed in rolling EvalScope perf batches
until the target duration is reached. At the end, all batch DBs are aggregated
to compute stability metrics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shlex
import shutil
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR if (SCRIPT_DIR / 'evalscope').exists() else SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = WORKSPACE_DIR / 'mixed_stability_outputs'


@dataclass
class RequestRecord:
    start_time: float
    completed_time: float
    success: bool
    ttft: float | None
    tpot: float | None
    completion_tokens: int


@dataclass
class BatchSummary:
    batch_index: int
    duration_seconds: float
    total_requests: int
    success_requests: int
    failed_requests: int
    success_rate_percent: float
    avg_ttft_seconds: float
    avg_tpot_seconds: float
    avg_tps: float


def find_evalscope_repo_root() -> Path | None:
    candidates = [
        SCRIPT_DIR,
        SCRIPT_DIR / 'evalscope',
        SCRIPT_DIR.parent / 'evalscope',
    ]
    for candidate in candidates:
        if (candidate / 'pyproject.toml').exists() and (candidate / 'evalscope' / 'cli' / 'cli.py').exists():
            return candidate
    return None


def is_python_executable(path: Path) -> bool:
    return path.name.lower().startswith('python')


def resolve_evalscope_cmd(evalscope_bin: str) -> List[str]:
    repo_root = find_evalscope_repo_root()
    repo_venv_python = repo_root / '.venv' / 'bin' / 'python' if repo_root else None

    if os.path.sep in evalscope_bin or evalscope_bin.startswith('.'):
        resolved = Path(evalscope_bin).expanduser()
        if not resolved.is_absolute():
            resolved = resolved.resolve()
        if is_python_executable(resolved):
            return [str(resolved), '-m', 'evalscope.cli.cli']
        return [str(resolved)]

    if evalscope_bin == 'evalscope' and repo_venv_python and repo_venv_python.exists():
        return [str(repo_venv_python), '-m', 'evalscope.cli.cli']

    found = shutil.which(evalscope_bin)
    if found:
        found_path = Path(found)
        if is_python_executable(found_path):
            return [str(found_path), '-m', 'evalscope.cli.cli']
        return [str(found_path)]

    raise FileNotFoundError(
        f'Cannot find EvalScope executable: {evalscope_bin}. '
        'Pass --evalscope-bin explicitly or run this script from a workspace that contains the evalscope repo.'
    )


def sanitize_name(value: str) -> str:
    cleaned = []
    for ch in value:
        if ch.isalnum() or ch in ('-', '_', '.'):
            cleaned.append(ch)
        else:
            cleaned.append('_')
    normalized = ''.join(cleaned).strip('._')
    return normalized or 'model'


def build_default_output_root(model_name: str) -> Path:
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    return DEFAULT_OUTPUT_ROOT / f'{sanitize_name(model_name)}_{timestamp}'


def replace_arg(cmd: List[str], flag: str, value: str) -> None:
    idx = cmd.index(flag)
    cmd[idx + 1] = value


def build_common_args(args: argparse.Namespace, run_name: str, output_root: Path) -> List[str]:
    cmd = [
        *args.evalscope_cmd,
        'perf',
        '--model',
        args.model,
        '--url',
        args.url,
        '--api',
        args.api,
        '--parallel',
        '1',
        '--number',
        '1',
        '--max-tokens',
        '1',
        '--outputs-dir',
        str(output_root),
        '--name',
        run_name,
        '--stream' if args.stream else '--no-stream',
        '--log-every-n-query',
        str(args.log_every_n_query),
    ]
    if args.api_key is not None:
        cmd.extend(['--api-key', args.api_key])
    if args.no_test_connection:
        cmd.append('--no-test-connection')
    if args.connect_timeout is not None:
        cmd.extend(['--connect-timeout', str(args.connect_timeout)])
    if args.read_timeout is not None:
        cmd.extend(['--read-timeout', str(args.read_timeout)])
    if args.total_timeout is not None:
        cmd.extend(['--total-timeout', str(args.total_timeout)])
    if args.debug:
        cmd.append('--debug')
    return cmd


def build_vl_cmd(args: argparse.Namespace, run_name: str, output_root: Path, parallel: int, number: int) -> List[str]:
    cmd = build_common_args(args, run_name, output_root)
    replace_arg(cmd, '--parallel', str(parallel))
    replace_arg(cmd, '--number', str(number))
    replace_arg(cmd, '--max-tokens', str(args.vl_output_tokens))
    cmd.extend([
        '--dataset',
        'random_vl',
        '--tokenizer-path',
        args.vl_tokenizer_path,
        '--min-prompt-length',
        str(args.vl_min_prompt_length),
        '--max-prompt-length',
        str(args.vl_max_prompt_length),
        '--prefix-length',
        str(args.vl_prefix_length),
        '--image-width',
        str(args.image_width),
        '--image-height',
        str(args.image_height),
        '--image-num',
        str(args.image_num),
        '--image-format',
        args.image_format,
    ])
    return cmd


def build_text_cmd(
    args: argparse.Namespace,
    run_name: str,
    output_root: Path,
    parallel: int,
    number: int,
) -> List[str]:
    cmd = build_common_args(args, run_name, output_root)
    replace_arg(cmd, '--parallel', str(parallel))
    replace_arg(cmd, '--number', str(number))
    replace_arg(cmd, '--max-tokens', str(args.text_output_tokens))
    cmd.extend([
        '--dataset',
        'random',
        '--tokenizer-path',
        args.text_tokenizer_path,
        '--min-prompt-length',
        str(args.text_min_prompt_length),
        '--max-prompt-length',
        str(args.text_max_prompt_length),
        '--prefix-length',
        str(args.text_prefix_length),
    ])
    if args.text_tokenize_prompt:
        cmd.append('--tokenize-prompt')
    return cmd


def run_job(name: str, cmd: Sequence[str]) -> int:
    print(f'[{name}] command: {shlex.join(cmd)}', flush=True)
    env = os.environ.copy()
    repo_root = find_evalscope_repo_root()
    existing_pythonpath = env.get('PYTHONPATH')
    if repo_root:
        env['PYTHONPATH'] = str(repo_root) if not existing_pythonpath else f'{repo_root}:{existing_pythonpath}'

    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(repo_root if repo_root else SCRIPT_DIR),
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f'[{name}] {line}', end='', flush=True)
    return proc.wait()


def find_named_run_dir(batch_output_root: Path, run_name: str) -> Path:
    matches = [p for p in batch_output_root.rglob(run_name) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f'Cannot find run directory for {run_name} under {batch_output_root}')
    return max(matches, key=lambda p: p.stat().st_mtime)


def find_result_db(run_dir: Path) -> Path:
    db_candidates = list(run_dir.rglob('benchmark_data.db'))
    if not db_candidates:
        raise FileNotFoundError(f'Cannot find benchmark_data.db under {run_dir}')
    return max(db_candidates, key=lambda p: p.stat().st_mtime)


def _to_optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_requests_from_db(db_path: Path) -> List[RequestRecord]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            'SELECT start_time, completed_time, success, first_chunk_latency, '
            'time_per_output_token, completion_tokens FROM result'
        ).fetchall()

    records: List[RequestRecord] = []
    for row in rows:
        d = dict(row)
        records.append(
            RequestRecord(
                start_time=float(d.get('start_time') or 0.0),
                completed_time=float(d.get('completed_time') or 0.0),
                success=bool(d.get('success', 0)),
                ttft=_to_optional_float(d.get('first_chunk_latency')),
                tpot=_to_optional_float(d.get('time_per_output_token')),
                completion_tokens=int(d.get('completion_tokens') or 0),
            )
        )
    return records


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None]
    if not vals:
        return math.nan
    return statistics.fmean(vals)


def safe_decay(final_value: float, initial_value: float, reverse: bool = False) -> float:
    if initial_value == 0 or math.isnan(initial_value) or math.isnan(final_value):
        return math.nan
    if reverse:
        return (initial_value - final_value) / initial_value * 100.0
    return (final_value - initial_value) / initial_value * 100.0


def summarize_batch(batch_index: int, records: List[RequestRecord]) -> BatchSummary:
    ordered = sorted(records, key=lambda r: (r.completed_time, r.start_time))
    success_records = [r for r in ordered if r.success]
    total_requests = len(ordered)
    success_requests = len(success_records)
    failed_requests = total_requests - success_requests
    success_rate = (success_requests / total_requests * 100.0) if total_requests else math.nan

    if success_records:
        start_ts = min(r.start_time for r in success_records)
        end_ts = max(r.completed_time for r in success_records)
        duration = max(end_ts - start_ts, 0.0)
        avg_ttft = mean(r.ttft for r in success_records if r.ttft is not None)
        avg_tpot = mean(r.tpot for r in success_records if r.tpot is not None)
        total_tokens = sum(max(r.completion_tokens, 0) for r in success_records)
        avg_tps = (total_tokens / duration) if duration > 0 else math.nan
    else:
        start_ts = min((r.start_time for r in ordered), default=math.nan)
        max_completed = max((r.completed_time for r in ordered if r.completed_time > 0), default=math.nan)
        max_started = max((r.start_time for r in ordered), default=math.nan)
        if math.isnan(max_completed) or max_completed < start_ts:
            end_ts = max_started
        else:
            end_ts = max_completed
        duration = max(end_ts - start_ts, 0.0) if not math.isnan(start_ts) and not math.isnan(end_ts) else math.nan
        avg_ttft = math.nan
        avg_tpot = math.nan
        avg_tps = math.nan

    return BatchSummary(
        batch_index=batch_index,
        duration_seconds=duration,
        total_requests=total_requests,
        success_requests=success_requests,
        failed_requests=failed_requests,
        success_rate_percent=success_rate,
        avg_ttft_seconds=avg_ttft,
        avg_tpot_seconds=avg_tpot,
        avg_tps=avg_tps,
    )


def choose_window_batch_count(total_batches: int) -> int:
    if total_batches <= 3:
        return 0
    candidates = [
        k for k in range(1, total_batches // 2 + 1)
        if 0.15 <= (k / total_batches) <= 0.25
    ]
    if candidates:
        target = total_batches * 0.2
        return min(candidates, key=lambda k: (abs(k - target), k))
    return 1


def summarize_batch_window(batches: Sequence[BatchSummary]) -> dict:
    total_requests = sum(batch.total_requests for batch in batches)
    success_requests = sum(batch.success_requests for batch in batches)
    failed_requests = sum(batch.failed_requests for batch in batches)
    success_rate = (success_requests / total_requests * 100.0) if total_requests else math.nan
    total_duration = sum(
        batch.duration_seconds for batch in batches if batch.duration_seconds is not None and not math.isnan(batch.duration_seconds)
    )
    return {
        'batch_indexes': [batch.batch_index for batch in batches],
        'batch_count': len(batches),
        'duration_seconds': total_duration,
        'total_requests': total_requests,
        'success_requests': success_requests,
        'failed_requests': failed_requests,
        'success_rate_percent': success_rate,
        'avg_ttft_seconds': mean(batch.avg_ttft_seconds for batch in batches if not math.isnan(batch.avg_ttft_seconds)),
        'avg_tpot_seconds': mean(batch.avg_tpot_seconds for batch in batches if not math.isnan(batch.avg_tpot_seconds)),
        'avg_tps': mean(batch.avg_tps for batch in batches if not math.isnan(batch.avg_tps)),
    }


def analyze_batches(batch_summaries: Sequence[BatchSummary]) -> dict:
    total_batches = len(batch_summaries)
    overall = summarize_batch_window(batch_summaries)
    window_batch_count = choose_window_batch_count(total_batches)

    if window_batch_count == 0:
        note = (
            f'批次数为 {total_batches}，不超过 3，按前后批次对比没有统计意义，因此仅展示整体汇总，不计算衰减。'
        )
        empty_window = {
            'batch_indexes': [],
            'batch_count': 0,
            'duration_seconds': math.nan,
            'total_requests': 0,
            'success_requests': 0,
            'failed_requests': 0,
            'success_rate_percent': math.nan,
            'avg_ttft_seconds': math.nan,
            'avg_tpot_seconds': math.nan,
            'avg_tps': math.nan,
        }
        return {
            'comparable': False,
            'note': note,
            'total_batches': total_batches,
            'window_batch_count': 0,
            'overall': overall,
            'early_window': empty_window,
            'late_window': empty_window,
            'ttft_decay_rate_percent': math.nan,
            'tpot_decay_rate_percent': math.nan,
            'tps_decay_rate_percent': math.nan,
        }

    early_batches = list(batch_summaries[:window_batch_count])
    late_batches = list(batch_summaries[-window_batch_count:])
    early_window = summarize_batch_window(early_batches)
    late_window = summarize_batch_window(late_batches)
    return {
        'comparable': True,
        'note': f'按批次统计：前 {window_batch_count} 个批次 vs 后 {window_batch_count} 个批次。',
        'total_batches': total_batches,
        'window_batch_count': window_batch_count,
        'overall': overall,
        'early_window': early_window,
        'late_window': late_window,
        'ttft_decay_rate_percent': safe_decay(
            late_window['avg_ttft_seconds'], early_window['avg_ttft_seconds'], reverse=False
        ),
        'tpot_decay_rate_percent': safe_decay(
            late_window['avg_tpot_seconds'], early_window['avg_tpot_seconds'], reverse=False
        ),
        'tps_decay_rate_percent': safe_decay(
            late_window['avg_tps'], early_window['avg_tps'], reverse=True
        ),
    }


def format_float(value: float, digits: int = 4, percent: bool = False) -> str:
    if value is None or math.isnan(value) or math.isinf(value):
        return 'N/A'
    if percent:
        return f'{value:.2f}%'
    return f'{value:.{digits}f}'


def format_seconds(value: float) -> str:
    if value is None or math.isnan(value):
        return 'N/A'
    return f'{value:.2f}s'


def format_duration(seconds: float) -> str:
    if seconds is None or math.isnan(seconds):
        return 'N/A'
    seconds_int = int(round(seconds))
    hours, rem = divmod(seconds_int, 3600)
    minutes, secs = divmod(rem, 60)
    return f'{hours}h{minutes:02d}m{secs:02d}s'


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    return [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(':---' for _ in headers) + ' |',
        *('| ' + ' | '.join(row) + ' |' for row in rows),
    ]


def three_line_rows(analysis: dict) -> List[List[str]]:
    return [
        [
            '长周期',
            format_duration(analysis['overall']['duration_seconds']),
            str(analysis['overall']['total_requests']),
            format_float(analysis['overall']['success_rate_percent'], digits=2, percent=True),
        ],
        [
            'TTFT 衰减',
            format_seconds(analysis['early_window']['avg_ttft_seconds']),
            format_seconds(analysis['late_window']['avg_ttft_seconds']),
            format_float(analysis['ttft_decay_rate_percent'], digits=2, percent=True),
        ],
        [
            'TPOT 衰减',
            format_seconds(analysis['early_window']['avg_tpot_seconds']),
            format_seconds(analysis['late_window']['avg_tpot_seconds']),
            format_float(analysis['tpot_decay_rate_percent'], digits=2, percent=True),
        ],
        [
            'TPS 衰减',
            format_float(analysis['early_window']['avg_tps'], digits=4),
            format_float(analysis['late_window']['avg_tps'], digits=4),
            format_float(analysis['tps_decay_rate_percent'], digits=2, percent=True),
        ],
    ]


def run_lane_pair(batch_index: int, args: argparse.Namespace, output_root: Path) -> dict:
    batch_root = output_root / 'batches' / f'batch_{batch_index:04d}'
    batch_root.mkdir(parents=True, exist_ok=True)
    vl_name = f'vl_batch_{batch_index:04d}'
    text_name = f'text_batch_{batch_index:04d}'

    vl_cmd = build_vl_cmd(args, vl_name, batch_root, args.vl_parallel, args.vl_number)
    text_cmd = build_text_cmd(args, text_name, batch_root, args.text_parallel, args.text_number)

    results: Dict[str, int] = {}
    errors: Dict[str, str] = {}
    lock = threading.Lock()

    def target(name: str, cmd: List[str]) -> None:
        try:
            rc = run_job(name, cmd)
        except Exception as exc:
            rc = 1
            with lock:
                errors[name] = str(exc)
            print(f'[{name}] unexpected error: {exc}', file=sys.stderr, flush=True)
        with lock:
            results[name] = rc

    threads = [
        threading.Thread(target=target, args=(f'VL-{batch_index:04d}', vl_cmd), daemon=False),
        threading.Thread(target=target, args=(f'TEXT-{batch_index:04d}', text_cmd), daemon=False),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    failed = {name: rc for name, rc in results.items() if rc != 0}
    if failed:
        raise RuntimeError(f'Failed lane batch jobs: {failed}, errors={errors}')

    vl_run_dir = find_named_run_dir(batch_root, vl_name)
    text_run_dir = find_named_run_dir(batch_root, text_name)
    vl_db = find_result_db(vl_run_dir)
    text_db = find_result_db(text_run_dir)
    return {
        'vl_run_dir': vl_run_dir,
        'text_run_dir': text_run_dir,
        'vl_db': vl_db,
        'text_db': text_db,
    }


def run_warmup(args: argparse.Namespace, output_root: Path) -> None:
    if not args.enable_warmup:
        return
    warmup_root = output_root / 'warmup'
    warmup_root.mkdir(parents=True, exist_ok=True)
    print('Starting warmup...', flush=True)
    run_lane_pair(0, args, warmup_root)
    print('Warmup completed.', flush=True)


def write_reports(output_root: Path, args: argparse.Namespace, meta: dict) -> None:
    (output_root / 'mixed_stability_analysis.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )

    combined = meta['analysis']['combined']
    text_lane = meta['analysis']['text_lane']
    vl_lane = meta['analysis']['vl_lane']

    lines = [
        '# EvalScope Mixed Stability Perf Report',
        '',
        f'- Generated At: {meta["generated_at"]}',
        f'- Model: {args.model}',
        f'- URL: {args.url}',
        f'- Target Duration: {args.duration_minutes:.2f}m',
        f'- Output Root: {output_root}',
        '',
        '## Combined',
        '',
        *markdown_table(['项目', '初值/时长', '末值/请求数', '结果'], three_line_rows(combined)),
        '',
        '## Text Lane',
        '',
        *markdown_table(['项目', '初值/时长', '末值/请求数', '结果'], three_line_rows(text_lane)),
        '',
        '## VL Lane',
        '',
        *markdown_table(['项目', '初值/时长', '末值/请求数', '结果'], three_line_rows(vl_lane)),
        '',
        '## 说明',
        '',
        '- Combined 为两路请求明细按批次合并后的整体稳定性结果。',
        '- 长周期时长为各批次有效执行时长之和，不包含批次切换空档。',
        '- TTFT/TPOT/TPS 衰减按批次统计，比较前后相同数量的批次。',
        '- 当总批次数不超过 3 时，不进行前后批次衰减统计，只输出整体汇总。',
        f'- Combined: {combined["note"]}',
        f'- Text Lane: {text_lane["note"]}',
        f'- VL Lane: {vl_lane["note"]}',
    ]
    (output_root / 'mixed_stability_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    tsv_lines = [
        'scope\tmetric\tvalue_1\tvalue_2\tvalue_3',
    ]
    for scope, analysis in (
        ('combined', combined),
        ('text_lane', text_lane),
        ('vl_lane', vl_lane),
    ):
        rows = three_line_rows(analysis)
        for row in rows:
            tsv_lines.append(f'{scope}\t{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}')
    (output_root / 'mixed_stability_report.tsv').write_text('\n'.join(tsv_lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run mixed EvalScope stability benchmarking using the same traffic types as run_evalscope_mixed_perf.py.'
    )
    parser.add_argument('--model', required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--api-key', nargs='?', const='', default=os.getenv('OPENAI_API_KEY'))
    parser.add_argument('--api', default='openai')
    parser.add_argument('--evalscope-bin', default='evalscope')
    parser.add_argument('--tokenizer-path', required=True)
    parser.add_argument('--output-root', default=None)
    parser.add_argument(
        '--duration-minutes',
        type=float,
        default=24.0 * 60.0,
        help='Target benchmark duration in minutes.',
    )
    parser.add_argument('--stream', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--no-test-connection', action='store_true', default=False)
    parser.add_argument('--connect-timeout', type=int, default=None)
    parser.add_argument('--read-timeout', type=int, default=None)
    parser.add_argument('--total-timeout', type=int, default=6 * 60 * 60)
    parser.add_argument('--log-every-n-query', type=int, default=20)
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--enable-warmup', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--vl-parallel', type=int, default=3)
    parser.add_argument('--vl-number', type=int, default=100)
    parser.add_argument('--vl-min-prompt-length', type=int, default=1000)
    parser.add_argument('--vl-max-prompt-length', type=int, default=1000)
    parser.add_argument('--vl-output-tokens', type=int, default=250)
    parser.add_argument('--vl-prefix-length', type=int, default=0)
    parser.add_argument('--vl-tokenizer-path', default=None)
    parser.add_argument('--image-width', type=int, default=720)
    parser.add_argument('--image-height', type=int, default=1280)
    parser.add_argument('--image-num', type=int, default=2)
    parser.add_argument('--image-format', default='RGB')

    parser.add_argument('--text-parallel', type=int, default=5)
    parser.add_argument('--text-number', type=int, default=200)
    parser.add_argument('--text-min-prompt-length', type=int, default=2000)
    parser.add_argument('--text-max-prompt-length', type=int, default=10000)
    parser.add_argument('--text-output-tokens', type=int, default=50)
    parser.add_argument('--text-prefix-length', type=int, default=0)
    parser.add_argument('--text-tokenizer-path', default=None)
    parser.add_argument('--text-tokenize-prompt', action='store_true', default=False)

    args = parser.parse_args()
    if args.duration_minutes <= 0:
        parser.error('--duration-minutes must be > 0')
    args.evalscope_cmd = resolve_evalscope_cmd(args.evalscope_bin)
    if not args.output_root:
        args.output_root = str(build_default_output_root(args.model))
    if not args.vl_tokenizer_path:
        args.vl_tokenizer_path = args.tokenizer_path
    if not args.text_tokenizer_path:
        args.text_tokenizer_path = args.tokenizer_path
    return args


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_warmup(args, output_root)

    target_seconds = args.duration_minutes * 60.0
    started_at = time.time()
    batch_index = 0

    vl_db_paths: List[str] = []
    text_db_paths: List[str] = []
    vl_run_dirs: List[str] = []
    text_run_dirs: List[str] = []
    combined_batches: List[BatchSummary] = []
    text_batches: List[BatchSummary] = []
    vl_batches: List[BatchSummary] = []

    while True:
        elapsed = time.time() - started_at
        if batch_index > 0 and elapsed >= target_seconds:
            break

        batch_index += 1
        batch_result = run_lane_pair(batch_index, args, output_root)
        vl_run_dirs.append(str(batch_result['vl_run_dir']))
        text_run_dirs.append(str(batch_result['text_run_dir']))
        vl_db_paths.append(str(batch_result['vl_db']))
        text_db_paths.append(str(batch_result['text_db']))
        vl_records = load_requests_from_db(batch_result['vl_db'])
        text_records = load_requests_from_db(batch_result['text_db'])
        vl_batches.append(summarize_batch(batch_index, vl_records))
        text_batches.append(summarize_batch(batch_index, text_records))
        combined_batches.append(summarize_batch(batch_index, vl_records + text_records))

        elapsed = time.time() - started_at
        print(
            f'[progress] batch={batch_index}, elapsed={format_duration(elapsed)}, '
            f'target={format_duration(target_seconds)}, text_requests={text_batches[-1].total_requests}, '
            f'vl_requests={vl_batches[-1].total_requests}',
            flush=True,
        )

    analysis = {
        'combined': analyze_batches(combined_batches),
        'text_lane': analyze_batches(text_batches),
        'vl_lane': analyze_batches(vl_batches),
    }

    meta = {
        'generated_at': dt.datetime.now().isoformat(timespec='seconds'),
        'model': args.model,
        'url': args.url,
        'api': args.api,
        'target_duration_minutes': args.duration_minutes,
        'batch_count': batch_index,
        'vl_config': {
            'parallel': args.vl_parallel,
            'number_per_batch': args.vl_number,
            'min_prompt_length': args.vl_min_prompt_length,
            'max_prompt_length': args.vl_max_prompt_length,
            'output_tokens': args.vl_output_tokens,
            'image_width': args.image_width,
            'image_height': args.image_height,
            'image_num': args.image_num,
            'image_format': args.image_format,
        },
        'text_config': {
            'parallel': args.text_parallel,
            'number_per_batch': args.text_number,
            'min_prompt_length': args.text_min_prompt_length,
            'max_prompt_length': args.text_max_prompt_length,
            'output_tokens': args.text_output_tokens,
        },
        'vl_run_dirs': vl_run_dirs,
        'text_run_dirs': text_run_dirs,
        'vl_db_paths': vl_db_paths,
        'text_db_paths': text_db_paths,
        'combined_batch_summaries': [batch.__dict__ for batch in combined_batches],
        'text_batch_summaries': [batch.__dict__ for batch in text_batches],
        'vl_batch_summaries': [batch.__dict__ for batch in vl_batches],
        'analysis': analysis,
    }
    write_reports(output_root, args, meta)

    print('\nMixed stability summary:', flush=True)
    for scope in ('combined', 'text_lane', 'vl_lane'):
        item = analysis[scope]
        print(
            f'  {scope}: duration={format_duration(item["overall"]["duration_seconds"])}, '
            f'requests={item["overall"]["total_requests"]}, success_rate={format_float(item["overall"]["success_rate_percent"], digits=2, percent=True)}, '
            f'ttft_decay={format_float(item["ttft_decay_rate_percent"], digits=2, percent=True)}, '
            f'tps_decay={format_float(item["tps_decay_rate_percent"], digits=2, percent=True)}',
            flush=True,
        )
    print(f'Reports written to: {output_root}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
