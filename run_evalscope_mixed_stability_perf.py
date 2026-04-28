#!/usr/bin/env python3
"""Run a long-duration mixed EvalScope perf benchmark and generate stability reports.

This script preserves the request types from `run_evalscope_mixed_perf.py`:

- one multimodal lane using `random_vl`
- one text-only lane using `random`

Each lane runs continuously in-process by reusing EvalScope's dataset/API/client
and DB helpers. The runner stops scheduling new requests when the target
duration is reached, then waits for in-flight requests to finish.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import os
import random
import sqlite3
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR if (SCRIPT_DIR / 'evalscope').exists() else SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = WORKSPACE_DIR / 'mixed_stability_outputs'


@dataclass
class RequestRecord:
    start_time: float
    completed_time: float
    success: bool
    latency: float | None
    ttft: float | None
    tpot: float | None
    completion_tokens: int


@dataclass
class WindowSummary:
    label: str
    start_offset_seconds: float
    end_offset_seconds: float
    duration_seconds: float
    total_requests: int
    success_requests: int
    failed_requests: int
    success_rate_percent: float
    avg_ttft_seconds: float
    p99_ttft_seconds: float
    avg_tpot_seconds: float
    p99_tpot_seconds: float
    avg_tps: float
    p99_tps: float
    avg_e2e_seconds: float
    p99_e2e_seconds: float


@dataclass
class ContinuousLaneResult:
    run_dir: Path
    db_path: Path
    records: List[RequestRecord]


@dataclass
class WarmupLaneResult:
    lane: str
    total_requests: int
    success_requests: int
    failed_requests: int
    duration_seconds: float


def ensure_evalscope_importable() -> None:
    """Require EvalScope from the active Python environment.

    Do not add sibling source trees or local virtualenvs to sys.path here. The
    runner must use the package selected by the invoking interpreter, e.g.
    `conda run -n evalscope python ...`.
    """
    try:
        import evalscope  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            'Cannot import EvalScope from the active Python environment. '
            'Install evalscope into this environment or run with the intended '
            'environment Python, for example: conda run -n evalscope python ...'
        ) from exc


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


def ensure_process_tmpdir(output_root: Path) -> None:
    tmp_dir = output_root / '.tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir_str = str(tmp_dir)
    os.environ.setdefault('TMPDIR', tmp_dir_str)
    os.environ.setdefault('TEMP', tmp_dir_str)
    os.environ.setdefault('TMP', tmp_dir_str)


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
            'SELECT start_time, completed_time, success, latency, first_chunk_latency, '
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
                latency=_to_optional_float(d.get('latency')),
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


def percentile(values: Iterable[float], percent: float) -> float:
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * percent / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return vals[int(rank)]
    weight = rank - lower
    return vals[lower] * (1.0 - weight) + vals[upper] * weight


def _bucket_tps(
    records: Sequence[RequestRecord],
    window_start: float,
    duration_seconds: float,
    bucket_seconds: int = 60,
) -> List[float]:
    if duration_seconds <= 0:
        return []
    bucket_count = max(1, math.ceil(duration_seconds / bucket_seconds))
    tokens_by_bucket = [0 for _ in range(bucket_count)]
    for record in records:
        if not record.success:
            continue
        offset = max(record.start_time - window_start, 0.0)
        index = min(int(offset // bucket_seconds), bucket_count - 1)
        tokens_by_bucket[index] += max(record.completion_tokens, 0)

    rates: List[float] = []
    for index, token_count in enumerate(tokens_by_bucket):
        bucket_start = index * bucket_seconds
        bucket_end = min((index + 1) * bucket_seconds, duration_seconds)
        bucket_duration = max(bucket_end - bucket_start, 0.0)
        if bucket_duration > 0:
            rates.append(token_count / bucket_duration)
    return rates


def summarize_time_window(
    label: str,
    records: Sequence[RequestRecord],
    base_start: float,
    start_offset_seconds: float,
    end_offset_seconds: float,
) -> WindowSummary:
    total_requests = len(records)
    success_records = [r for r in records if r.success]
    success_requests = len(success_records)
    failed_requests = total_requests - success_requests
    success_rate = (success_requests / total_requests * 100.0) if total_requests else math.nan

    ttft_values = [r.ttft for r in success_records if r.ttft is not None]
    tpot_values = [r.tpot for r in success_records if r.tpot is not None]
    e2e_values = [r.latency for r in success_records if r.latency is not None]
    fallback_duration = max(end_offset_seconds - start_offset_seconds, 0.0)
    duration = fallback_duration
    total_tokens = sum(max(r.completion_tokens, 0) for r in success_records)
    avg_tps = (total_tokens / duration) if duration > 0 else math.nan
    bucket_tps = _bucket_tps(success_records, base_start + start_offset_seconds, fallback_duration)

    return WindowSummary(
        label=label,
        start_offset_seconds=start_offset_seconds,
        end_offset_seconds=end_offset_seconds,
        duration_seconds=duration,
        total_requests=total_requests,
        success_requests=success_requests,
        failed_requests=failed_requests,
        success_rate_percent=success_rate,
        avg_ttft_seconds=mean(ttft_values),
        p99_ttft_seconds=percentile(ttft_values, 99),
        avg_tpot_seconds=mean(tpot_values),
        p99_tpot_seconds=percentile(tpot_values, 99),
        avg_tps=avg_tps,
        p99_tps=percentile(bucket_tps, 99),
        avg_e2e_seconds=mean(e2e_values),
        p99_e2e_seconds=percentile(e2e_values, 99),
    )


def summarize_time_windows(records: Sequence[RequestRecord], duration_minutes: float, window_minutes: float) -> dict:
    ordered = sorted(records, key=lambda r: (r.start_time, r.completed_time))
    target_seconds = duration_minutes * 60.0
    window_seconds = window_minutes * 60.0
    window_count = max(1, math.ceil(target_seconds / window_seconds))
    base_start = min((r.start_time for r in ordered), default=time.time())

    windows: List[WindowSummary] = []
    for index in range(window_count):
        start_offset = index * window_seconds
        end_offset = min((index + 1) * window_seconds, target_seconds)
        window_start = base_start + start_offset
        window_end = base_start + end_offset
        window_records = [r for r in ordered if window_start <= r.start_time < window_end]
        label = f'{format_duration(start_offset)}-{format_duration(end_offset)}'
        windows.append(summarize_time_window(label, window_records, base_start, start_offset, end_offset))

    overall_label = f'Overall ({format_duration(target_seconds)})'
    overall = summarize_time_window(overall_label, ordered, base_start, 0.0, target_seconds)
    return {
        'window_minutes': window_minutes,
        'target_duration_minutes': duration_minutes,
        'window_count': window_count,
        'base_start': base_start,
        'overall_label': overall_label,
        'windows': [window.__dict__ for window in windows],
        'overall': overall.__dict__,
    }


def format_float(value: float, digits: int = 4, percent: bool = False) -> str:
    if value is None or math.isnan(value) or math.isinf(value):
        return 'N/A'
    if percent:
        return f'{value:.2f}%'
    return f'{value:.{digits}f}'


def format_duration(seconds: float) -> str:
    if seconds is None or math.isnan(seconds):
        return 'N/A'
    seconds_int = int(round(seconds))
    hours, rem = divmod(seconds_int, 3600)
    minutes, secs = divmod(rem, 60)
    return f'{hours}h{minutes:02d}m{secs:02d}s'


def compute_fluctuation_percent(values: Sequence[float]) -> float:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if len(valid) < 2:
        return math.nan
    min_val = min(valid)
    if min_val == 0:
        return math.nan
    return (max(valid) - min_val) / min_val * 100.0


def _fluctuation_cells(time_analysis: dict, metric: str) -> List[str]:
    windows = time_analysis['windows']
    if metric == 'success_rate':
        values = [w['success_rate_percent'] for w in windows]
        f = compute_fluctuation_percent(values)
        return [format_float(f, digits=2, percent=True)] * 2
    if metric == 'request_count':
        values = [w['total_requests'] for w in windows]
        f = compute_fluctuation_percent(values)
        return [format_float(f, digits=2, percent=True)] * 2
    avg_key = f'avg_{metric}'
    p99_key = f'p99_{metric}'
    avg_values = [w[avg_key] for w in windows]
    p99_values = [w[p99_key] for w in windows]
    return [
        format_float(compute_fluctuation_percent(avg_values), digits=2, percent=True),
        format_float(compute_fluctuation_percent(p99_values), digits=2, percent=True),
    ]


CALCULATION_METHOD_LINES = [
    'TTFT、TPOT、E2E：每个时间窗口内仅统计成功请求，单元格为 avg/p99。',
    'TPS avg：时间窗口内成功请求 completion_tokens 总数 / 窗口时长。',
    'TPS p99：时间窗口内按 1 分钟 bucket 计算 token/s 后取 p99。',
    'Avg波动：该指标各时间窗口 avg 值的 (max - min) / min × 100%，不包含 Overall。',
    'P99波动：该指标各时间窗口 p99 值的 (max - min) / min × 100%，不包含 Overall。',
    '请求成功率和请求数只有单值序列，Avg波动和 P99波动列展示同一个波动值。',
]


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    return [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(':---' for _ in headers) + ' |',
        *('| ' + ' | '.join(row) + ' |' for row in rows),
    ]


def text_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    all_rows = [list(headers), *[list(row) for row in rows]]
    widths = [
        max(len(str(row[index])) for row in all_rows)
        for index in range(len(headers))
    ]

    def format_row(row: Sequence[str]) -> str:
        return '  '.join(str(value).ljust(widths[index]) for index, value in enumerate(row))

    separator = '  '.join('-' * width for width in widths)
    return [
        format_row(headers),
        separator,
        *(format_row(row) for row in rows),
    ]


def compact_metric_cell(avg_value: float, p99_value: float, digits: int = 4) -> str:
    return f'{format_float(avg_value, digits=digits)}/{format_float(p99_value, digits=digits)}'


def format_metric_cell(avg_value: float, p99_value: float, digits: int = 4, suffix: str = '') -> str:
    return f'avg={format_float(avg_value, digits=digits)}{suffix} / p99={format_float(p99_value, digits=digits)}{suffix}'


def window_table_headers(time_analysis: dict) -> List[str]:
    headers = ['指标']
    headers.extend(window['label'] for window in time_analysis['windows'])
    headers.append(time_analysis.get('overall_label', 'Overall'))
    headers.append('Avg波动')
    headers.append('P99波动')
    return headers


def window_metric_rows(time_analysis: dict) -> List[List[str]]:
    windows = list(time_analysis['windows'])
    overall = time_analysis['overall']
    series = windows + [overall]
    return [
        [
            'TTFT(s)',
            *(format_metric_cell(item['avg_ttft_seconds'], item['p99_ttft_seconds']) for item in series),
            *_fluctuation_cells(time_analysis, 'ttft_seconds'),
        ],
        [
            'TPOT(s)',
            *(format_metric_cell(item['avg_tpot_seconds'], item['p99_tpot_seconds']) for item in series),
            *_fluctuation_cells(time_analysis, 'tpot_seconds'),
        ],
        [
            'TPS(tok/s)',
            *(format_metric_cell(item['avg_tps'], item['p99_tps']) for item in series),
            *_fluctuation_cells(time_analysis, 'tps'),
        ],
        [
            'E2E(s)',
            *(format_metric_cell(item['avg_e2e_seconds'], item['p99_e2e_seconds']) for item in series),
            *_fluctuation_cells(time_analysis, 'e2e_seconds'),
        ],
    ]


def compact_window_metric_rows(time_analysis: dict) -> List[List[str]]:
    windows = list(time_analysis['windows'])
    overall = time_analysis['overall']
    series = windows + [overall]
    return [
        [
            'TTFT(s)',
            *(compact_metric_cell(item['avg_ttft_seconds'], item['p99_ttft_seconds']) for item in series),
            *_fluctuation_cells(time_analysis, 'ttft_seconds'),
        ],
        [
            'TPOT(s)',
            *(compact_metric_cell(item['avg_tpot_seconds'], item['p99_tpot_seconds']) for item in series),
            *_fluctuation_cells(time_analysis, 'tpot_seconds'),
        ],
        [
            'TPS(tok/s)',
            *(compact_metric_cell(item['avg_tps'], item['p99_tps']) for item in series),
            *_fluctuation_cells(time_analysis, 'tps'),
        ],
        [
            'E2E(s)',
            *(compact_metric_cell(item['avg_e2e_seconds'], item['p99_e2e_seconds']) for item in series),
            *_fluctuation_cells(time_analysis, 'e2e_seconds'),
        ],
    ]


def success_rate_row(time_analysis: dict) -> List[List[str]]:
    series = list(time_analysis['windows']) + [time_analysis['overall']]
    return [
        [
            '请求成功率',
            *(format_float(item['success_rate_percent'], digits=2, percent=True) for item in series),
            *_fluctuation_cells(time_analysis, 'success_rate'),
        ]
    ]


def request_count_row(time_analysis: dict) -> List[List[str]]:
    series = list(time_analysis['windows']) + [time_analysis['overall']]
    return [
        [
            '请求数',
            *(f'{item["success_requests"]}/{item["total_requests"]}' for item in series),
            *_fluctuation_cells(time_analysis, 'request_count'),
        ]
    ]


def print_analysis_tables(time_analysis: dict) -> None:
    headers = window_table_headers(time_analysis)
    rows = compact_window_metric_rows(time_analysis)
    success_rows = success_rate_row(time_analysis)
    count_rows = request_count_row(time_analysis)
    print('\nMixed stability window metrics (avg/p99):', flush=True)
    for line in text_table(headers, rows):
        print(line, flush=True)
    print('\nMixed stability success rate:', flush=True)
    for line in text_table(headers, success_rows):
        print(line, flush=True)
    print('\nMixed stability request counts (success/total):', flush=True)
    for line in text_table(headers, count_rows):
        print(line, flush=True)
    print('\n计算方法:', flush=True)
    for line in CALCULATION_METHOD_LINES:
        print(f'- {line}', flush=True)


def _first_scalar(value):
    if isinstance(value, list):
        return value[0]
    return value


def build_evalscope_lane_args(args: argparse.Namespace, lane: str, run_dir: Path):
    ensure_evalscope_importable()
    from evalscope.perf.arguments import Arguments

    if lane == 'vl':
        lane_args = Arguments(
            model=args.model,
            url=args.url,
            api=args.api,
            api_key=args.api_key,
            tokenizer_path=args.vl_tokenizer_path,
            outputs_dir=str(run_dir),
            no_timestamp=True,
            name='vl_continuous',
            dataset='random_vl',
            parallel=args.vl_parallel,
            number=args.vl_number,
            max_prompt_length=args.vl_max_prompt_length,
            min_prompt_length=args.vl_min_prompt_length,
            prefix_length=args.vl_prefix_length,
            max_tokens=args.vl_output_tokens,
            image_width=args.image_width,
            image_height=args.image_height,
            image_num=args.image_num,
            image_format=args.image_format,
            stream=args.stream,
            no_test_connection=args.no_test_connection,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            total_timeout=args.total_timeout,
            log_every_n_query=args.log_every_n_query,
            debug=args.debug,
        )
    else:
        lane_args = Arguments(
            model=args.model,
            url=args.url,
            api=args.api,
            api_key=args.api_key,
            tokenizer_path=args.text_tokenizer_path,
            outputs_dir=str(run_dir),
            no_timestamp=True,
            name='text_continuous',
            dataset='random',
            parallel=args.text_parallel,
            number=args.text_number,
            max_prompt_length=args.text_max_prompt_length,
            min_prompt_length=args.text_min_prompt_length,
            prefix_length=args.text_prefix_length,
            max_tokens=args.text_output_tokens,
            tokenize_prompt=args.text_tokenize_prompt,
            stream=args.stream,
            no_test_connection=args.no_test_connection,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            total_timeout=args.total_timeout,
            log_every_n_query=args.log_every_n_query,
            debug=args.debug,
        )

    lane_args.parallel = _first_scalar(lane_args.parallel)
    lane_args.number = _first_scalar(lane_args.number)
    lane_args.rate = args.vl_rate if lane == 'vl' else args.text_rate
    return lane_args


def build_request_pool(lane_args, api_plugin) -> List[dict]:
    from evalscope.perf.plugin import DatasetRegistry
    from evalscope.perf.utils.db_util import load_prompt

    if lane_args.prompt:
        prompt = load_prompt(lane_args.prompt)
        messages = [{'role': 'user', 'content': prompt}] if lane_args.apply_chat_template else prompt
        request = api_plugin.build_request(messages)
        return [request]

    message_generator = DatasetRegistry.get_class(lane_args.dataset)(lane_args)
    requests: List[dict] = []
    for messages in message_generator.build_messages():
        request = api_plugin.build_request(messages)
        if request is not None:
            requests.append(request)
        if len(requests) >= lane_args.number:
            break

    if not requests:
        raise ValueError(f'No requests generated for dataset {lane_args.dataset}')
    return requests


async def write_continuous_lane_results(queue: asyncio.Queue, done_event: asyncio.Event, lane_args, api_plugin) -> Tuple[object, Path]:
    from evalscope.perf.utils.benchmark_util import MetricsAccumulator
    from evalscope.perf.utils.db_util import create_result_table, get_result_db_path, insert_benchmark_data

    accumulator = MetricsAccumulator(concurrency=lane_args.parallel, rate=lane_args.rate)
    db_path = Path(get_result_db_path(lane_args))
    commit_every = lane_args.db_commit_interval
    processed_since_commit = 0

    print(f'[{lane_args.name}] writer started, db={db_path}', flush=True)

    with sqlite3.connect(db_path, check_same_thread=False) as con:
        cursor = con.cursor()
        create_result_table(cursor)
        while not (done_event.is_set() and queue.empty()):
            try:
                benchmark_data = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            try:
                accumulator.update(benchmark_data, api_plugin)
                insert_benchmark_data(cursor, benchmark_data)
            except Exception as exc:
                print(f'[{lane_args.name}] writer process error: {exc}', file=sys.stderr, flush=True)
                queue.task_done()
                continue

            processed_since_commit += 1
            if processed_since_commit >= commit_every:
                await asyncio.to_thread(con.commit)
                processed_since_commit = 0

            if int(accumulator.n_total) % lane_args.log_every_n_query == 0:
                message = accumulator.to_result().create_message(api_type=lane_args.api)
                print(f'[{lane_args.name}] {json.dumps(message, ensure_ascii=False)}', flush=True)

            queue.task_done()

        await asyncio.to_thread(con.commit)

    return accumulator.to_result(), db_path


async def run_continuous_lane(lane: str, args: argparse.Namespace, output_root: Path, target_seconds: float) -> ContinuousLaneResult:
    ensure_evalscope_importable()
    from evalscope.perf.http_client import AioHttpClient, test_connection
    from evalscope.perf.plugin import ApiRegistry
    from evalscope.perf.utils.db_util import summary_result

    run_dir = output_root / 'continuous' / lane
    run_dir.mkdir(parents=True, exist_ok=True)
    lane_args = build_evalscope_lane_args(args, lane, run_dir)
    api_plugin = ApiRegistry.get_class(lane_args.api)(lane_args)

    if not lane_args.no_test_connection:
        ok = await test_connection(lane_args, api_plugin)
        if not ok:
            raise TimeoutError(f'{lane} connection test failed')

    requests = build_request_pool(lane_args, api_plugin)
    print(
        f'[{lane}] continuous run: parallel={lane_args.parallel}, sample_pool={len(requests)}, '
        f'rate={lane_args.rate}, target={format_duration(target_seconds)}, output={run_dir}',
        flush=True,
    )

    queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, lane_args.parallel * lane_args.queue_size_multiplier))
    done_event = asyncio.Event()
    writer_task = asyncio.create_task(write_continuous_lane_results(queue, done_event, lane_args, api_plugin))

    scheduled = 0
    request_index = 0
    schedule_lock = asyncio.Lock()
    started_at = time.time()
    deadline = started_at + target_seconds
    next_request_time = started_at

    async def worker(worker_index: int, client):
        nonlocal scheduled, request_index, next_request_time
        while time.time() < deadline:
            async with schedule_lock:
                if time.time() >= deadline:
                    break
                if lane_args.rate != -1:
                    wait_seconds = max(0.0, next_request_time - time.time())
                    if wait_seconds > 0:
                        await asyncio.sleep(wait_seconds)
                    if time.time() >= deadline:
                        break
                    next_request_time = time.time() + random.expovariate(lane_args.rate)
                request = requests[request_index % len(requests)]
                request_index += 1
                scheduled += 1
            remaining = (deadline + args.request_timeout) - time.time()
            if remaining <= 0:
                break
            request_timeout = min(remaining, args.request_timeout)
            try:
                benchmark_data = await asyncio.wait_for(client.post(request), timeout=request_timeout)
            except asyncio.TimeoutError:
                print(
                    f'[{lane}] worker-{worker_index} request timed out after {request_timeout:.0f}s, skipping',
                    flush=True,
                )
                continue
            except Exception as exc:
                print(f'[{lane}] worker-{worker_index} request error: {exc}', flush=True)
                continue
            benchmark_data.update_gpu_usage()
            try:
                await asyncio.wait_for(queue.put(benchmark_data), timeout=args.queue_put_timeout)
            except asyncio.TimeoutError:
                print(
                    f'[{lane}] worker-{worker_index} queue.put timed out after '
                    f'{args.queue_put_timeout:.0f}s (writer may have crashed), stopping',
                    flush=True,
                )
                break

    client = AioHttpClient(lane_args, api_plugin)
    async with client:
        worker_tasks = [
            asyncio.create_task(worker(index, client))
            for index in range(lane_args.parallel)
        ]
        results = await asyncio.gather(*worker_tasks, return_exceptions=True)
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(f'{lane} request worker failures: {failures[:3]}')

        print(
            f'[{lane}] duration reached; stop scheduling new requests, waited for active workers={lane_args.parallel}',
            flush=True,
        )

    await queue.join()
    done_event.set()
    metrics, db_path = await writer_task
    summary_result(lane_args, metrics, str(db_path))
    records = load_requests_from_db(db_path)
    print(f'[{lane}] scheduled={scheduled}, recorded={len(records)}, db={db_path}', flush=True)
    return ContinuousLaneResult(run_dir=run_dir, db_path=db_path, records=records)


def split_warmup_requests(total_requests: int) -> Tuple[int, int]:
    if total_requests <= 0:
        return 0, 0
    vl_requests = max(1, total_requests * 3 // 8) if total_requests > 1 else 0
    text_requests = total_requests - vl_requests
    return vl_requests, text_requests


async def run_warmup_lane(lane: str, args: argparse.Namespace, output_root: Path, request_count: int) -> WarmupLaneResult:
    ensure_evalscope_importable()
    from evalscope.perf.http_client import AioHttpClient, test_connection
    from evalscope.perf.plugin import ApiRegistry

    if request_count <= 0:
        return WarmupLaneResult(lane=lane, total_requests=0, success_requests=0, failed_requests=0, duration_seconds=0.0)

    run_dir = output_root / 'warmup' / lane
    run_dir.mkdir(parents=True, exist_ok=True)
    lane_args = build_evalscope_lane_args(args, lane, run_dir)
    lane_args.number = max(int(request_count), 1)
    lane_args.parallel = min(int(lane_args.parallel), int(request_count))
    api_plugin = ApiRegistry.get_class(lane_args.api)(lane_args)

    if not lane_args.no_test_connection:
        ok = await test_connection(lane_args, api_plugin)
        if not ok:
            raise TimeoutError(f'{lane} warmup connection test failed')

    requests = build_request_pool(lane_args, api_plugin)
    semaphore = asyncio.Semaphore(lane_args.parallel)

    async def post_request(client, request):
        async with semaphore:
            benchmark_data = await client.post(request)
        return benchmark_data

    started_at = time.time()
    client = AioHttpClient(lane_args, api_plugin)
    async with client:
        tasks = [
            asyncio.create_task(post_request(client, requests[index % len(requests)]))
            for index in range(request_count)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    success = 0
    failed = 0
    for result in results:
        if isinstance(result, Exception):
            failed += 1
        elif getattr(result, 'success', False):
            success += 1
        else:
            failed += 1
    duration = time.time() - started_at
    print(
        f'[warmup:{lane}] requests={request_count}, success={success}, failed={failed}, '
        f'duration={format_duration(duration)}',
        flush=True,
    )
    return WarmupLaneResult(
        lane=lane,
        total_requests=request_count,
        success_requests=success,
        failed_requests=failed,
        duration_seconds=duration,
    )


async def run_warmup(args: argparse.Namespace, output_root: Path) -> None:
    warmup_requests = args.warmup_requests
    if warmup_requests <= 0:
        print('[warmup] skipped because --warmup-requests <= 0', flush=True)
        return

    vl_requests, text_requests = split_warmup_requests(warmup_requests)
    print(
        f'[warmup] start: total={warmup_requests}, VL={vl_requests}, TEXT={text_requests}; '
        'warmup only prints logs and does not generate reports',
        flush=True,
    )
    started_at = time.time()
    vl_task = asyncio.create_task(run_warmup_lane('vl', args, output_root, vl_requests))
    text_task = asyncio.create_task(run_warmup_lane('text', args, output_root, text_requests))
    results = await asyncio.gather(vl_task, text_task)
    total = sum(result.total_requests for result in results)
    success = sum(result.success_requests for result in results)
    failed = sum(result.failed_requests for result in results)
    print(
        f'[warmup] done: requests={total}, success={success}, failed={failed}, '
        f'duration={format_duration(time.time() - started_at)}',
        flush=True,
    )
    if failed:
        raise RuntimeError(f'Warmup failed: {failed}/{total} requests failed')


async def run_continuous_mixed(args: argparse.Namespace, output_root: Path) -> Tuple[ContinuousLaneResult, ContinuousLaneResult]:
    target_seconds = args.duration_minutes * 60.0
    vl_task = asyncio.create_task(run_continuous_lane('vl', args, output_root, target_seconds))
    text_task = asyncio.create_task(run_continuous_lane('text', args, output_root, target_seconds))
    vl_result, text_result = await asyncio.gather(vl_task, text_task)
    return vl_result, text_result


def write_reports(output_root: Path, args: argparse.Namespace, meta: dict) -> None:
    (output_root / 'mixed_stability_analysis.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )

    time_analysis = meta['time_window_analysis']
    headers = window_table_headers(time_analysis)
    metric_rows = window_metric_rows(time_analysis)
    success_rows = success_rate_row(time_analysis)
    count_rows = request_count_row(time_analysis)

    lines = [
        '# EvalScope Mixed Stability Perf Report',
        '',
        f'- Generated At: {meta["generated_at"]}',
        f'- Model: {args.model}',
        f'- URL: {args.url}',
        f'- Target Duration: {args.duration_minutes:.2f}m',
        f'- Window Minutes: {args.window_minutes:.2f}m',
        f'- Output Root: {output_root}',
        '',
        '## Combined Window Metrics',
        '',
        *markdown_table(headers, metric_rows),
        '',
        '## Request Success Rate',
        '',
        *markdown_table(headers, success_rows),
        '',
        '## Request Counts',
        '',
        *markdown_table(headers, count_rows),
        '',
        '## 说明',
        '',
        '- 表格只统计图文和纯文本合并后的整体数据，不区分请求类型。',
        *(f'- {line}' for line in CALCULATION_METHOD_LINES),
        '- 到达目标时长后停止调度新请求，并等待已发出的 in-flight 请求完成。',
        f'- VL DB: {meta["vl_db_paths"][0]}',
        f'- Text DB: {meta["text_db_paths"][0]}',
    ]
    (output_root / 'mixed_stability_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    tsv_lines = [
        'metric\t' + '\t'.join(headers[1:]),
    ]
    for row in metric_rows + success_rows + count_rows:
        tsv_lines.append('\t'.join(row))
    (output_root / 'mixed_stability_report.tsv').write_text('\n'.join(tsv_lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run mixed EvalScope stability benchmarking using the same traffic types as run_evalscope_mixed_perf.py.'
    )
    parser.add_argument('--model', required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--api-key', nargs='?', const='', default=os.getenv('OPENAI_API_KEY'))
    parser.add_argument('--api', default='openai')
    parser.add_argument('--tokenizer-path', required=True)
    parser.add_argument('--output-root', default=None)
    parser.add_argument(
        '--duration-minutes',
        type=float,
        default=720.0,
        help='Target benchmark duration in minutes. Default 720 (12 hours).',
    )
    parser.add_argument(
        '--window-minutes',
        type=float,
        default=60.0,
        help='Analysis window size in minutes. Default 60 minutes.',
    )
    parser.add_argument('--stream', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--no-test-connection', action='store_true', default=False)
    parser.add_argument('--connect-timeout', type=int, default=None)
    parser.add_argument('--read-timeout', type=int, default=None)
    parser.add_argument('--total-timeout', type=int, default=6 * 60 * 60)
    parser.add_argument(
        '--request-timeout',
        type=float,
        default=300.0,
        help='Per request wall-clock timeout in seconds during continuous runs. Default 300.',
    )
    parser.add_argument(
        '--queue-put-timeout',
        type=float,
        default=30.0,
        help='Timeout in seconds when enqueueing completed request data for the result writer. Default 30.',
    )
    parser.add_argument('--log-every-n-query', type=int, default=20)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument(
        '--rate',
        type=float,
        default=1.0,
        help='Total mixed request start rate in req/s. Default 1.0. Use -1 for closed-loop max throughput.',
    )
    parser.add_argument(
        '--warmup-requests',
        type=int,
        default=30,
        help='Total warmup requests before the stability run. Set 0 to skip. Default 30.',
    )

    parser.add_argument(
        '--parallel',
        type=int,
        default=8,
        help='Total mixed concurrency. Must be a positive multiple of 8; split as 3/8 multimodal (VL) and 5/8 text-only. Default 8.',
    )
    parser.add_argument('--vl-parallel', type=int, default=2)
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

    parser.add_argument('--text-parallel', type=int, default=6)
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
    if args.window_minutes <= 0:
        parser.error('--window-minutes must be > 0')
    if args.warmup_requests < 0:
        parser.error('--warmup-requests must be >= 0')
    if args.request_timeout <= 0:
        parser.error('--request-timeout must be > 0')
    if args.queue_put_timeout <= 0:
        parser.error('--queue-put-timeout must be > 0')
    if args.rate != -1 and args.rate <= 0:
        parser.error('--rate must be > 0, or -1 for closed-loop max throughput')
    args.vl_rate = -1 if args.rate == -1 else args.rate * 3.0 / 8.0
    args.text_rate = -1 if args.rate == -1 else args.rate * 5.0 / 8.0
    if args.parallel is not None:
        if args.parallel <= 0:
            parser.error('--parallel must be a positive integer')
        if args.parallel % 8 != 0:
            parser.error('--parallel must be a multiple of 8 so multimodal:text-only concurrency can be split 3:5')
        args.vl_parallel = args.parallel * 3 // 8
        args.text_parallel = args.parallel * 5 // 8
    if args.vl_number <= 0:
        parser.error('--vl-number must be > 0')
    if args.text_number <= 0:
        parser.error('--text-number must be > 0')
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
    ensure_process_tmpdir(output_root)

    target_seconds = args.duration_minutes * 60.0
    if args.parallel is not None:
        print(
            f'Total concurrency {args.parallel}: VL={args.vl_parallel}, TEXT={args.text_parallel}',
            flush=True,
        )
    print(f'Total request rate {args.rate}: VL={args.vl_rate}, TEXT={args.text_rate}', flush=True)
    asyncio.run(run_warmup(args, output_root))
    print(f'Continuous mixed stability target: {format_duration(target_seconds)}', flush=True)

    vl_result, text_result = asyncio.run(run_continuous_mixed(args, output_root))
    combined_records = vl_result.records + text_result.records
    time_window_analysis = summarize_time_windows(combined_records, args.duration_minutes, args.window_minutes)

    meta = {
        'generated_at': dt.datetime.now().isoformat(timespec='seconds'),
        'model': args.model,
        'url': args.url,
        'api': args.api,
        'target_duration_minutes': args.duration_minutes,
        'window_minutes': args.window_minutes,
        'execution_mode': 'continuous',
        'summary_count': time_window_analysis['window_count'],
        'vl_config': {
            'parallel': args.vl_parallel,
            'sample_pool_size': args.vl_number,
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
            'sample_pool_size': args.text_number,
            'min_prompt_length': args.text_min_prompt_length,
            'max_prompt_length': args.text_max_prompt_length,
            'output_tokens': args.text_output_tokens,
        },
        'vl_run_dirs': [str(vl_result.run_dir)],
        'text_run_dirs': [str(text_result.run_dir)],
        'vl_db_paths': [str(vl_result.db_path)],
        'text_db_paths': [str(text_result.db_path)],
        'time_window_analysis': time_window_analysis,
    }
    write_reports(output_root, args, meta)

    print_analysis_tables(time_window_analysis)
    print(f'Reports written to: {output_root}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
