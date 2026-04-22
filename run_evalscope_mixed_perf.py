#!/usr/bin/env python3
"""Run two EvalScope native perf jobs concurrently.

Workload default:
- 300 multimodal requests, concurrency 2, each request uses 1000 text tokens
  plus 2 images (720x1280), output 250 tokens.
- 600 text-only requests, concurrency 6, input length sampled uniformly from
  2000 to 10000 tokens, output 50 tokens.

The script launches two independent `evalscope perf` subprocesses so each load
profile still uses EvalScope's native perf implementation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR if (SCRIPT_DIR / 'evalscope').exists() else SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = WORKSPACE_DIR / 'mixed_perf_outputs'


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


def resolve_evalscope_cmd(evalscope_bin: str) -> List[str]:
    repo_root = find_evalscope_repo_root()
    repo_venv_python = repo_root / '.venv' / 'bin' / 'python' if repo_root else None
    cli_module_cmd = [sys.executable, '-m', 'evalscope.cli.cli']

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

    cli_entry = repo_root / 'evalscope' / 'cli' / 'cli.py' if repo_root else None
    if cli_entry and cli_entry.exists():
        return cli_module_cmd

    raise FileNotFoundError(
        f'Cannot find EvalScope executable: {evalscope_bin}. '
        'Pass --evalscope-bin explicitly or run this script from a workspace that contains the evalscope repo.'
    )


def is_python_executable(path: Path) -> bool:
    name = path.name.lower()
    return name.startswith('python')


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
    model_stub = sanitize_name(model_name)
    return DEFAULT_OUTPUT_ROOT / f'{model_stub}_{timestamp}'


def validate_tokenizer_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.tokenizer_path:
        parser.error(
            '--tokenizer-path is required. Use the real tokenizer repo id or local tokenizer directory, '
            'for example: --tokenizer-path Qwen/Qwen3-VL-30B-A3B-Instruct'
        )


def build_common_args(args: argparse.Namespace, run_name: str) -> List[str]:
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
        str(args.parallel_placeholder),
        '--number',
        str(args.number_placeholder),
        '--max-tokens',
        str(args.max_tokens_placeholder),
        '--outputs-dir',
        str(Path(args.output_root) / run_name),
        '--name',
        run_name,
        '--stream' if args.stream else '--no-stream',
        '--log-every-n-query',
        str(args.log_every_n_query),
    ]

    if args.api_key:
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


def build_vl_cmd(args: argparse.Namespace) -> List[str]:
    run_name = args.vl_name
    common = build_common_args(args, run_name)
    replace_arg(common, '--parallel', str(args.vl_parallel))
    replace_arg(common, '--number', str(args.vl_number))
    replace_arg(common, '--max-tokens', str(args.vl_output_tokens))
    cmd = common + [
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
    ]
    return cmd


def build_text_cmd(args: argparse.Namespace) -> List[str]:
    run_name = args.text_name
    common = build_common_args(args, run_name)
    replace_arg(common, '--parallel', str(args.text_parallel))
    replace_arg(common, '--number', str(args.text_number))
    replace_arg(common, '--max-tokens', str(args.text_output_tokens))
    cmd = common + [
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
    ]
    if args.text_tokenize_prompt:
        cmd.append('--tokenize-prompt')
    return cmd


def replace_arg(cmd: List[str], flag: str, value: str) -> None:
    idx = cmd.index(flag)
    cmd[idx + 1] = value


def stream_output(name: str, pipe, sink) -> None:
    try:
        for line in iter(pipe.readline, ''):
            sink.write(f'[{name}] {line}')
            sink.flush()
    finally:
        pipe.close()


def run_job(name: str, cmd: List[str]) -> int:
    print(f'[{name}] command: {shlex.join(cmd)}', flush=True)
    env = os.environ.copy()
    repo_root = find_evalscope_repo_root()
    existing_pythonpath = env.get('PYTHONPATH')
    if repo_root:
        env['PYTHONPATH'] = str(repo_root) if not existing_pythonpath else f'{repo_root}:{existing_pythonpath}'

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(repo_root if repo_root else SCRIPT_DIR),
        )
    except Exception as exc:
        print(f'[{name}] failed to start subprocess: {exc}', file=sys.stderr, flush=True)
        return 1

    stdout_thread = threading.Thread(target=stream_output, args=(name, proc.stdout, sys.stdout), daemon=True)
    stderr_thread = threading.Thread(target=stream_output, args=(name, proc.stderr, sys.stderr), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    rc = proc.wait()
    stdout_thread.join()
    stderr_thread.join()
    print(f'[{name}] finished with exit code {rc}', flush=True)
    return rc


def extract_section(lines: List[str], title: str) -> List[str]:
    for idx, line in enumerate(lines):
        if line.strip() == title:
            start = idx
            end = len(lines)
            for j in range(idx + 1, len(lines)):
                if lines[j].strip() and lines[j].strip() != title and not _is_table_line(lines[j]) and not lines[j].startswith(' '):
                    end = j
                    break
            while end > start and not lines[end - 1].strip():
                end -= 1
            return lines[start:end]
    return []


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    table_markers = ('│', '┃', '┌', '┐', '└', '┘', '┏', '┓', '┡', '┢', '┠', '┨', '├', '┤', '┬', '┴', '┼', '╭', '╰')
    return stripped.startswith(table_markers)


def read_summary_sections(summary_path: Path) -> Dict[str, List[str]]:
    lines = summary_path.read_text(errors='replace').splitlines()
    sections: Dict[str, List[str]] = {}
    for title in ('Basic Information:', 'Detailed Performance Metrics', 'Request Metrics'):
        sections[title] = extract_section(lines, title)
    return sections


def parse_box_table_data_row(section: List[str]) -> List[str]:
    """Extract the first data row from EvalScope's rich box table output."""
    for line in section:
        stripped = line.strip()
        if not stripped.startswith('│'):
            continue
        cells = [cell.strip() for cell in stripped.strip('│').split('│')]
        if not cells:
            continue
        first = cells[0].replace(',', '')
        if first.isdigit():
            return cells
    return []


def parse_metric_rows(summary_path: Path) -> Dict[str, str]:
    sections = read_summary_sections(summary_path)
    detailed = parse_box_table_data_row(sections.get('Detailed Performance Metrics', []))
    request = parse_box_table_data_row(sections.get('Request Metrics', []))

    metrics: Dict[str, str] = {}
    detailed_keys = [
        'Conc.',
        'Rate',
        'RPS',
        'Avg Lat.(s)',
        'P99 Lat.(s)',
        'Avg TTFT(s)',
        'P99 TTFT(s)',
        'Avg TPOT(s)',
        'P99 TPOT(s)',
        'Gen. toks/s',
        'Success Rate',
    ]
    request_keys = [
        'Conc.',
        'Num Reqs',
        'Avg In Toks',
        'P99 In Toks',
        'Avg Out Toks',
        'P99 Out Toks',
    ]

    for key, value in zip(detailed_keys, detailed):
        metrics[key] = value
    for key, value in zip(request_keys, request):
        metrics.setdefault(key, value)
    return metrics


def markdown_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    return [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(':---' for _ in headers) + ' |',
        *('| ' + ' | '.join(row) + ' |' for row in rows),
    ]


def workload_column_title(model: str, workload_name: str, metrics: Dict[str, str]) -> str:
    conc = metrics.get('Conc.', '?')
    return f'{model}<br>{workload_name} (Conc.{conc})'


def generate_combined_report(output_root: Path, model: str, url: str, summaries: Dict[str, Path]) -> Path:
    timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metrics_by_name = {name: parse_metric_rows(path) for name, path in summaries.items()}

    text_metrics = metrics_by_name.get('TEXT', {})
    vl_metrics = metrics_by_name.get('VL', {})
    text_title = workload_column_title(model, '纯文本', text_metrics)
    vl_title = workload_column_title(model, '图文', vl_metrics)

    primary_metric_names = [
        'Conc.',
        'RPS',
        'Avg Lat.(s)',
        'P99 Lat.(s)',
        'Avg TTFT(s)',
        'P99 TTFT(s)',
        'Avg TPOT(s)',
        'P99 TPOT(s)',
        'Gen. toks/s',
        'Success Rate',
    ]
    request_metric_names = [
        'Num Reqs',
        'Avg In Toks',
        'P99 In Toks',
        'Avg Out Toks',
        'P99 Out Toks',
    ]

    def row(metric_name: str) -> List[str]:
        return [
            f'**{metric_name}**',
            text_metrics.get(metric_name, 'N/A'),
            vl_metrics.get(metric_name, 'N/A'),
        ]

    report_lines = [
        '# EvalScope Mixed Perf Combined Report',
        '',
        f'- Generated At: {timestamp}',
        f'- Model: {model}',
        f'- URL: {url}',
        f'- Output Root: {output_root}',
        '',
        '## 主要性能指标',
        '',
        *markdown_table(['指标', text_title, vl_title], [row(name) for name in primary_metric_names]),
        '',
        '## 请求规模',
        '',
        *markdown_table(['指标', text_title, vl_title], [row(name) for name in request_metric_names]),
        '',
        '## 原始报告路径',
        '',
    ]
    for name, path in summaries.items():
        report_lines.append(f'- {name}: {path}')

    report_path = output_root / 'combined_performance_summary.txt'
    report_path.write_text('\n'.join(report_lines) + '\n')
    return report_path


def find_latest_summary(run_dir: Path) -> Path | None:
    summaries = list(run_dir.rglob('performance_summary.txt'))
    if not summaries:
        return None
    return max(summaries, key=lambda path: path.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run multimodal and text-only EvalScope native perf jobs at the same time.'
    )
    parser.add_argument('--model', required=True, help='Model name passed to both perf jobs.')
    parser.add_argument('--url', required=True, help='OpenAI-compatible endpoint for both jobs.')
    parser.add_argument(
        '--api-key',
        nargs='?',
        const='',
        default=os.getenv('OPENAI_API_KEY'),
        help='API key for the endpoint. You can omit the option, or pass bare --api-key for empty auth.',
    )
    parser.add_argument('--api', default='openai', help='EvalScope perf API type.')
    parser.add_argument(
        '--evalscope-bin',
        default='evalscope',
        help='EvalScope executable. Example: evalscope or /path/to/evalscope',
    )
    parser.add_argument(
        '--tokenizer-path',
        required=True,
        help='Common tokenizer path for both workloads. Must be a local path or repo id such as Qwen/Qwen3-VL-30B-A3B-Instruct.',
    )
    parser.add_argument(
        '--output-root',
        default=None,
        help='Root directory used to store the two perf outputs. Default: /home/houxianzhou/vlm_eval_workspace/mixed_perf_outputs/<model>_<timestamp>',
    )
    parser.add_argument('--stream', action=argparse.BooleanOptionalAction, default=True, help='Use streaming mode.')
    parser.add_argument('--no-test-connection', action='store_true', default=False)
    parser.add_argument('--connect-timeout', type=int, default=None)
    parser.add_argument('--read-timeout', type=int, default=None)
    parser.add_argument('--total-timeout', type=int, default=6 * 60 * 60)
    parser.add_argument('--log-every-n-query', type=int, default=20)
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--vl-name', default='mixed_vl_2c_300n')
    parser.add_argument('--vl-number', type=int, default=300)
    parser.add_argument('--vl-parallel', type=int, default=2)
    parser.add_argument('--vl-min-prompt-length', type=int, default=1000)
    parser.add_argument('--vl-max-prompt-length', type=int, default=1000)
    parser.add_argument('--vl-output-tokens', type=int, default=250)
    parser.add_argument('--vl-prefix-length', type=int, default=0)
    parser.add_argument('--vl-tokenizer-path', default=None, help='Tokenizer path for random_vl generation.')
    parser.add_argument('--image-width', type=int, default=720)
    parser.add_argument('--image-height', type=int, default=1280)
    parser.add_argument('--image-num', type=int, default=2)
    parser.add_argument('--image-format', default='RGB')

    parser.add_argument('--text-name', default='mixed_text_6c_600n')
    parser.add_argument('--text-number', type=int, default=600)
    parser.add_argument('--text-parallel', type=int, default=6)
    parser.add_argument('--text-min-prompt-length', type=int, default=2000)
    parser.add_argument('--text-max-prompt-length', type=int, default=10000)
    parser.add_argument('--text-output-tokens', type=int, default=50)
    parser.add_argument('--text-prefix-length', type=int, default=0)
    parser.add_argument('--text-tokenizer-path', default=None, help='Tokenizer path for random text generation.')
    parser.add_argument(
        '--text-tokenize-prompt',
        action='store_true',
        default=False,
        help='Send raw token ids for the text workload via /v1/completions.',
    )

    args = parser.parse_args()
    args.parallel_placeholder = 1
    args.number_placeholder = 1
    args.max_tokens_placeholder = 1
    args.evalscope_cmd = resolve_evalscope_cmd(args.evalscope_bin)
    if not args.output_root:
        args.output_root = str(build_default_output_root(args.model))

    if not args.vl_tokenizer_path:
        args.vl_tokenizer_path = args.tokenizer_path
    if not args.text_tokenizer_path:
        args.text_tokenizer_path = args.tokenizer_path

    if args.text_min_prompt_length > args.text_max_prompt_length:
        parser.error('--text-min-prompt-length must be <= --text-max-prompt-length')
    if args.vl_min_prompt_length > args.vl_max_prompt_length:
        parser.error('--vl-min-prompt-length must be <= --vl-max-prompt-length')
    validate_tokenizer_args(parser, args)
    return args


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    vl_cmd = build_vl_cmd(args)
    text_cmd = build_text_cmd(args)

    results = {}
    errors = {}
    lock = threading.Lock()

    def target(name: str, cmd: List[str]) -> None:
        try:
            rc = run_job(name, cmd)
        except Exception as exc:  # defensive: threads must never fail silently
            rc = 1
            with lock:
                errors[name] = str(exc)
            print(f'[{name}] unexpected thread error: {exc}', file=sys.stderr, flush=True)
        with lock:
            results[name] = rc

    threads = [
        threading.Thread(target=target, args=('VL', vl_cmd), daemon=False),
        threading.Thread(target=target, args=('TEXT', text_cmd), daemon=False),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if len(results) != 2:
        print(f'Only collected partial results: {results}', file=sys.stderr)
        return 1

    if errors:
        print(f'Thread errors: {errors}', file=sys.stderr)

    failed = {name: rc for name, rc in results.items() if rc != 0}
    if failed:
        print(f'Failed jobs: {failed}', file=sys.stderr)
        return 1

    summary_candidates = {
        'VL': find_latest_summary(output_root / args.vl_name),
        'TEXT': find_latest_summary(output_root / args.text_name),
    }
    missing_summaries = {
        name: str(output_root / (args.vl_name if name == 'VL' else args.text_name))
        for name, path in summary_candidates.items()
        if path is None
    }
    if missing_summaries:
        print(f'Missing performance summaries: {missing_summaries}', file=sys.stderr)
        return 1

    combined_report = generate_combined_report(output_root, args.model, args.url, summary_candidates)  # type: ignore[arg-type]

    print('All perf jobs finished successfully.')
    print(f'Output root: {args.output_root}')
    print(f'Combined report: {combined_report}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
