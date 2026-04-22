#!/usr/bin/env python3
"""Smoke wrapper for run_evalscope_mixed_perf.py.

This keeps the same mixed request structure but uses small request counts and a
dedicated smoke output directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_SCRIPT = SCRIPT_DIR / 'run_evalscope_mixed_perf.py'
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / 'mixed_perf_smoke_outputs'


def main() -> int:
    argv = sys.argv[1:]
    if '--output-root' not in argv:
        argv.extend(['--output-root', str(DEFAULT_OUTPUT_ROOT)])

    smoke_defaults = [
        '--vl-name',
        'mixed_vl_smoke_2c_20n',
        '--vl-number',
        '20',
        '--vl-parallel',
        '2',
        '--vl-min-prompt-length',
        '256',
        '--vl-max-prompt-length',
        '256',
        '--vl-output-tokens',
        '32',
        '--image-width',
        '224',
        '--image-height',
        '224',
        '--image-num',
        '2',
        '--text-name',
        'mixed_text_smoke_6c_40n',
        '--text-number',
        '40',
        '--text-parallel',
        '6',
        '--text-min-prompt-length',
        '256',
        '--text-max-prompt-length',
        '512',
        '--text-output-tokens',
        '32',
        '--log-every-n-query',
        '10',
    ]

    cmd = [sys.executable, str(TARGET_SCRIPT), *smoke_defaults, *argv]
    env = os.environ.copy()
    return subprocess.call(cmd, env=env, cwd=str(SCRIPT_DIR))


if __name__ == '__main__':
    raise SystemExit(main())
