#!/usr/bin/env python3
"""Smoke wrapper for run_evalscope_mixed_stability_perf.py.

This keeps the same mixed request structure but reduces batch sizes and target
duration, and writes to a dedicated smoke output directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_SCRIPT = SCRIPT_DIR / 'run_evalscope_mixed_stability_perf.py'
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / 'mixed_stability_smoke_outputs'


def main() -> int:
    argv = sys.argv[1:]
    if '--output-root' not in argv:
        argv.extend(['--output-root', str(DEFAULT_OUTPUT_ROOT)])

    smoke_defaults = [
        '--duration-hours',
        '0.0333333333',
        '--vl-parallel',
        '2',
        '--vl-number',
        '10',
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
        '1',
        '--text-parallel',
        '6',
        '--text-number',
        '20',
        '--text-min-prompt-length',
        '256',
        '--text-max-prompt-length',
        '512',
        '--text-output-tokens',
        '32',
        '--log-every-n-query',
        '10',
        '--analysis-window-ratio',
        '0.2',
    ]

    cmd = [sys.executable, str(TARGET_SCRIPT), *smoke_defaults, *argv]
    env = os.environ.copy()
    return subprocess.call(cmd, env=env, cwd=str(SCRIPT_DIR))


if __name__ == '__main__':
    raise SystemExit(main())
