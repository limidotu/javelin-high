"""Render one 16:9 reel into proof/gen_i from an existing 2.5 scan."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_job import run_job
from v1_select import next_gen_n


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else next_gen_n(ROOT)
    scan = ROOT / "scan"
    return run_job(skip_scan=True, scan_dir=scan, gen_n=n)


if __name__ == "__main__":
    raise SystemExit(main())
