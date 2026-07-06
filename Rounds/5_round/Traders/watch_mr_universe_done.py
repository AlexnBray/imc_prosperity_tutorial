#!/usr/bin/env python3
"""Block until MR universe optimisation finishes, then notify + print.

Polls for the summary JSON written by ``run_mr_universe_comparison_batch`` (all
50×2 studies done). Optional desktop notification via ``notify-send`` (Linux).

Usage::

    # Terminal A — watch tqdm-style progress (rewrite lines; still useful in tail)
    tail -f /home/dansp/projects/imc_prosperity_tutorial/mr_universe_run.log

    # Terminal B — wait until done, then beep + notify-send (if available)
    cd Rounds/5_round/Traders
    python3 watch_mr_universe_done.py

Env:
    MR_UNIVERSE_SUMMARY_JSON — same path as batch (default: repo root JSON)
    MR_WATCH_POLL_SEC        — seconds between checks (default: 30)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_TRADERS = Path(__file__).resolve().parent
_REPO = _TRADERS.parent.parent.parent
_DEFAULT_JSON = _REPO / "mr_universe_comparison_results.json"


def main() -> int:
    target = Path(os.getenv("MR_UNIVERSE_SUMMARY_JSON", str(_DEFAULT_JSON))).resolve()
    poll = max(5, int(os.getenv("MR_WATCH_POLL_SEC", "30")))

    print(f"Watching for: {target}", flush=True)
    print(f"Poll every {poll}s · Ctrl+C to stop waiting", flush=True)

    while not target.is_file():
        time.sleep(poll)

    msg = "MR universe finished — wrote mr_universe_comparison_results.json"
    print(f"\n✓ {msg}", flush=True)

    title = os.getenv("MR_WATCH_NOTIFY_TITLE", "IMC MR universe")
    body = os.getenv("MR_WATCH_NOTIFY_BODY", msg)
    try:
        subprocess.run(
            ["notify-send", title, body],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except OSError:
        pass

    if os.getenv("MR_WATCH_BELL", "1").strip().lower() in {"1", "true", "yes"}:
        sys.stdout.write("\a")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
