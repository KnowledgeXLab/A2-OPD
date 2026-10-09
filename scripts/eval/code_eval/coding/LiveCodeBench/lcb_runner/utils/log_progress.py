"""Log-friendly progress lines for non-interactive batch jobs."""

from __future__ import annotations

import os
import sys
from datetime import datetime


def progress_timestamp() -> str:
    tz_name = os.environ.get("OPD_LOG_TZ", "Asia/Shanghai")
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(tz_name)).strftime("%F %T")
    except Exception:
        return datetime.now().strftime("%F %T")


def format_progress(
    note: str,
    completed: int,
    total: int,
    *,
    detail: str = "",
    bar_width: int = 16,
    with_timestamp: bool = True,
) -> str:
    pct = 100.0 * completed / total if total else 0.0
    filled = int(bar_width * completed / total) if total else 0
    bar = "#" * filled + "." * (bar_width - filled)
    width = len(str(total)) if total else 1
    prefix = f"[{progress_timestamp()}] " if with_timestamp else ""
    msg = f"{prefix}[{note}] {completed:>{width}}/{total} ({pct:5.1f}%) [{bar}]"
    if detail:
        msg += f" {detail}"
    return msg


def should_log_progress(completed: int, total: int, *, min_interval: int = 1) -> bool:
    if completed >= total:
        return True
    if total <= 0:
        return False
    step = max(min_interval, total // 50)
    return completed % step == 0


def log_progress(
    note: str,
    completed: int,
    total: int,
    *,
    detail: str = "",
    min_interval: int = 1,
) -> None:
    if not should_log_progress(completed, total, min_interval=min_interval):
        return
    print(format_progress(note, completed, total, detail=detail), flush=True)


def is_interactive() -> bool:
    return sys.stdout.isatty()
