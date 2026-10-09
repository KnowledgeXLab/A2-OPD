"""Log-friendly progress lines for non-interactive eval batch jobs."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any


def progress_timestamp() -> str:
    """Local time for log progress lines (default Asia/Shanghai, no timezone suffix)."""
    fmt = os.environ.get("OPD_LOG_TS_FMT", "%Y-%m-%d %H:%M:%S")
    tz_name = os.environ.get("OPD_LOG_TZ", "Asia/Shanghai")
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(tz_name)).strftime(fmt)
    except Exception:
        return datetime.now().strftime(fmt)


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


def sample_id_range(batch_items: list[dict[str, Any]]) -> str:
    ids = [item.get("id") for item in batch_items if item.get("id") is not None]
    if not ids:
        return ""
    if len(ids) == 1:
        return str(ids[0])
    return f"{ids[0]}..{ids[-1]}"


def format_batch_detail(
    batch_idx: int,
    num_batches: int,
    batch_size: int,
    *,
    n: int = 1,
    sample_range: str = "",
) -> str:
    if n > 1:
        task_part = f"| {batch_size} samples x{n}"
    else:
        task_part = f"| {batch_size} samples"
    parts = [f"| batch {batch_idx}/{num_batches}", task_part]
    if sample_range:
        parts.append(f"| {sample_range}")
    return " ".join(parts)


def is_interactive() -> bool:
    return sys.stdout.isatty()
