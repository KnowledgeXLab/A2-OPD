import os
import sys
from datetime import datetime
from typing import Any, Iterable, Iterator, Optional, TypeVar

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

T = TypeVar("T")


def progress_timestamp() -> str:
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


class _LogProgress:
    """Progress helper that prints percentage lines for non-interactive logs."""

    def __init__(self, note: str):
        self.note = note
        self.console = Console(force_terminal=False, no_color=True)
        self.completed = 0
        self.total = 0

    def track(self, iterable: Iterable[T]) -> Iterator[T]:
        self.total = len(iterable) if hasattr(iterable, "__len__") else 0
        for item in iterable:
            yield item
            self.completed += 1
            if self.total > 0:
                print(
                    format_progress(self.note, self.completed, self.total),
                    flush=True,
                )

    def __enter__(self) -> "_LogProgress":
        return self

    def __exit__(self, *args: Any) -> None:
        return None


def progress(note: str = "processing"):
    if sys.stdout.isatty():
        return Progress(
            TextColumn(f"{note} •" + "[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        )
    return _LogProgress(note)


def log_tqdm(iterable: Iterable[T], desc: str = "progress", **kwargs: Any):
    from tqdm import tqdm

    is_tty = sys.stdout.isatty()
    return tqdm(
        iterable,
        desc=desc,
        file=sys.stdout,
        dynamic_ncols=not is_tty,
        mininterval=0.1 if is_tty else 5.0,
        bar_format="{desc}: {percentage:3.0f}%|{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        **kwargs,
    )
