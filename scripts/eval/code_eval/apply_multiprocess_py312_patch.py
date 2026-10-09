#!/usr/bin/env python3
"""One-time patch for installed multiprocess/resource_tracker.py (optional).

Prefer PYTHONSTARTUP via multiprocess_py312_compat.py for eval runs.
Use this if you want the fix inside site-packages (survives until reinstall).
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    try:
        import multiprocess
    except ImportError:
        print("multiprocess not installed; nothing to patch", file=sys.stderr)
        return 0

    path = Path(multiprocess.__file__).parent / "resource_tracker.py"
    text = path.read_text()
    old = "if self._lock._recursion_count() > 1:"
    new = "if getattr(self._lock, '_recursion_count', lambda: 0)() > 1:"

    if new in text:
        print(f"already patched: {path}")
        return 0
    if old not in text:
        print(f"unexpected resource_tracker.py layout; skipped: {path}", file=sys.stderr)
        return 1

    backup = path.with_suffix(".py.bak")
    if not backup.exists():
        backup.write_text(text)
    path.write_text(text.replace(old, new, 1))
    print(f"patched: {path}")
    print(f"backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
