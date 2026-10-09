"""Fix multiprocess ResourceTracker cleanup on Python 3.12.

Python 3.12 removed RLock._recursion_count(), but multiprocess 0.70.x still
calls it during ResourceTracker.__del__. That prints noisy tracebacks at process
exit without affecting eval results.

Loaded via PYTHONSTARTUP from run_eval_code.sh before any ProcessPool workers
spawn. Safe to import multiple times.
"""

from __future__ import annotations

import os

_PATCHED = False


def apply_multiprocess_py312_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    try:
        import multiprocess.resource_tracker as rt
    except ImportError:
        return

    if getattr(rt.ResourceTracker._stop_locked, "_py312_patched", False):
        return

    def _stop_locked_patched(
        self,
        close=os.close,
        waitpid=os.waitpid,
        waitstatus_to_exitcode=os.waitstatus_to_exitcode,
    ):
        rc_fn = getattr(self._lock, "_recursion_count", None)
        if rc_fn is not None:
            try:
                if rc_fn() > 1:
                    return self._reentrant_call_error()
            except Exception:
                pass
        if self._fd is None:
            return
        if self._pid is None:
            return
        close(self._fd)
        self._fd = None
        waitpid(self._pid, 0)
        self._pid = None

    _stop_locked_patched._py312_patched = True  # type: ignore[attr-defined]
    rt.ResourceTracker._stop_locked = _stop_locked_patched


apply_multiprocess_py312_patch()
