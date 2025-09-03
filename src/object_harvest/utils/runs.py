from __future__ import annotations

import os

__all__ = ["resolve_run_dir_base"]


def resolve_run_dir_base(out: str, resume: bool) -> str:
    """Resolve the base directory for writer based on resume and out path.

    If resume and out is a directory (not already a run-*), select the latest run-* under it.
    Otherwise, return out unchanged.
    """
    writer_base = out
    if resume:
        try:
            if os.path.isdir(out) and not os.path.basename(out).startswith("run-"):
                run_dirs = [
                    os.path.join(out, d)
                    for d in os.listdir(out)
                    if d.startswith("run-") and os.path.isdir(os.path.join(out, d))
                ]
                if run_dirs:
                    writer_base = max(run_dirs, key=os.path.getmtime)
        except Exception:
            pass
    return writer_base
