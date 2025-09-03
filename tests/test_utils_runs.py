import time
from pathlib import Path

from object_harvest.utils.runs import resolve_run_dir_base


def test_resolve_run_dir_base_no_resume(tmp_path: Path):
    base = tmp_path / "outputs"
    base.mkdir()
    assert resolve_run_dir_base(str(base), False) == str(base)


def test_resolve_run_dir_base_resume_latest(tmp_path: Path):
    base = tmp_path / "outputs"
    base.mkdir()
    r1 = base / "run-20240101-000000-a"
    r2 = base / "run-20240101-000001-b"
    r1.mkdir()
    r2.mkdir()
    # Ensure r2 is most recent
    time.sleep(0.01)
    (r2 / "marker").write_text("x")
    resolved = resolve_run_dir_base(str(base), True)
    assert resolved.endswith("run-20240101-000001-b")


def test_resolve_run_dir_base_resume_passthrough(tmp_path: Path):
    run_dir = tmp_path / "outputs" / "run-20240101-000000-a"
    run_dir.mkdir(parents=True)
    assert resolve_run_dir_base(str(run_dir), True) == str(run_dir)
