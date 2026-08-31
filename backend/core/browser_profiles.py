"""Safe Chromium profile lifecycle for concurrent application workers."""
import shutil
import time
from pathlib import Path
from uuid import uuid4


def create_worker_profiles(base_profile: Path, worker_count: int) -> tuple[Path | None, list[Path]]:
    """Create isolated, authenticated browser profiles for parallel workers.

    Chromium permits only one process to use a profile at a time. A single
    worker keeps using the normal persistent profile; parallel workers receive
    short-lived copies so they retain login cookies without sharing singleton
    locks.
    """
    if worker_count <= 1:
        return None, [base_profile]

    base_profile.mkdir(parents=True, exist_ok=True)
    run_dir = base_profile.parent / "browser_workers" / f"run-{uuid4().hex}"
    run_dir.mkdir(parents=True)
    ignore_transient_files = shutil.ignore_patterns(
        "SingletonCookie", "SingletonLock", "SingletonSocket", "LOCK",
        "DevToolsActivePort", "RunningChromeVersion",
    )
    profiles = []
    try:
        for worker_id in range(1, worker_count + 1):
            profile_dir = run_dir / f"worker-{worker_id}"
            _copy_profile_snapshot(base_profile, profile_dir, ignore_transient_files)
            profiles.append(profile_dir)
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    return run_dir, profiles


def _copy_profile_snapshot(source: Path, destination: Path, ignore) -> None:
    """Copy a profile snapshot while tolerating Chromium runtime files vanishing."""
    for attempt in range(3):
        try:
            shutil.copytree(source, destination, ignore=ignore)
            return
        except shutil.Error as exc:
            shutil.rmtree(destination, ignore_errors=True)
            errors = exc.args[0]
            source_files_disappeared = all(
                "No such file or directory" in error and not Path(src).exists()
                for src, _destination, error in errors
            )
            if not source_files_disappeared or attempt == 2:
                raise
            time.sleep(0.1)


def remove_worker_profiles(run_dir: Path | None) -> None:
    """Remove only the temporary profiles created for one parallel run."""
    if run_dir is not None:
        shutil.rmtree(run_dir, ignore_errors=True)


def remove_stale_worker_profiles(base_profile: Path) -> None:
    """Remove run directories left behind after an interrupted application run."""
    workers_root = base_profile.parent / "browser_workers"
    if not workers_root.exists():
        return
    for run_dir in workers_root.glob("run-*"):
        if run_dir.is_dir() and not run_dir.is_symlink():
            shutil.rmtree(run_dir, ignore_errors=True)
