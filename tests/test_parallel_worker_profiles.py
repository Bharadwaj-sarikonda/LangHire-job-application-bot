"""Regression tests for safe parallel Chromium profile handling."""
from backend.core.browser_profiles import (
    create_worker_profiles,
    remove_stale_worker_profiles,
    remove_worker_profiles,
)


def test_parallel_workers_get_isolated_profile_copies(monkeypatch, tmp_path):
    source = tmp_path / "browser_profile"
    (source / "Default").mkdir(parents=True)
    (source / "Default" / "Cookies").write_text("authenticated-session")
    (source / "SingletonLock").write_text("stale-lock")
    (source / "RunningChromeVersion").write_text("transient-version")
    run_dir, profiles = create_worker_profiles(source, 4)
    try:
        assert run_dir is not None
        assert len(profiles) == 4
        assert len(set(profiles)) == 4
        for profile in profiles:
            assert profile != source
            assert (profile / "Default" / "Cookies").read_text() == "authenticated-session"
            assert not (profile / "SingletonLock").exists()
            assert not (profile / "RunningChromeVersion").exists()
    finally:
        remove_worker_profiles(run_dir)

    assert not run_dir.exists()
    assert (source / "Default" / "Cookies").read_text() == "authenticated-session"


def test_single_worker_uses_persistent_profile_without_cleanup(tmp_path):
    source = tmp_path / "browser_profile"
    source.mkdir()
    run_dir, profiles = create_worker_profiles(source, 1)
    remove_worker_profiles(run_dir)

    assert run_dir is None
    assert profiles == [source]
    assert source.exists()


def test_stale_parallel_profiles_are_removed_without_touching_main_profile(tmp_path):
    source = tmp_path / "browser_profile"
    source.mkdir()
    stale_run = tmp_path / "browser_workers" / "run-interrupted" / "worker-1"
    stale_run.mkdir(parents=True)
    (stale_run / "Cookies").write_text("temporary")

    remove_stale_worker_profiles(source)

    assert source.exists()
    assert not stale_run.parent.exists()
