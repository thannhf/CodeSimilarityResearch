
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.safety.persistence_temp import (
    OwnershipVerificationState,
    TempSyncError,
    create_owned_temp_file,
    sync_owned_temp_file,
)
import src.safety.persistence_temp as persistence_temp

def release_temp(temp):
    try:
        os.close(temp.fd)
    except OSError:
        pass

    try:
        temp.path.unlink(missing_ok=True)
    except OSError:
        pass


def test_sync_calls_fsync_on_retained_fd(tmp_path, monkeypatch):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    calls = []

    monkeypatch.setattr(
        os,
        "fsync",
        lambda fd: calls.append(fd),
    )

    try:
        result = sync_owned_temp_file(temp)

        assert result is None
        assert calls == [temp.fd]
    finally:
        release_temp(temp)


def test_sync_does_not_retry_fsync(tmp_path, monkeypatch):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    calls = {"count": 0}

    def fake_fsync(fd):
        calls["count"] += 1

    monkeypatch.setattr(os, "fsync", fake_fsync)

    try:
        sync_owned_temp_file(temp)

        assert calls["count"] == 1
    finally:
        release_temp(temp)


def test_sync_keeps_descriptor_open(tmp_path, monkeypatch):
    temp = create_owned_temp_file(tmp_path / "output.bin")

    monkeypatch.setattr(os, "fsync", lambda fd: None)

    try:
        sync_owned_temp_file(temp)

        # FD vẫn usable sau sync.
        descriptor_stat = os.fstat(temp.fd)
        assert descriptor_stat.st_ino == temp.identity[1]
    finally:
        release_temp(temp)


def test_sync_preserves_temp_content(tmp_path, monkeypatch):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    payload = b"sync must not modify file content"
    os.write(temp.fd, payload)

    monkeypatch.setattr(os, "fsync", lambda fd: None)

    try:
        sync_owned_temp_file(temp)

        assert temp.path.read_bytes() == payload
    finally:
        release_temp(temp)


def test_sync_rejects_invalid_ownership(tmp_path, monkeypatch):
    calls = []

    monkeypatch.setattr(
        os,
        "fsync",
        lambda fd: calls.append(fd),
    )

    with pytest.raises(TypeError):
        sync_owned_temp_file(tmp_path / "invalid")

    assert calls == []


def test_sync_rejects_missing_temp_path(tmp_path, monkeypatch):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    os.close(temp.fd)
    temp.path.unlink()

    calls = []

    monkeypatch.setattr(
        os,
        "fsync",
        lambda fd: calls.append(fd),
    )

    with pytest.raises(TempSyncError, match="missing"):
        sync_owned_temp_file(temp)

    assert calls == []


def test_sync_rejects_replaced_file(tmp_path, monkeypatch):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    os.close(temp.fd)

    temp.path.unlink()
    temp.path.write_bytes(b"foreign file")

    calls = []

    monkeypatch.setattr(
        os,
        "fsync",
        lambda fd: calls.append(fd),
    )

    try:
        with pytest.raises(TempSyncError, match="mismatch"):
            sync_owned_temp_file(temp)

        assert calls == []
    finally:
        temp.path.unlink(missing_ok=True)


def test_sync_rejects_replaced_symlink(tmp_path, monkeypatch):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    target = tmp_path / "foreign-target.txt"
    target.write_text("foreign target", encoding="utf-8")

    os.close(temp.fd)
    temp.path.unlink()

    try:
        temp.path.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is not supported here.")

    calls = []

    monkeypatch.setattr(
        os,
        "fsync",
        lambda fd: calls.append(fd),
    )

    try:
        with pytest.raises(TempSyncError, match="mismatch"):
            sync_owned_temp_file(temp)

        assert calls == []
    finally:
        temp.path.unlink(missing_ok=True)


def test_sync_rejects_closed_descriptor(tmp_path, monkeypatch):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    os.close(temp.fd)

    calls = []

    monkeypatch.setattr(
        os,
        "fsync",
        lambda fd: calls.append(fd),
    )

    try:
        with pytest.raises(TempSyncError, match="unverifiable"):
            sync_owned_temp_file(temp)

        assert calls == []
    finally:
        temp.path.unlink(missing_ok=True)


def test_sync_rejects_foreign_descriptor(tmp_path, monkeypatch):
    temp = create_owned_temp_file(tmp_path / "output.bin")

    foreign_path = tmp_path / "foreign.bin"
    foreign_path.write_bytes(b"foreign")
    foreign_fd = os.open(foreign_path, os.O_RDONLY)

    tampered = replace(temp, fd=foreign_fd)
    calls = []

    monkeypatch.setattr(
        os,
        "fsync",
        lambda fd: calls.append(fd),
    )

    try:
        with pytest.raises(TempSyncError, match="mismatch"):
            sync_owned_temp_file(tampered)

        assert calls == []
    finally:
        os.close(foreign_fd)
        release_temp(temp)


def test_sync_rejects_forged_identity(tmp_path, monkeypatch):
    temp = create_owned_temp_file(tmp_path / "output.bin")

    forged_identity = (
        temp.identity[0] + 1,
        temp.identity[1] + 1,
    )
    tampered = replace(temp, identity=forged_identity)
    calls = []

    monkeypatch.setattr(
        os,
        "fsync",
        lambda fd: calls.append(fd),
    )

    try:
        with pytest.raises(TempSyncError, match="mismatch"):
            sync_owned_temp_file(tampered)

        assert calls == []
    finally:
        release_temp(temp)


def test_sync_wraps_oserror_and_preserves_cause(tmp_path, monkeypatch):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    original_error = OSError("simulated fsync failure")
    calls = []

    def failing_fsync(fd):
        calls.append(fd)
        raise original_error

    monkeypatch.setattr(os, "fsync", failing_fsync)

    try:
        with pytest.raises(TempSyncError) as exc_info:
            sync_owned_temp_file(temp)

        assert calls == [temp.fd]
        assert exc_info.value.__cause__ is original_error
    finally:
        release_temp(temp)


def test_sync_interrupted_is_not_retried(tmp_path, monkeypatch):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    calls = {"count": 0}
    original_error = InterruptedError("simulated interruption")

    def interrupted_fsync(fd):
        calls["count"] += 1
        raise original_error

    monkeypatch.setattr(os, "fsync", interrupted_fsync)

    try:
        with pytest.raises(TempSyncError) as exc_info:
            sync_owned_temp_file(temp)

        assert calls["count"] == 1
        assert exc_info.value.__cause__ is original_error
    finally:
        release_temp(temp)


def test_sync_failure_keeps_descriptor_open(tmp_path, monkeypatch):
    temp = create_owned_temp_file(tmp_path / "output.bin")

    monkeypatch.setattr(
        os,
        "fsync",
        lambda fd: (_ for _ in ()).throw(
            OSError("simulated failure")
        ),
    )

    try:
        with pytest.raises(TempSyncError):
            sync_owned_temp_file(temp)

        # Sync failure không close FD.
        assert os.fstat(temp.fd).st_ino == temp.identity[1]
    finally:
        release_temp(temp)


def test_sync_does_not_reopen_temp_path(tmp_path, monkeypatch):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    fsync_calls = []

    monkeypatch.setattr(
        os,
        "fsync",
        lambda fd: fsync_calls.append(fd),
    )

    # Sync primitive không được reopen pathname.
    def forbidden_open(*args, **kwargs):
        raise AssertionError("Temp pathname must not be reopened")

    monkeypatch.setattr(os, "open", forbidden_open)

    try:
        sync_owned_temp_file(temp)

        assert fsync_calls == [temp.fd]
    finally:
        monkeypatch.undo()
        release_temp(temp)


def test_sync_does_not_cleanup_or_commit(tmp_path, monkeypatch):
    destination = tmp_path / "output.bin"
    temp = create_owned_temp_file(destination)
    payload = b"still only a temp file"
    os.write(temp.fd, payload)

    monkeypatch.setattr(os, "fsync", lambda fd: None)

    try:
        sync_owned_temp_file(temp)

        assert temp.path.exists()
        assert temp.path.read_bytes() == payload
        assert not destination.exists()
    finally:
        release_temp(temp)


@pytest.mark.parametrize(
    "state",
    [
        OwnershipVerificationState.MISSING,
        OwnershipVerificationState.MISMATCH,
        OwnershipVerificationState.UNVERIFIABLE,
    ],
)
def test_sync_fail_closed_for_all_unverified_states(
    tmp_path,
    monkeypatch,
    state,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    fsync_calls = []

    fake_verification = SimpleNamespace(
        state=state,
    )

    monkeypatch.setattr(
        persistence_temp,
        "verify_temp_ownership",
        lambda ownership: fake_verification,
    )
    monkeypatch.setattr(
        os,
        "fsync",
        lambda fd: fsync_calls.append(fd),
    )

    try:
        with pytest.raises(TempSyncError):
            sync_owned_temp_file(temp)

        # Không được gọi fsync với bất kỳ ownership state nào
        # chưa VERIFIED.
        assert fsync_calls == []
    finally:
        release_temp(temp)


def test_sync_verifies_ownership_before_fsync(
    tmp_path,
    monkeypatch,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    events = []

    real_verifier = persistence_temp.verify_temp_ownership

    def tracked_verifier(ownership):
        events.append(("verify", ownership.fd))
        return real_verifier(ownership)

    def tracked_fsync(fd):
        events.append(("fsync", fd))

    monkeypatch.setattr(
        persistence_temp,
        "verify_temp_ownership",
        tracked_verifier,
    )
    monkeypatch.setattr(
        os,
        "fsync",
        tracked_fsync,
    )

    try:
        sync_owned_temp_file(temp)

        assert events == [
            ("verify", temp.fd),
            ("fsync", temp.fd),
        ]
    finally:
        release_temp(temp)


def test_sync_passes_original_ownership_to_verifier(
    tmp_path,
    monkeypatch,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    verified_objects = []

    real_verifier = persistence_temp.verify_temp_ownership

    def tracked_verifier(ownership):
        verified_objects.append(ownership)
        return real_verifier(ownership)

    monkeypatch.setattr(
        persistence_temp,
        "verify_temp_ownership",
        tracked_verifier,
    )
    monkeypatch.setattr(
        os,
        "fsync",
        lambda fd: None,
    )

    try:
        sync_owned_temp_file(temp)

        assert len(verified_objects) == 1
        assert verified_objects[0] is temp
    finally:
        release_temp(temp)


def test_sync_does_not_call_fsync_when_verification_raises(
    tmp_path,
    monkeypatch,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    fsync_calls = []

    injected_error = RuntimeError(
        "injected ownership verification crash"
    )

    def failing_verifier(ownership):
        raise injected_error

    monkeypatch.setattr(
        persistence_temp,
        "verify_temp_ownership",
        failing_verifier,
    )
    monkeypatch.setattr(
        os,
        "fsync",
        lambda fd: fsync_calls.append(fd),
    )

    try:
        with pytest.raises(RuntimeError) as exc_info:
            sync_owned_temp_file(temp)

        assert exc_info.value is injected_error
        assert fsync_calls == []
    finally:
        release_temp(temp)


def test_sync_does_not_write_or_change_temp_size(
    tmp_path,
    monkeypatch,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    payload = b"payload before sync"

    os.write(temp.fd, payload)
    size_before = os.fstat(temp.fd).st_size
    position_before = os.lseek(temp.fd, 0, os.SEEK_CUR)

    fsync_calls = []

    monkeypatch.setattr(
        os,
        "fsync",
        lambda fd: fsync_calls.append(fd),
    )

    try:
        sync_owned_temp_file(temp)

        size_after = os.fstat(temp.fd).st_size
        position_after = os.lseek(temp.fd, 0, os.SEEK_CUR)

        assert size_after == size_before
        assert position_after == position_before
        assert temp.path.read_bytes() == payload
        assert fsync_calls == [temp.fd]
    finally:
        release_temp(temp)


def test_sync_has_no_retry_after_fsync_oserror(
    tmp_path,
    monkeypatch,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    fsync_calls = []

    def failing_fsync(fd):
        fsync_calls.append(fd)
        raise OSError("injected persistent sync failure")

    monkeypatch.setattr(
        os,
        "fsync",
        failing_fsync,
    )

    try:
        with pytest.raises(TempSyncError):
            sync_owned_temp_file(temp)

        assert len(fsync_calls) == 1
    finally:
        release_temp(temp)
