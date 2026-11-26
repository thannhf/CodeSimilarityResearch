
import os
from types import SimpleNamespace

import pytest

import src.safety.persistence_temp as persistence_temp
from src.safety.persistence_temp import (
    OwnershipVerificationState,
    TempSyncError,
    create_owned_temp_file,
    sync_owned_temp_file,
    write_owned_temp_bytes,
)


def cleanup_temp(temp):
    try:
        os.close(temp.fd)
    except OSError:
        pass

    try:
        temp.path.unlink(missing_ok=True)
    except OSError:
        pass


@pytest.mark.parametrize(
    "failure",
    [
        OSError("generic fsync failure"),
        PermissionError("permission denied"),
        InterruptedError("interrupted fsync"),
        OSError(5, "I/O error"),
    ],
    ids=[
        "generic-oserror",
        "permission-error",
        "interrupted-error",
        "io-error",
    ],
)
def test_fsync_failure_is_wrapped_without_retry(
    tmp_path,
    monkeypatch,
    failure,
):
    temp = create_owned_temp_file(tmp_path / "destination.bin")
    write_owned_temp_bytes(temp, b"sync evidence")
    calls = []

    def injected_fsync(fd):
        calls.append(fd)
        raise failure

    monkeypatch.setattr(os, "fsync", injected_fsync)

    try:
        with pytest.raises(TempSyncError) as exc_info:
            sync_owned_temp_file(temp)

        assert calls == [temp.fd]
        assert exc_info.value.__cause__ is failure
    finally:
        cleanup_temp(temp)


@pytest.mark.parametrize(
    "verification_state",
    [
        OwnershipVerificationState.MISSING,
        OwnershipVerificationState.MISMATCH,
        OwnershipVerificationState.UNVERIFIABLE,
    ],
    ids=[
        "missing",
        "mismatch",
        "unverifiable",
    ],
)
def test_sync_fails_closed_when_ownership_is_not_verified(
    tmp_path,
    monkeypatch,
    verification_state,
):
    temp = create_owned_temp_file(tmp_path / "destination.bin")
    fsync_calls = []

    fake_verification = SimpleNamespace(state=verification_state)

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

        # Ownership không VERIFIED thì tuyệt đối không gọi fsync.
        assert fsync_calls == []
    finally:
        cleanup_temp(temp)


def test_sync_failure_preserves_temp_destination_and_fd(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "destination.bin"
    original_destination = b"original destination data"
    destination.write_bytes(original_destination)

    temp = create_owned_temp_file(destination)
    payload = b"new temp payload"
    write_owned_temp_bytes(temp, payload)

    fsync_calls = []

    def injected_fsync(fd):
        fsync_calls.append(fd)
        raise OSError("injected sync failure")

    monkeypatch.setattr(os, "fsync", injected_fsync)

    try:
        with pytest.raises(TempSyncError):
            sync_owned_temp_file(temp)

        # FI-09: temp không bị cleanup.
        assert temp.path.exists()
        assert temp.path.read_bytes() == payload

        # FI-10: destination không bị thay đổi.
        assert destination.read_bytes() == original_destination

        # FI-11: retained FD vẫn còn mở.
        assert os.fstat(temp.fd).st_ino == temp.identity[1]

        # Một lần gọi duy nhất, không retry.
        assert fsync_calls == [temp.fd]
    finally:
        cleanup_temp(temp)
