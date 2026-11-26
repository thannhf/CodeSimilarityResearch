
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.safety.persistence_temp as persistence_temp

from src.safety.persistence_temp import (
    OwnershipVerificationState,
    PreCommitDenialReason,
    PreCommitGateState,
    TempFileOwnership,
    create_owned_temp_file,
    evaluate_pre_commit_gate,
)


def release_temp(temp):
    try:
        os.close(temp.fd)
    except OSError:
        pass

    try:
        temp.path.unlink(missing_ok=True)
    except OSError:
        pass


def evaluate_valid(temp, **overrides):
    args = {
        "ownership": temp,
        "expected_bytes": 10,
        "bytes_written": 10,
        "sync_completed": True,
    }
    args.update(overrides)
    return evaluate_pre_commit_gate(**args)


def prepare_gate_evidence(temp, payload=b"x" * 10, *, sync=True):
    persistence_temp.write_owned_temp_bytes(temp, payload)
    if sync:
        persistence_temp.sync_owned_temp_file(temp)


def test_gate_passes_with_verified_ownership_and_complete_write(
    tmp_path,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")

    try:
        prepare_gate_evidence(temp)
        result = evaluate_valid(temp)

        assert result.state is PreCommitGateState.PASSED
        assert result.reason is None
        assert result.allowed is True
    finally:
        release_temp(temp)


@pytest.mark.parametrize(
    "verification_state",
    [
        OwnershipVerificationState.MISSING,
        OwnershipVerificationState.MISMATCH,
        OwnershipVerificationState.UNVERIFIABLE,
    ],
)
def test_gate_denies_unverified_ownership(
    tmp_path,
    monkeypatch,
    verification_state,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")

    monkeypatch.setattr(
        persistence_temp,
        "verify_temp_ownership",
        lambda ownership: SimpleNamespace(
            state=verification_state
        ),
    )

    try:
        result = evaluate_valid(temp)

        assert result.state is PreCommitGateState.DENIED
        assert (
            result.reason
            is PreCommitDenialReason.OWNERSHIP_NOT_VERIFIED
        )
        assert result.allowed is False
    finally:
        release_temp(temp)


@pytest.mark.parametrize(
    ("expected_bytes", "bytes_written"),
    [
        (10, 9),
        (10, 0),
        (10, 11),
    ],
)
def test_gate_denies_incomplete_or_excess_write(
    tmp_path,
    expected_bytes,
    bytes_written,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")

    try:
        result = evaluate_valid(
            temp,
            expected_bytes=expected_bytes,
            bytes_written=bytes_written,
        )

        assert result.state is PreCommitGateState.DENIED
        assert result.reason is PreCommitDenialReason.WRITE_NOT_COMPLETED
    finally:
        release_temp(temp)


def test_gate_denies_sync_not_completed(tmp_path):
    temp = create_owned_temp_file(tmp_path / "output.bin")

    try:
        prepare_gate_evidence(temp, sync=False)
        result = evaluate_valid(
            temp,
            sync_completed=False,
        )

        assert result.state is PreCommitGateState.DENIED
        assert result.reason is PreCommitDenialReason.SYNC_NOT_COMPLETED
    finally:
        release_temp(temp)


def test_gate_passes_empty_payload_when_sync_completed(tmp_path):
    temp = create_owned_temp_file(tmp_path / "output.bin")

    try:
        prepare_gate_evidence(temp, b"")
        result = evaluate_valid(
            temp,
            expected_bytes=0,
            bytes_written=0,
            sync_completed=True,
        )

        assert result.state is PreCommitGateState.PASSED
        assert result.allowed is True
    finally:
        release_temp(temp)


@pytest.mark.parametrize(
    ("expected_bytes", "bytes_written"),
    [
        (-1, 0),
        (0, -1),
    ],
)
def test_gate_rejects_negative_byte_counts(
    tmp_path,
    expected_bytes,
    bytes_written,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")

    try:
        with pytest.raises(ValueError):
            evaluate_valid(
                temp,
                expected_bytes=expected_bytes,
                bytes_written=bytes_written,
            )
    finally:
        release_temp(temp)


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_bytes", True),
        ("expected_bytes", 1.5),
        ("expected_bytes", "10"),
        ("bytes_written", True),
        ("bytes_written", 1.5),
        ("bytes_written", "10"),
        ("sync_completed", 1),
        ("sync_completed", None),
    ],
)
def test_gate_rejects_invalid_input_types(
    tmp_path,
    field,
    value,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")

    try:
        with pytest.raises(TypeError):
            evaluate_valid(temp, **{field: value})
    finally:
        release_temp(temp)


def test_gate_rejects_invalid_ownership_type():
    with pytest.raises(TypeError):
        evaluate_pre_commit_gate(
            ownership="not ownership",
            expected_bytes=0,
            bytes_written=0,
            sync_completed=True,
        )


def test_gate_does_not_call_fsync_or_write(
    tmp_path,
    monkeypatch,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    prepare_gate_evidence(temp)
    fsync_calls = []
    write_calls = []

    monkeypatch.setattr(
        os,
        "fsync",
        lambda fd: fsync_calls.append(fd),
    )
    monkeypatch.setattr(
        os,
        "write",
        lambda fd, data: write_calls.append((fd, data)),
    )

    try:
        result = evaluate_valid(temp)

        assert result.allowed is True
        assert fsync_calls == []
        assert write_calls == []
    finally:
        release_temp(temp)


def test_gate_does_not_modify_temp_or_destination(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "destination.bin"
    destination.write_bytes(b"original destination")

    temp = create_owned_temp_file(destination)
    payload = b"temp payload"
    persistence_temp.write_owned_temp_bytes(temp, payload)
    persistence_temp.sync_owned_temp_file(temp)

    original_temp_bytes = temp.path.read_bytes()
    original_temp_size = os.fstat(temp.fd).st_size

    monkeypatch.setattr(os, "fsync", lambda fd: None)

    try:
        result = evaluate_valid(
            temp,
            expected_bytes=len(payload),
            bytes_written=len(payload),
        )

        assert result.allowed is True
        assert temp.path.exists()
        assert temp.path.read_bytes() == original_temp_bytes
        assert os.fstat(temp.fd).st_size == original_temp_size
        assert destination.read_bytes() == b"original destination"
    finally:
        release_temp(temp)


def test_gate_does_not_cleanup_temp(tmp_path):
    temp = create_owned_temp_file(tmp_path / "output.bin")

    try:
        prepare_gate_evidence(temp)
        evaluate_valid(temp)

        assert temp.path.exists()
        assert os.fstat(temp.fd)
    finally:
        release_temp(temp)


def test_gate_ownership_failure_has_precedence(
    tmp_path,
    monkeypatch,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")

    monkeypatch.setattr(
        persistence_temp,
        "verify_temp_ownership",
        lambda ownership: SimpleNamespace(
            state=OwnershipVerificationState.MISMATCH
        ),
    )

    try:
        result = evaluate_valid(
            temp,
            expected_bytes=10,
            bytes_written=1,
            sync_completed=False,
        )

        assert result.state is PreCommitGateState.DENIED
        assert (
            result.reason
            is PreCommitDenialReason.OWNERSHIP_NOT_VERIFIED
        )
    finally:
        release_temp(temp)


def test_gate_does_not_commit_destination(tmp_path):
    destination = tmp_path / "destination.bin"
    temp = create_owned_temp_file(destination)

    try:
        prepare_gate_evidence(temp)
        result = evaluate_valid(temp)

        assert result.allowed is True
        assert not destination.exists()
    finally:
        release_temp(temp)


def test_gate_reverifies_ownership_on_each_evaluation(
    tmp_path,
    monkeypatch,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    prepare_gate_evidence(temp)
    calls = []

    real_verifier = persistence_temp.verify_temp_ownership

    def tracked_verifier(ownership):
        calls.append(ownership)
        return real_verifier(ownership)

    monkeypatch.setattr(
        persistence_temp,
        "verify_temp_ownership",
        tracked_verifier,
    )

    try:
        first = evaluate_valid(temp)
        second = evaluate_valid(temp)

        assert first.allowed is True
        assert second.allowed is False
        assert len(calls) == 2
        assert calls[0] is temp
        assert calls[1] is temp
    finally:
        release_temp(temp)


def test_gate_propagates_verifier_exception_without_side_effects(
    tmp_path,
    monkeypatch,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    injected_error = RuntimeError("injected verifier failure")
    events = []

    def failing_verifier(ownership):
        events.append("verify")
        raise injected_error

    monkeypatch.setattr(
        persistence_temp,
        "verify_temp_ownership",
        failing_verifier,
    )
    monkeypatch.setattr(os, "fsync", lambda fd: events.append("fsync"))
    monkeypatch.setattr(os, "write", lambda fd, data: events.append("write"))

    try:
        with pytest.raises(RuntimeError) as exc_info:
            evaluate_valid(temp)

        assert exc_info.value is injected_error
        assert events == ["verify"]
    finally:
        release_temp(temp)


def test_gate_write_failure_has_precedence_over_sync_failure(
    tmp_path,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")

    try:
        result = evaluate_valid(
            temp,
            expected_bytes=10,
            bytes_written=9,
            sync_completed=False,
        )

        assert result.state is PreCommitGateState.DENIED
        assert result.reason is PreCommitDenialReason.WRITE_NOT_COMPLETED
    finally:
        release_temp(temp)


def test_gate_denial_does_not_call_commit_primitives(
    tmp_path,
    monkeypatch,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    events = []

    monkeypatch.setattr(
        os,
        "fsync",
        lambda fd: events.append("fsync"),
    )
    monkeypatch.setattr(
        os,
        "write",
        lambda fd, data: events.append("write"),
    )
    monkeypatch.setattr(
        os,
        "replace",
        lambda *args, **kwargs: events.append("replace"),
    )
    monkeypatch.setattr(
        os,
        "link",
        lambda *args, **kwargs: events.append("link"),
    )

    try:
        result = evaluate_valid(
            temp,
            bytes_written=5,
            sync_completed=False,
        )

        assert result.allowed is False
        assert events == []
    finally:
        release_temp(temp)


def test_gate_result_is_immutable(tmp_path):
    temp = create_owned_temp_file(tmp_path / "output.bin")

    try:
        result = evaluate_valid(temp)

        with pytest.raises(AttributeError):
            result.state = PreCommitGateState.DENIED
    finally:
        release_temp(temp)


def test_gate_verifies_ownership_once_per_invocation(
    tmp_path,
    monkeypatch,
):
    temp = create_owned_temp_file(tmp_path / "output.bin")
    prepare_gate_evidence(temp)
    verification_calls = []

    real_verifier = persistence_temp.verify_temp_ownership

    def tracked_verifier(ownership):
        verification_calls.append(ownership)
        return real_verifier(ownership)

    monkeypatch.setattr(
        persistence_temp,
        "verify_temp_ownership",
        tracked_verifier,
    )

    try:
        evaluate_valid(temp)

        assert verification_calls == [temp]
    finally:
        release_temp(temp)
