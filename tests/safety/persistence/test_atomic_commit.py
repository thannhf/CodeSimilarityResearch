import os
from pathlib import Path
import pytest
from src.safety.persistence_temp import (
    AtomicCommitMode,
    AtomicCommitOutcome,
    PreCommitGateState,
    commit_owned_temp_file,
    create_owned_temp_file,
    evaluate_pre_commit_gate,
    sync_owned_temp_file,
    write_owned_temp_bytes,
    PreCommitDenialReason,
    TempFileOwnership,
    PreCommitGateResult,
)
from src.safety.persistence_temp import (
    verify_temp_ownership,
    OwnershipVerificationState
)
import threading
from concurrent.futures import ThreadPoolExecutor

def test_fail_if_exists_commit_succeeds(tmp_path):
    destination = tmp_path / "artifact.bin"
    payload = b"atomic-commit-test"

    ownership = create_owned_temp_file(destination)

    try:
        bytes_written = write_owned_temp_bytes(ownership, payload)
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.COMMITTED
        assert destination.exists()
        assert destination.read_bytes() == payload

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_replace_commit_succeeds_and_replaces_existing_content(tmp_path):
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"old-content")

    payload = b"new-atomic-content"
    ownership = create_owned_temp_file(destination)

    try:
        bytes_written = write_owned_temp_bytes(ownership, payload)
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.REPLACE,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.COMMITTED
        assert destination.read_bytes() == b"new-atomic-content"
        assert not ownership.path.exists()

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)



def test_fail_if_exists_commit_does_not_overwrite_existing_destination(
    tmp_path,
):
    destination = tmp_path / "artifact.bin"
    old_payload = b"old-content"
    new_payload = b"new-atomic-content"

    # Pre-existing destination.
    destination.write_bytes(old_payload)

    ownership = create_owned_temp_file(destination)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            new_payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(new_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        # Existing destination must not be overwritten.
        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert destination.exists()
        assert destination.read_bytes() == old_payload

        # Temp file must remain available after rejection.
        assert ownership.path.exists()
        assert ownership.path.read_bytes() == new_payload

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_commit_is_blocked_when_pre_commit_gate_is_denied(
    tmp_path,
    monkeypatch,
):
    from src.safety.persistence_temp import (
        PreCommitGateResult,
        PreCommitGateState,
        PreCommitDenialReason,
    )

    destination = tmp_path / "artifact.bin"
    payload = b"gate-denied-payload"

    ownership = create_owned_temp_file(destination)

    mutation_calls = []

    def forbidden_link(*args, **kwargs):
        mutation_calls.append("link")
        raise AssertionError("os.link must not be called when gate is denied")

    def forbidden_replace(*args, **kwargs):
        mutation_calls.append("replace")
        raise AssertionError("os.replace must not be called when gate is denied")

    monkeypatch.setattr(os, "link", forbidden_link)
    monkeypatch.setattr(os, "replace", forbidden_replace)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            payload,
        )
        sync_owned_temp_file(ownership)

        gate = PreCommitGateResult(
            state=PreCommitGateState.DENIED,
            reason=PreCommitDenialReason.SYNC_NOT_COMPLETED,
        )

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        # Denied gate must reject the commit.
        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED

        # No destination mutation.
        assert not destination.exists()

        # Temp file must remain intact.
        assert ownership.path.exists()
        assert ownership.path.read_bytes() == payload

        # Neither atomic mutation operation may be invoked.
        assert mutation_calls == []

    finally:
        monkeypatch.undo()
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_commit_is_blocked_when_temp_ownership_mismatches(
    tmp_path,
):
    import os

    destination = tmp_path / "artifact.bin"
    payload = b"original-owned-payload"
    replacement_payload = b"replacement-unowned-payload"

    ownership = create_owned_temp_file(destination)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        # Replace the temp pathname with an unowned file.
        ownership.path.unlink()
        ownership.path.write_bytes(replacement_payload)

        verification = verify_temp_ownership(ownership)

        assert verification.state is OwnershipVerificationState.MISMATCH

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        # Ownership mismatch must block commit.
        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED

        # Destination must not be created.
        assert not destination.exists()

        # The replacement file must remain untouched.
        assert ownership.path.exists()
        assert ownership.path.read_bytes() == replacement_payload

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_commit_is_blocked_when_temp_path_is_missing(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "artifact.bin"
    payload = b"temp-missing-payload"

    ownership = create_owned_temp_file(destination)

    mutation_calls = []

    def forbidden_link(*args, **kwargs):
        mutation_calls.append("link")
        raise AssertionError("os.link must not be called")

    def forbidden_replace(*args, **kwargs):
        mutation_calls.append("replace")
        raise AssertionError("os.replace must not be called")

    monkeypatch.setattr(os, "link", forbidden_link)
    monkeypatch.setattr(os, "replace", forbidden_replace)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        # Simulate the temp pathname disappearing after the gate.
        ownership.path.unlink()

        verification = verify_temp_ownership(ownership)

        assert verification.state is OwnershipVerificationState.MISSING

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()
        assert not ownership.path.exists()
        assert mutation_calls == []

    finally:
        monkeypatch.undo()
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_commit_is_blocked_when_ownership_is_unverifiable(
    tmp_path,
    monkeypatch,
):
    import src.safety.persistence_temp as temp_module

    destination = tmp_path / "artifact.bin"
    payload = b"unverifiable-ownership-payload"

    ownership = create_owned_temp_file(destination)

    mutation_calls = []

    def forbidden_link(*args, **kwargs):
        mutation_calls.append("link")
        raise AssertionError("os.link must not be called")

    def forbidden_replace(*args, **kwargs):
        mutation_calls.append("replace")
        raise AssertionError("os.replace must not be called")

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        # Inject fstat failure only during commit-time verification.
        monkeypatch.setattr(
            temp_module.os,
            "fstat",
            lambda fd: (_ for _ in ()).throw(
                OSError("simulated descriptor verification failure")
            ),
        )
        monkeypatch.setattr(os, "link", forbidden_link)
        monkeypatch.setattr(os, "replace", forbidden_replace)

        verification = verify_temp_ownership(ownership)

        assert verification.state is OwnershipVerificationState.UNVERIFIABLE

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()

        assert ownership.path.exists()
        assert ownership.path.read_bytes() == payload

        assert mutation_calls == []

    finally:
        monkeypatch.undo()
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_atomic_mutation_oserror_returns_outcome_unknown(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "artifact.bin"
    payload = b"atomic-mutation-error-payload"

    ownership = create_owned_temp_file(destination)

    mutation_calls = []

    def failing_link(src, dst, *args, **kwargs):
        mutation_calls.append((src, dst))
        raise PermissionError("simulated atomic link failure")

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        monkeypatch.setattr(os, "link", failing_link)

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        # An OSError during atomic mutation must not be
        # reported as definite success or definite rejection.
        assert result.outcome is AtomicCommitOutcome.OUTCOME_UNKNOWN

        # The original error must be preserved in the result.
        assert isinstance(result.error, PermissionError)

        # No retry: exactly one mutation attempt.
        assert len(mutation_calls) == 1

        # In this simulated failure, destination was not created.
        assert not destination.exists()

        # Temp remains available; no automatic cleanup.
        assert ownership.path.exists()
        assert ownership.path.read_bytes() == payload

    finally:
        monkeypatch.undo()
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_atomic_mutation_filenotfound_returns_not_committed(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "artifact.bin"
    payload = b"atomic-filenotfound-payload"

    ownership = create_owned_temp_file(destination)

    mutation_calls = []

    def failing_link(src, dst, *args, **kwargs):
        mutation_calls.append((src, dst))
        raise FileNotFoundError("simulated atomic link source missing")

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        monkeypatch.setattr(os, "link", failing_link)

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        # FileNotFoundError is classified as a definite rejection.
        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED

        # No retry.
        assert len(mutation_calls) == 1

        # Destination remains absent.
        assert not destination.exists()

        # Temp is preserved for explicit recovery/cleanup.
        assert ownership.path.exists()
        assert ownership.path.read_bytes() == payload

    finally:
        monkeypatch.undo()
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_atomic_mutation_fileexists_returns_not_committed(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "artifact.bin"
    old_payload = b"existing-destination-content"
    new_payload = b"atomic-fileexists-payload"

    destination.write_bytes(old_payload)
    ownership = create_owned_temp_file(destination)

    mutation_calls = []

    def failing_link(src, dst, *args, **kwargs):
        mutation_calls.append((src, dst))
        raise FileExistsError("simulated destination collision")

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            new_payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(new_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        monkeypatch.setattr(os, "link", failing_link)

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        # FileExistsError must be classified specifically.
        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED

        # No retry.
        assert len(mutation_calls) == 1

        # Existing destination must remain untouched.
        assert destination.exists()
        assert destination.read_bytes() == old_payload

        # Temp remains intact.
        assert ownership.path.exists()
        assert ownership.path.read_bytes() == new_payload

    finally:
        monkeypatch.undo()
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_fail_if_exists_handles_destination_race_at_atomic_boundary(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "artifact.bin"
    temp_payload = b"owned-temp-payload"
    racing_payload = b"racing-writer-payload"

    ownership = create_owned_temp_file(destination)

    real_link = os.link
    mutation_calls = []

    def racing_link(src, dst, *args, **kwargs):
        mutation_calls.append((src, dst))

        # Simulate another actor winning the race
        # immediately before the real atomic link.
        Path(dst).write_bytes(racing_payload)

        # The real link must now fail with FileExistsError.
        return real_link(src, dst, *args, **kwargs)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            temp_payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(temp_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED
        assert not destination.exists()

        monkeypatch.setattr(os, "link", racing_link)

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        # Late collision must be classified as NOT_COMMITTED.
        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED

        # The racing actor's content must not be overwritten.
        assert destination.exists()
        assert destination.read_bytes() == racing_payload

        # Only one atomic mutation attempt; no retry.
        assert len(mutation_calls) == 1

        # Our owned temp file remains intact.
        assert ownership.path.exists()
        assert ownership.path.read_bytes() == temp_payload

    finally:
        monkeypatch.undo()
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_replace_handles_destination_race_at_atomic_boundary(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "artifact.bin"
    temp_payload = b"owned-temp-payload"
    racing_payload = b"racing-writer-payload"

    ownership = create_owned_temp_file(destination)

    real_replace = os.replace
    mutation_calls = []

    def racing_replace(src, dst, *args, **kwargs):
        mutation_calls.append((src, dst))

        # Simulate another actor creating the destination
        # immediately before our atomic replace.
        Path(dst).write_bytes(racing_payload)

        # REPLACE semantics intentionally replace the racing file.
        return real_replace(src, dst, *args, **kwargs)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            temp_payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(temp_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED
        assert not destination.exists()

        monkeypatch.setattr(os, "replace", racing_replace)

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.REPLACE,
            gate=gate,
        )

        # REPLACE mode should commit despite the late destination.
        assert result.outcome is AtomicCommitOutcome.COMMITTED

        # Our temp content replaces the racing writer's content.
        assert destination.exists()
        assert destination.read_bytes() == temp_payload
        assert destination.read_bytes() != racing_payload

        # The temp pathname should be consumed by replace.
        assert not ownership.path.exists()

        # Exactly one atomic mutation attempt.
        assert len(mutation_calls) == 1

    finally:
        monkeypatch.undo()
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_temp_path_substitution_after_verification_exposes_toctou(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "artifact.bin"
    owned_payload = b"original-owned-payload"
    substituted_payload = b"substituted-unowned-payload"

    ownership = create_owned_temp_file(destination)
    real_link = os.link
    mutation_calls = []

    def substitute_then_link(src, dst, *args, **kwargs):
        mutation_calls.append((src, dst))

        # Ownership has already been verified by commit.
        # Substitute the pathname immediately before the
        # actual atomic link operation.
        Path(src).unlink()
        Path(src).write_bytes(substituted_payload)

        return real_link(src, dst, *args, **kwargs)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            owned_payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(owned_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        monkeypatch.setattr(os, "link", substitute_then_link)

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        # Characterize the currently known TOCTOU limitation:
        # verification succeeded before pathname substitution.
        assert result.outcome is AtomicCommitOutcome.COMMITTED

        assert destination.exists()
        assert destination.read_bytes() == substituted_payload
        assert destination.read_bytes() != owned_payload

        assert len(mutation_calls) == 1

    finally:
        monkeypatch.undo()
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)

def test_atomic_mutation_succeeds_then_raises_outcome_unknown(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "artifact.bin"
    payload = b"mutation-completed-before-error"

    ownership = create_owned_temp_file(destination)

    real_link = os.link
    mutation_calls = []

    def link_then_raise(src, dst, *args, **kwargs):
        mutation_calls.append((src, dst))

        # Perform the real atomic mutation first.
        real_link(src, dst, *args, **kwargs)

        # Simulate an error reported after mutation occurred.
        raise PermissionError("simulated post-mutation error")

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        monkeypatch.setattr(os, "link", link_then_raise)

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        # The API cannot claim certainty after the injected error.
        assert result.outcome is AtomicCommitOutcome.OUTCOME_UNKNOWN
        assert isinstance(result.error, PermissionError)

        # Mutation actually happened before the error.
        assert destination.exists()
        assert destination.read_bytes() == payload

        # No retry.
        assert len(mutation_calls) == 1

        # Hard-link commit leaves the owned temp path present.
        assert ownership.path.exists()
        assert ownership.path.read_bytes() == payload

    finally:
        monkeypatch.undo()
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_temp_path_symlink_substitution_at_mutation_boundary(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "artifact.bin"
    attacker_target = tmp_path / "attacker.bin"

    owned_payload = b"original-owned-payload"
    attacker_payload = b"attacker-controlled-payload"

    attacker_target.write_bytes(attacker_payload)
    ownership = create_owned_temp_file(destination)

    real_link = os.link
    mutation_calls = []

    def substitute_with_symlink_then_link(src, dst, *args, **kwargs):
        mutation_calls.append((src, dst))

        Path(src).unlink()
        try:
            Path(src).symlink_to(attacker_target)
        except OSError as exc:
            pytest.skip(f"Symlink creation unavailable on this platform: {exc}")

        return real_link(src, dst, *args, **kwargs)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            owned_payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(owned_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        monkeypatch.setattr(os, "link", substitute_with_symlink_then_link)

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        # Characterize behavior when temp pathname becomes a symlink.
        assert result.outcome is AtomicCommitOutcome.COMMITTED
        assert destination.exists()
        assert destination.read_bytes() == attacker_payload
        assert destination.read_bytes() != owned_payload
        assert len(mutation_calls) == 1

    finally:
        monkeypatch.undo()
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_temp_path_hardlink_substitution_at_mutation_boundary(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "artifact.bin"
    attacker_target = tmp_path / "attacker.bin"

    owned_payload = b"original-owned-payload"
    attacker_payload = b"attacker-controlled-payload"

    attacker_target.write_bytes(attacker_payload)
    ownership = create_owned_temp_file(destination)

    real_link = os.link
    mutation_calls = []

    def substitute_with_hardlink_then_link(src, dst, *args, **kwargs):
        mutation_calls.append((src, dst))

        Path(src).unlink()
        real_link(attacker_target, src)

        return real_link(src, dst, *args, **kwargs)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            owned_payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(owned_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        monkeypatch.setattr(
            os,
            "link",
            substitute_with_hardlink_then_link,
        )

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        # Characterize the hard-link pathname substitution exposure.
        assert result.outcome is AtomicCommitOutcome.COMMITTED
        assert destination.exists()
        assert destination.read_bytes() == attacker_payload
        assert destination.read_bytes() != owned_payload
        assert len(mutation_calls) == 1

    finally:
        monkeypatch.undo()
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_temp_path_hardlink_substitution_before_verification_is_blocked(
    tmp_path,
):
    destination = tmp_path / "artifact.bin"
    attacker_target = tmp_path / "attacker.bin"

    owned_payload = b"original-owned-payload"
    attacker_payload = b"attacker-controlled-payload"

    attacker_target.write_bytes(attacker_payload)
    ownership = create_owned_temp_file(destination)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            owned_payload,
        )
        sync_owned_temp_file(ownership)

        # Substitute the temp pathname BEFORE ownership verification.
        ownership.path.unlink()
        os.link(attacker_target, ownership.path)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(owned_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        # Ownership verification should detect the substitution.
        assert gate.state is not PreCommitGateState.PASSED

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()
        assert ownership.path.exists()
        assert ownership.path.read_bytes() == attacker_payload

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_temp_content_mutation_after_gate_before_commit(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "artifact.bin"

    owned_payload = b"original-owned-payload"
    mutated_payload = b"X" * len(owned_payload)

    ownership = create_owned_temp_file(destination)
    real_link = os.link
    mutation_calls = []

    def mutate_content_then_link(src, dst, *args, **kwargs):
        mutation_calls.append((src, dst))

        # Preserve pathname and file identity, but mutate content.
        Path(src).write_bytes(mutated_payload)

        return real_link(src, dst, *args, **kwargs)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            owned_payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(owned_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        monkeypatch.setattr(os, "link", mutate_content_then_link)

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        # Characterize content-integrity exposure at mutation boundary.
        assert result.outcome is AtomicCommitOutcome.COMMITTED
        assert destination.exists()
        assert destination.read_bytes() == mutated_payload
        assert destination.read_bytes() != owned_payload
        assert len(mutation_calls) == 1

    finally:
        monkeypatch.undo()
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_temp_content_mutation_after_sync_before_gate(
    tmp_path,
):
    destination = tmp_path / "artifact.bin"

    owned_payload = b"original-owned-payload"
    mutated_payload = b"modified-owned-payload"

    assert len(owned_payload) == len(mutated_payload)

    ownership = create_owned_temp_file(destination)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            owned_payload,
        )
        sync_owned_temp_file(ownership)

        # Mutate content after sync, before gate evaluation.
        ownership.path.write_bytes(mutated_payload)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(owned_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        # Characterize whether the gate accepts same-length mutation.
        assert gate.state is PreCommitGateState.PASSED

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.COMMITTED
        assert destination.read_bytes() == mutated_payload
        assert destination.read_bytes() != owned_payload

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_temp_truncation_after_sync_before_gate(
    tmp_path,
):
    destination = tmp_path / "artifact.bin"

    owned_payload = b"original-owned-payload"
    truncated_payload = owned_payload[:5]

    ownership = create_owned_temp_file(destination)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            owned_payload,
        )
        sync_owned_temp_file(ownership)

        # Truncate the temp file after sync, before gate evaluation.
        ownership.path.write_bytes(truncated_payload)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(owned_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        # Gate must reject modified temp-file content/size.
        assert gate.state is PreCommitGateState.DENIED
        assert gate.reason is PreCommitDenialReason.WRITE_NOT_COMPLETED

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)

def test_temp_extension_after_sync_before_gate(
    tmp_path,
):
    destination = tmp_path / "artifact.bin"

    owned_payload = b"original-owned-payload"
    appended_payload = b"-attacker-extra"

    ownership = create_owned_temp_file(destination)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            owned_payload,
        )
        sync_owned_temp_file(ownership)

        # Extend the file after sync, before gate evaluation.
        with ownership.path.open("ab") as stream:
            stream.write(appended_payload)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(owned_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        # Gate must reject modified temp-file content/size.
        assert gate.state is PreCommitGateState.DENIED
        assert gate.reason is PreCommitDenialReason.WRITE_NOT_COMPLETED

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)

def test_committed_temp_cannot_be_reused_for_second_commit(
    tmp_path,
):
    destination = tmp_path / "artifact.bin"
    payload = b"transaction-payload"

    ownership = create_owned_temp_file(destination)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        first_result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert first_result.outcome is AtomicCommitOutcome.COMMITTED
        assert destination.read_bytes() == payload

        # Replay the same ownership and stale gate.
        second_result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert second_result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert destination.read_bytes() == payload

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_committed_temp_replay_under_replace_does_not_mutate_destination(
    tmp_path,
):
    destination = tmp_path / "artifact.bin"
    payload = b"transaction-payload"
    replacement_payload = b"replacement-content"

    ownership = create_owned_temp_file(destination)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        first_result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.REPLACE,
            gate=gate,
        )

        assert first_result.outcome is AtomicCommitOutcome.COMMITTED
        assert destination.read_bytes() == payload

        # Change destination so replay would cause visible data loss
        # if the stale temp were accepted.
        destination.write_bytes(replacement_payload)

        second_result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.REPLACE,
            gate=gate,
        )

        assert second_result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert destination.read_bytes() == replacement_payload

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_fail_if_exists_temp_can_be_replayed_to_second_destination(
    tmp_path,
):
    destination_a = tmp_path / "artifact-a.bin"
    destination_b = tmp_path / "artifact-b.bin"
    payload = b"transaction-payload"

    ownership = create_owned_temp_file(destination_a)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        first_result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination_a,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert first_result.outcome is AtomicCommitOutcome.COMMITTED
        assert destination_a.read_bytes() == payload

        # Attempt replay to a different destination.
        second_result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination_b,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        # Ownership is consumed after the first successful commit.
        # Cross-destination replay must be rejected.
        assert second_result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination_b.exists()

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_replace_temp_cannot_be_replayed_to_second_destination(
    tmp_path,
):
    destination_a = tmp_path / "artifact-a.bin"
    destination_b = tmp_path / "artifact-b.bin"
    payload = b"transaction-payload"

    ownership = create_owned_temp_file(destination_a)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        first_result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination_a,
            mode=AtomicCommitMode.REPLACE,
            gate=gate,
        )

        assert first_result.outcome is AtomicCommitOutcome.COMMITTED
        assert destination_a.read_bytes() == payload
        assert not ownership.path.exists()

        second_result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination_b,
            mode=AtomicCommitMode.REPLACE,
            gate=gate,
        )

        assert second_result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination_b.exists()

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_stale_gate_reuse_after_temp_content_mutation(
    tmp_path,
):
    destination = tmp_path / "artifact.bin"

    original_payload = b"original-transaction"
    mutated_payload = b"changed--transaction"
    assert len(original_payload) == len(mutated_payload)

    ownership = create_owned_temp_file(destination)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            original_payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(original_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.PASSED

        # Mutate content while retaining the same temp file identity.
        ownership.path.write_bytes(mutated_payload)

        # Reuse the stale gate without re-evaluation.
        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.COMMITTED
        assert destination.read_bytes() == mutated_payload
        assert destination.read_bytes() != original_payload

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)


def test_gate_passed_for_temp_a_can_be_reused_with_temp_b(
    tmp_path,
):
    destination_a = tmp_path / "artifact-a.bin"
    destination_b = tmp_path / "artifact-b.bin"

    payload_a = b"payload-from-transaction-a"
    payload_b = b"payload-from-transaction-b"

    ownership_a = create_owned_temp_file(destination_a)
    ownership_b = create_owned_temp_file(destination_b)

    try:
        bytes_written_a = write_owned_temp_bytes(
            ownership_a,
            payload_a,
        )
        sync_owned_temp_file(ownership_a)

        gate_a = evaluate_pre_commit_gate(
            ownership=ownership_a,
            expected_bytes=len(payload_a),
            bytes_written=bytes_written_a,
            sync_completed=True,
        )

        assert gate_a.state is PreCommitGateState.PASSED

        write_owned_temp_bytes(
            ownership_b,
            payload_b,
        )
        sync_owned_temp_file(ownership_b)

        # Transaction A's gate must not authorize transaction B.
        result_b = commit_owned_temp_file(
            ownership=ownership_b,
            destination=destination_b,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate_a,
        )

        assert result_b.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination_b.exists()

    finally:
        os.close(ownership_a.fd)
        os.close(ownership_b.fd)
        ownership_a.path.unlink(missing_ok=True)
        ownership_b.path.unlink(missing_ok=True)


def test_gate_passed_for_temp_a_can_authorize_temp_b_with_different_size(
    tmp_path,
):
    destination_a = tmp_path / "artifact-a.bin"
    destination_b = tmp_path / "artifact-b.bin"

    payload_a = b"short-payload"
    payload_b = b"a substantially longer payload from transaction b"

    assert len(payload_a) != len(payload_b)

    ownership_a = create_owned_temp_file(destination_a)
    ownership_b = create_owned_temp_file(destination_b)

    try:
        bytes_written_a = write_owned_temp_bytes(
            ownership_a,
            payload_a,
        )
        sync_owned_temp_file(ownership_a)

        gate_a = evaluate_pre_commit_gate(
            ownership=ownership_a,
            expected_bytes=len(payload_a),
            bytes_written=bytes_written_a,
            sync_completed=True,
        )

        assert gate_a.state is PreCommitGateState.PASSED

        write_owned_temp_bytes(
            ownership_b,
            payload_b,
        )
        sync_owned_temp_file(ownership_b)

        # A gate bound to A must not authorize B, even with
        # a different payload size.
        result_b = commit_owned_temp_file(
            ownership=ownership_b,
            destination=destination_b,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate_a,
        )

        assert result_b.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination_b.exists()

    finally:
        os.close(ownership_a.fd)
        os.close(ownership_b.fd)
        ownership_a.path.unlink(missing_ok=True)
        ownership_b.path.unlink(missing_ok=True)

def test_passed_gate_from_synced_a_authorizes_unsynced_b(
    tmp_path,
):
    destination_a = tmp_path / "artifact-a.bin"
    destination_b = tmp_path / "artifact-b.bin"

    payload_a = b"synced-transaction-a"
    payload_b = b"unsynced-transaction-b"

    ownership_a = create_owned_temp_file(destination_a)
    ownership_b = create_owned_temp_file(destination_b)

    try:
        bytes_written_a = write_owned_temp_bytes(
            ownership_a,
            payload_a,
        )
        sync_owned_temp_file(ownership_a)

        gate_a = evaluate_pre_commit_gate(
            ownership=ownership_a,
            expected_bytes=len(payload_a),
            bytes_written=bytes_written_a,
            sync_completed=True,
        )

        assert gate_a.state is PreCommitGateState.PASSED

        # Write B, but deliberately do NOT sync B.
        bytes_written_b = write_owned_temp_bytes(
            ownership_b,
            payload_b,
        )
        assert bytes_written_b == len(payload_b)

        # A's authorization cannot bypass B's missing sync.
        result_b = commit_owned_temp_file(
            ownership=ownership_b,
            destination=destination_b,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate_a,
        )

        assert result_b.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination_b.exists()

    finally:
        os.close(ownership_a.fd)
        os.close(ownership_b.fd)
        ownership_a.path.unlink(missing_ok=True)
        ownership_b.path.unlink(missing_ok=True)

def test_forged_pass_gate_authorizes_unsynced_temp_commit(
    tmp_path,
):
    from src.safety.persistence_temp import PreCommitGateResult

    destination = tmp_path / "artifact.bin"
    payload = b"written-but-never-synced"

    ownership = create_owned_temp_file(destination)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            payload,
        )
        assert bytes_written == len(payload)

        # Forge a PASS gate directly. No evaluation and no sync.
        forged_gate = PreCommitGateResult(
            state=PreCommitGateState.PASSED,
            reason=None,
        )

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=forged_gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)

def test_gate_accepts_caller_claimed_sync_without_actual_sync(
    tmp_path,
):
    destination = tmp_path / "artifact.bin"
    payload = b"write-completed-but-no-fsync"

    ownership = create_owned_temp_file(destination)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            payload,
        )

        # Deliberately skip sync_owned_temp_file().
        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.DENIED
        assert gate.reason is PreCommitDenialReason.SYNC_NOT_COMPLETED

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)

def test_gate_accepts_falsely_claimed_write_completion(
    tmp_path,
):
    destination = tmp_path / "artifact.bin"

    actual_payload = b"short"
    expected_payload = b"full-payload-expected"

    ownership = create_owned_temp_file(destination)

    try:
        actual_bytes_written = write_owned_temp_bytes(
            ownership,
            actual_payload,
        )
        assert actual_bytes_written == len(actual_payload)
        assert len(actual_payload) < len(expected_payload)

        sync_owned_temp_file(ownership)

        # Forge the byte-count claim to match expected size.
        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(expected_payload),
            bytes_written=len(expected_payload),
            sync_completed=True,
        )

        assert gate.state is PreCommitGateState.DENIED
        assert gate.reason is PreCommitDenialReason.WRITE_NOT_COMPLETED

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)

def test_passed_gate_with_denial_reason_still_authorizes_commit(
    tmp_path,
):
    from src.safety.persistence_temp import PreCommitGateResult

    destination = tmp_path / "artifact.bin"
    payload = b"gate-state-reason-conflict"

    ownership = create_owned_temp_file(destination)

    try:
        write_owned_temp_bytes(
            ownership,
            payload,
        )
        sync_owned_temp_file(ownership)

        contradictory_gate = PreCommitGateResult(
            state=PreCommitGateState.PASSED,
            reason=PreCommitDenialReason.SYNC_NOT_COMPLETED,
        )

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=contradictory_gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)

def test_forged_temp_ownership_can_commit_arbitrary_file(
    tmp_path,
):
    from src.safety.persistence_temp import TempFileOwnership

    source = tmp_path / "arbitrary-source.bin"
    destination = tmp_path / "artifact.bin"

    payload = b"arbitrary-file-not-created-by-temp-helper"

    fd = os.open(
        source,
        os.O_CREAT | os.O_EXCL | os.O_RDWR,
        0o600,
    )

    try:
        os.write(fd, payload)
        file_stat = os.fstat(fd)
        identity = (file_stat.st_dev, file_stat.st_ino)

        forged_ownership = TempFileOwnership(
            transaction_id="forged-transaction",
            path=source,
            fd=fd,
            identity=identity,
            attempt=0,
        )

        gate = PreCommitGateResult(
            state=PreCommitGateState.PASSED,
            reason=None,
        )

        result = commit_owned_temp_file(
            ownership=forged_ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()

    finally:
        os.close(fd)
        source.unlink(missing_ok=True)


def test_commit_accepts_forged_ownership_metadata(
    tmp_path,
):
    from src.safety.persistence_temp import (
        TempFileOwnership,
        PreCommitGateResult,
    )

    source = tmp_path / "arbitrary-source.bin"
    destination = tmp_path / "artifact.bin"
    payload = b"invalid-ownership-metadata"

    fd = os.open(
        source,
        os.O_CREAT | os.O_EXCL | os.O_RDWR,
        0o600,
    )

    try:
        os.write(fd, payload)
        file_stat = os.fstat(fd)
        identity = (file_stat.st_dev, file_stat.st_ino)

        forged_ownership = TempFileOwnership(
            transaction_id="",
            path=source,
            fd=fd,
            identity=identity,
            attempt=-999,
        )

        forged_gate = PreCommitGateResult(
            state=PreCommitGateState.PASSED,
            reason=None,
        )

        result = commit_owned_temp_file(
            ownership=forged_ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=forged_gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()

    finally:
        os.close(fd)
        source.unlink(missing_ok=True)


def test_commit_accepts_wrong_types_in_ownership_metadata(
    tmp_path,
):
    from src.safety.persistence_temp import (
        TempFileOwnership,
        PreCommitGateResult,
    )

    source = tmp_path / "arbitrary-source.bin"
    destination = tmp_path / "artifact.bin"
    payload = b"wrong-metadata-types"

    fd = os.open(
        source,
        os.O_CREAT | os.O_EXCL | os.O_RDWR,
        0o600,
    )

    try:
        os.write(fd, payload)
        file_stat = os.fstat(fd)
        identity = (file_stat.st_dev, file_stat.st_ino)

        forged_ownership = TempFileOwnership(
            transaction_id=None,
            path=source,
            fd=fd,
            identity=identity,
            attempt="not-an-integer",
        )

        forged_gate = PreCommitGateResult(
            state=PreCommitGateState.PASSED,
            reason=None,
        )

        result = commit_owned_temp_file(
            ownership=forged_ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=forged_gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()

    finally:
        os.close(fd)
        source.unlink(missing_ok=True)


def test_forged_ownership_path_and_fd_refer_to_different_files(
    tmp_path,
):
    from src.safety.persistence_temp import (
        TempFileOwnership,
        PreCommitGateResult,
    )

    source_a = tmp_path / "source-a.bin"
    source_b = tmp_path / "source-b.bin"
    destination = tmp_path / "artifact.bin"

    payload_a = b"file-a-path"
    payload_b = b"file-b-descriptor"

    fd_a = os.open(
        source_a,
        os.O_CREAT | os.O_EXCL | os.O_RDWR,
        0o600,
    )
    fd_b = os.open(
        source_b,
        os.O_CREAT | os.O_EXCL | os.O_RDWR,
        0o600,
    )

    try:
        os.write(fd_a, payload_a)
        os.write(fd_b, payload_b)

        stat_a = os.fstat(fd_a)
        identity_a = (stat_a.st_dev, stat_a.st_ino)

        forged_ownership = TempFileOwnership(
            transaction_id="forged-split-identity",
            path=source_a,
            fd=fd_b,
            identity=identity_a,
            attempt=0,
        )

        forged_gate = PreCommitGateResult(
            state=PreCommitGateState.PASSED,
            reason=None,
        )

        result = commit_owned_temp_file(
            ownership=forged_ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=forged_gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()

    finally:
        os.close(fd_a)
        os.close(fd_b)
        source_a.unlink(missing_ok=True)
        source_b.unlink(missing_ok=True)


def test_forged_identity_tuple_blocks_commit(
    tmp_path,
):
    from src.safety.persistence_temp import (
        TempFileOwnership,
        PreCommitGateResult,
    )

    source = tmp_path / "source.bin"
    destination = tmp_path / "artifact.bin"
    payload = b"identity-tuple-forgery"

    fd = os.open(
        source,
        os.O_CREAT | os.O_EXCL | os.O_RDWR,
        0o600,
    )

    try:
        os.write(fd, payload)

        file_stat = os.fstat(fd)
        real_identity = (file_stat.st_dev, file_stat.st_ino)
        forged_identity = (
            real_identity[0],
            real_identity[1] + 1,
        )

        assert forged_identity != real_identity

        forged_ownership = TempFileOwnership(
            transaction_id="forged-identity",
            path=source,
            fd=fd,
            identity=forged_identity,
            attempt=0,
        )

        forged_gate = PreCommitGateResult(
            state=PreCommitGateState.PASSED,
            reason=None,
        )

        result = commit_owned_temp_file(
            ownership=forged_ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=forged_gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()

    finally:
        os.close(fd)
        source.unlink(missing_ok=True)


def test_invalid_descriptor_blocks_commit(
    tmp_path,
):
    from src.safety.persistence_temp import (
        TempFileOwnership,
        PreCommitGateResult,
    )

    destination = tmp_path / "artifact.bin"
    payload = b"invalid-descriptor-test"

    ownership = create_owned_temp_file(destination)

    try:
        write_owned_temp_bytes(ownership, payload)
        sync_owned_temp_file(ownership)

        forged_ownership = TempFileOwnership(
            transaction_id=ownership.transaction_id,
            path=ownership.path,
            fd=-1,
            identity=ownership.identity,
            attempt=ownership.attempt,
        )

        forged_gate = PreCommitGateResult(
            state=PreCommitGateState.PASSED,
            reason=None,
        )

        result = commit_owned_temp_file(
            ownership=forged_ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=forged_gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)

def test_closed_fd_before_commit_blocks_commit(
    tmp_path,
):
    source = tmp_path / "source.bin"
    destination = tmp_path / "artifact.bin"
    payload = b"closed-fd-before-commit"

    ownership = create_owned_temp_file(source)

    try:
        os.write(ownership.fd, payload)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=len(payload),
            sync_completed=True,
        )

        os.close(ownership.fd)

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()

    finally:
        source.unlink(missing_ok=True)

def test_closed_fd_commit_returns_controlled_outcome(
    tmp_path,
):
    source = tmp_path / "source.bin"
    destination = tmp_path / "artifact.bin"
    payload = b"closed-fd-controlled-outcome"

    ownership = create_owned_temp_file(source)

    try:
        os.write(ownership.fd, payload)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=len(payload),
            sync_completed=True,
        )

        os.close(ownership.fd)

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()

    finally:
        source.unlink(missing_ok=True)

def test_commit_destination_directory_returns_not_committed(
    tmp_path,
):
    source = tmp_path / "source.bin"
    destination = tmp_path / "artifact.bin"
    payload = b"destination-is-directory"

    destination.mkdir()

    ownership = create_owned_temp_file(source)

    try:
        os.write(ownership.fd, payload)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=len(payload),
            sync_completed=True,
        )

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert destination.is_dir()
        assert not (destination / source.name).exists()

    finally:
        os.close(ownership.fd)
        source.unlink(missing_ok=True)

def test_fail_if_exists_preserves_existing_hardlink_destination(
    tmp_path,
):
    source = tmp_path / "source.bin"
    destination = tmp_path / "artifact.bin"
    original = tmp_path / "original.bin"

    original_payload = b"hardlink-original"
    commit_payload = b"new-commit-payload"

    original.write_bytes(original_payload)

    try:
        os.link(original, destination)
    except OSError as exc:
        pytest.skip(f"Hard link unavailable: {exc}")

    ownership = create_owned_temp_file(source)

    try:
        os.write(ownership.fd, commit_payload)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(commit_payload),
            bytes_written=len(commit_payload),
            sync_completed=True,
        )

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert original.read_bytes() == original_payload
        assert destination.read_bytes() == original_payload

    finally:
        os.close(ownership.fd)
        source.unlink(missing_ok=True)

def test_replace_destination_hardlink_preserves_original_inode(
    tmp_path,
):
    source = tmp_path / "source.bin"
    destination = tmp_path / "artifact.bin"
    original = tmp_path / "original.bin"

    original_payload = b"hardlink-original"
    replacement_payload = b"replacement-content"

    original.write_bytes(original_payload)

    try:
        os.link(original, destination)
    except OSError as exc:
        pytest.skip(f"Hard link unavailable: {exc}")

    original_identity = (
        original.stat().st_dev,
        original.stat().st_ino,
    )

    ownership = create_owned_temp_file(source)

    try:
        bytes_written = write_owned_temp_bytes(ownership, replacement_payload)
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(replacement_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.REPLACE,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.COMMITTED
        assert destination.read_bytes() == replacement_payload
        assert original.read_bytes() == original_payload

        assert (
            original.stat().st_dev,
            original.stat().st_ino,
        ) == original_identity

        assert (
            destination.stat().st_dev,
            destination.stat().st_ino,
        ) != original_identity

    finally:
        os.close(ownership.fd)
        source.unlink(missing_ok=True)


def test_replace_readonly_destination_behavior(
    tmp_path,
):
    source = tmp_path / "source.bin"
    destination = tmp_path / "artifact.bin"

    old_payload = b"readonly-old"
    new_payload = b"readonly-new"

    destination.write_bytes(old_payload)

    ownership = create_owned_temp_file(source)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            new_payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(new_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        destination.chmod(0o444)

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.REPLACE,
            gate=gate,
        )

        assert result.outcome in {
            AtomicCommitOutcome.COMMITTED,
            AtomicCommitOutcome.OUTCOME_UNKNOWN,
        }

        if result.outcome is AtomicCommitOutcome.COMMITTED:
            assert destination.read_bytes() == new_payload
        else:
            assert destination.exists()

    finally:
        destination.chmod(0o666)
        os.close(ownership.fd)
        source.unlink(missing_ok=True)


def test_replace_commit_destination_exact_content(
    tmp_path,
):
    source = tmp_path / "source.bin"
    destination = tmp_path / "artifact.bin"

    old_payload = b"old-content-must-be-replaced"
    new_payload = b"new-content-exactly-once"

    destination.write_bytes(old_payload)

    ownership = create_owned_temp_file(source)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            new_payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(new_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.REPLACE,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.COMMITTED
        assert destination.read_bytes() == new_payload
        assert destination.stat().st_size == len(new_payload

        )

    finally:
        os.close(ownership.fd)
        source.unlink(missing_ok=True)

def test_replace_commit_empty_payload(
    tmp_path,
):
    source = tmp_path / "source.bin"
    destination = tmp_path / "artifact.bin"

    destination.write_bytes(b"old-content")

    ownership = create_owned_temp_file(source)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            b"",
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=0,
            bytes_written=bytes_written,
            sync_completed=True,
        )

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.REPLACE,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.COMMITTED
        assert destination.exists()
        assert destination.read_bytes() == b""
        assert destination.stat().st_size == 0

    finally:
        os.close(ownership.fd)
        source.unlink(missing_ok=True)

def test_replace_commit_large_payload_integrity(
    tmp_path,
):
    source = tmp_path / "source.bin"
    destination = tmp_path / "artifact.bin"

    old_payload = b"old"
    new_payload = b"L" * (8 * 1024 * 1024)

    destination.write_bytes(old_payload)

    ownership = create_owned_temp_file(source)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            new_payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(new_payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.REPLACE,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.COMMITTED
        assert destination.stat().st_size == len(new_payload)
        assert destination.read_bytes() == new_payload

    finally:
        os.close(ownership.fd)
        source.unlink(missing_ok=True)

def test_replace_commit_when_destination_is_temp_path(
    tmp_path,
):
    source = tmp_path / "artifact.bin"
    payload = b"alias-destination-payload"

    ownership = create_owned_temp_file(source)

    try:
        bytes_written = write_owned_temp_bytes(
            ownership,
            payload,
        )
        sync_owned_temp_file(ownership)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=ownership.path,
            mode=AtomicCommitMode.REPLACE,
            gate=gate,
        )

        assert result.outcome in {
            AtomicCommitOutcome.COMMITTED,
            AtomicCommitOutcome.OUTCOME_UNKNOWN,
        }

    finally:
        os.close(ownership.fd)
        source.unlink(missing_ok=True)

def test_commit_rejects_invalid_mode(
    tmp_path,
):
    source = tmp_path / "source.bin"
    destination = tmp_path / "artifact.bin"

    ownership = create_owned_temp_file(source)

    try:
        payload = b"invalid-mode-test"
        os.write(ownership.fd, payload)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=len(payload),
            sync_completed=True,
        )

        with pytest.raises((ValueError, TypeError)):
            commit_owned_temp_file(
                ownership=ownership,
                destination=destination,
                mode="FORCE_OVERWRITE",
                gate=gate,
            )

        assert not destination.exists()

    finally:
        os.close(ownership.fd)
        source.unlink(missing_ok=True)

def test_commit_rejects_relative_destination(
    tmp_path,
):
    source = tmp_path / "source.bin"
    payload = b"relative-destination-test"

    ownership = create_owned_temp_file(source)

    try:
        os.write(ownership.fd, payload)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=len(payload),
            sync_completed=True,
        )

        relative_destination = Path("relative_artifact.bin")

        with pytest.raises(ValueError):
            commit_owned_temp_file(
                ownership=ownership,
                destination=relative_destination,
                mode=AtomicCommitMode.REPLACE,
                gate=gate,
            )

        assert not relative_destination.exists()

    finally:
        os.close(ownership.fd)
        source.unlink(missing_ok=True)

def test_replace_commit_replaces_destination_symlink_not_target(
    tmp_path,
):
    source = tmp_path / "source.bin"
    destination = tmp_path / "artifact.bin"
    target = tmp_path / "protected.bin"

    protected_payload = b"protected-original"
    new_payload = b"replacement-payload"

    target.write_bytes(protected_payload)

    try:
        destination.symlink_to(target)

    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Symlink capability unavailable: {exc}")

    ownership = create_owned_temp_file(source)

    try:
        os.write(ownership.fd, new_payload)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(new_payload),
            bytes_written=len(new_payload),
            sync_completed=True,
        )

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.REPLACE,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.COMMITTED
        assert destination.read_bytes() == new_payload
        assert not destination.is_symlink()

        # Symlink target must remain untouched.
        assert target.read_bytes() == protected_payload

    finally:
        os.close(ownership.fd)
        source.unlink(missing_ok=True)

def test_commit_rejects_temp_path_removed_after_gate(
    tmp_path,
):
    source = tmp_path / "source.bin"
    destination = tmp_path / "artifact.bin"
    payload = b"temp-path-removed-after-gate"

    ownership = create_owned_temp_file(source)

    try:
        os.write(ownership.fd, payload)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=len(payload),
            sync_completed=True,
        )

        ownership.path.unlink()

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.REPLACE,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)

def test_commit_rejects_temp_path_substitution_after_gate(
    tmp_path,
):
    source = tmp_path / "source.bin"
    destination = tmp_path / "artifact.bin"

    original_payload = b"legitimate-original"
    attacker_payload = b"attacker-replacement"

    ownership = create_owned_temp_file(source)

    try:
        os.write(ownership.fd, original_payload)

        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(original_payload),
            bytes_written=len(original_payload),
            sync_completed=True,
        )

        ownership.path.unlink()
        ownership.path.write_bytes(attacker_payload)

        result = commit_owned_temp_file(
            ownership=ownership,
            destination=destination,
            mode=AtomicCommitMode.REPLACE,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert not destination.exists()
        assert ownership.path.read_bytes() == attacker_payload

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)

def test_r1_registry_rejects_forged_temp_ownership(
    tmp_path,
):
    from src.safety.persistence_temp import TempFileOwnership

    source = tmp_path / "forged-source.bin"
    destination = tmp_path / "artifact.bin"
    payload = b"forged-payload"

    fd = os.open(
        source,
        os.O_CREAT | os.O_EXCL | os.O_RDWR,
        0o600,
    )

    try:
        os.write(fd, payload)
        file_stat = os.fstat(fd)
        identity = (file_stat.st_dev, file_stat.st_ino)

        forged_ownership = TempFileOwnership(
            transaction_id="forged-r1-transaction",
            path=source,
            fd=fd,
            identity=identity,
            attempt=0,
        )

        gate = PreCommitGateResult(
            state=PreCommitGateState.PASSED,
            reason=None,
        )

        result = commit_owned_temp_file(
            ownership=forged_ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert destination.exists() is False
        assert source.read_bytes() == payload

    finally:
        os.close(fd)

def test_r1_registry_rejects_cloned_genuine_ownership(
    tmp_path,
):
    from src.safety.persistence_temp import TempFileOwnership

    destination = tmp_path / "artifact.bin"
    payload = b"genuine-ownership-clone-test"

    genuine = create_owned_temp_file(destination)

    try:
        os.write(genuine.fd, payload)

        cloned_ownership = TempFileOwnership(
            transaction_id=genuine.transaction_id,
            path=genuine.path,
            fd=genuine.fd,
            identity=genuine.identity,
            attempt=genuine.attempt,
        )

        gate = PreCommitGateResult(
            state=PreCommitGateState.PASSED,
            reason=None,
        )

        result = commit_owned_temp_file(
            ownership=cloned_ownership,
            destination=destination,
            mode=AtomicCommitMode.FAIL_IF_EXISTS,
            gate=gate,
        )

        assert result.outcome is AtomicCommitOutcome.NOT_COMMITTED
        assert destination.exists() is False
        assert genuine.path.read_bytes() == payload

    finally:
        os.close(genuine.fd)

def test_r1_registry_rejects_replay_after_successful_commit(
    tmp_path,
):
    from src.safety.persistence_temp import (
        TempOwnershipLifecycle,
        _TEMP_OWNERSHIP_REGISTRY,
    )

    destination = tmp_path / "artifact.bin"

    ownership = create_owned_temp_file(destination)
    payload = b"replay-protection"

    write_owned_temp_bytes(ownership, payload)
    sync_owned_temp_file(ownership)

    gate = evaluate_pre_commit_gate(
        ownership=ownership,
        expected_bytes=len(payload),
        bytes_written=len(payload),
        sync_completed=True,
    )

    first_result = commit_owned_temp_file(
        ownership=ownership,
        destination=destination,
        mode=AtomicCommitMode.FAIL_IF_EXISTS,
        gate=gate,
    )

    assert first_result.outcome is AtomicCommitOutcome.COMMITTED
    assert destination.read_bytes() == payload

    record = _TEMP_OWNERSHIP_REGISTRY[
        ownership.transaction_id
    ]
    assert record.lifecycle is TempOwnershipLifecycle.CONSUMED

    second_destination = tmp_path / "replay.bin"

    second_result = commit_owned_temp_file(
        ownership=ownership,
        destination=second_destination,
        mode=AtomicCommitMode.FAIL_IF_EXISTS,
        gate=gate,
    )

    assert second_result.outcome is AtomicCommitOutcome.NOT_COMMITTED
    assert not second_destination.exists()
    assert destination.read_bytes() == payload

def test_r1_registry_rejects_replay_replace_after_successful_commit(
    tmp_path,
):
    destination = tmp_path / "artifact.bin"

    ownership = create_owned_temp_file(destination)
    payload = b"original-committed-content"

    write_owned_temp_bytes(ownership, payload)
    sync_owned_temp_file(ownership)

    gate = evaluate_pre_commit_gate(
        ownership=ownership,
        expected_bytes=len(payload),
        bytes_written=len(payload),
        sync_completed=True,
    )

    first_result = commit_owned_temp_file(
        ownership=ownership,
        destination=destination,
        mode=AtomicCommitMode.REPLACE,
        gate=gate,
    )

    assert first_result.outcome is AtomicCommitOutcome.COMMITTED
    assert destination.read_bytes() == payload

    second_destination = tmp_path / "replay.bin"
    second_destination.write_bytes(b"protected-content")

    second_result = commit_owned_temp_file(
        ownership=ownership,
        destination=second_destination,
        mode=AtomicCommitMode.REPLACE,
        gate=gate,
    )

    assert second_result.outcome is AtomicCommitOutcome.NOT_COMMITTED
    assert second_destination.read_bytes() == b"protected-content"
    assert destination.read_bytes() == payload

def test_r1_concurrent_commit_same_ownership(
    tmp_path,
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from src.safety.persistence_temp import (
        AtomicCommitMode,
        AtomicCommitOutcome,
        commit_owned_temp_file,
        create_owned_temp_file,
        write_owned_temp_bytes,
        sync_owned_temp_file,
        evaluate_pre_commit_gate,
    )

    destination = tmp_path / "artifact.bin"
    payload = b"concurrent-payload"

    ownership = create_owned_temp_file(destination)

    try:
        # Establish trusted write evidence.
        bytes_written = write_owned_temp_bytes(
            ownership,
            payload,
        )
        assert bytes_written == len(payload)

        # Establish trusted sync evidence.
        sync_owned_temp_file(ownership)

        # Issue one valid authorization token.
        gate = evaluate_pre_commit_gate(
            ownership=ownership,
            expected_bytes=len(payload),
            bytes_written=bytes_written,
            sync_completed=True,
        )
        assert gate.state.value == "passed"

        barrier = Barrier(2)

        def worker():
            barrier.wait()
            return commit_owned_temp_file(
                ownership=ownership,
                destination=destination,
                mode=AtomicCommitMode.REPLACE,
                gate=gate,
            )

        # Both threads race to consume the same authorization.
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda _: worker(),
                    range(2),
                )
            )

        # Exactly one thread may successfully commit.
        assert sum(
            result.outcome is AtomicCommitOutcome.COMMITTED
            for result in results
        ) == 1

        assert sum(
            result.outcome is AtomicCommitOutcome.NOT_COMMITTED
            for result in results
        ) == 1

        assert destination.read_bytes() == payload

    finally:
        os.close(ownership.fd)
        ownership.path.unlink(missing_ok=True)