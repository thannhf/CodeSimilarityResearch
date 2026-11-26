from __future__ import annotations
import hashlib
import os
import secrets
import uuid
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import ctypes
import msvcrt
from ctypes import wintypes
from concurrent.futures import ThreadPoolExecutor
import threading

TEMP_CREATE_MAX_ATTEMPTS = 8
TEMP_FILE_MODE = 0o600
TEMP_WRITE_MAX_INTERRUPTED_RETRIES = 8


class TempCreationError(Exception):
    """Raised when a transaction temp file cannot be safely created."""


if os.name == "nt":
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _CreateFileW = _kernel32.CreateFileW
    _CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    _CreateFileW.restype = wintypes.HANDLE


    def _windows_create_temp_fd(path: Path) -> int:
        GENERIC_READ = 0x80000000
        GENERIC_WRITE = 0x40000000

        FILE_SHARE_READ = 0x00000001
        FILE_SHARE_WRITE = 0x00000002
        FILE_SHARE_DELETE = 0x00000004

        CREATE_NEW = 1
        FILE_ATTRIBUTE_NORMAL = 0x00000080

        handle = _CreateFileW(
            str(path),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            None,
            CREATE_NEW,
            FILE_ATTRIBUTE_NORMAL,
            None,
        )

        INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

        if handle == INVALID_HANDLE_VALUE:
            error_code = ctypes.get_last_error()
            raise ctypes.WinError(error_code)

        try:
            fd = msvcrt.open_osfhandle(
                handle,
                os.O_BINARY | os.O_RDWR,
            )
        except BaseException:
            _kernel32.CloseHandle(handle)
            raise

        # open_osfhandle takes ownership of the Windows handle
        # on success. Do not CloseHandle again after this point.
        return fd


@dataclass(frozen=True)
class TempFileOwnership:
    transaction_id: str
    path: Path
    fd: int
    identity: tuple[int, int]
    attempt: int

class TempOwnershipLifecycle(Enum):
    ACTIVE = "active"
    CONSUMED = "consumed"
    REVOKED = "revoked"

class TempCleanupOutcome(Enum):
    CLEANED = "cleaned"
    NOT_NEEDED = "not_needed"
    PRESERVED = "preserved"
    FAILED = "failed"

class DirectorySyncState(Enum):
    SYNCED = "synced"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"

@dataclass(frozen=True)
class DirectorySyncResult:
    state: DirectorySyncState
    error: str | None = None

@dataclass(frozen=True)
class TempCleanupResult:
    outcome: TempCleanupOutcome
    path: Path
    file_removed: bool
    fd_closed: bool
    error: str | None = None

@dataclass
class TempOwnershipRegistryRecord:
    ownership: TempFileOwnership
    lifecycle: TempOwnershipLifecycle = (
        TempOwnershipLifecycle.ACTIVE
    )
    write_completed: bool = False
    bytes_written: int = 0
    payload_digest: str | None = None
    sync_completed: bool = False
    authorization_consumed: bool = False
    authorization_token: str | None = None

_TEMP_OWNERSHIP_REGISTRY: dict[
    str,
    TempOwnershipRegistryRecord,
] = {}
_TEMP_OWNERSHIP_REGISTRY_LOCK = threading.Lock()

class OwnershipVerificationState(Enum):
    VERIFIED = "verified"
    MISSING = "missing"
    MISMATCH = "mismatch"
    UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class OwnershipVerification:
    state: OwnershipVerificationState
    expected_identity: tuple[int, int]
    descriptor_identity: tuple[int, int] | None
    path_identity: tuple[int, int] | None
    error: str | None = None

def create_owned_temp_file(destination: Path) -> TempFileOwnership:
    """
    Create an exclusively-owned temp file in the destination directory.
    Does not write payload, flush, fsync, commit, or perform cleanup.
    """
    if not isinstance(destination, Path):
        raise TypeError("destination must be a Path.")

    if not destination.is_absolute():
        raise ValueError("destination must be absolute.")

    transaction_id = uuid.uuid4().hex
    parent = destination.parent

    
    if os.name != "nt":
        binary_flag = getattr(os, "O_BINARY", 0)
        flags = (os.O_CREAT, os.O_EXCL, os.O_WRONLY, binary_flag)


    for attempt in range(TEMP_CREATE_MAX_ATTEMPTS):
        temp_path = parent / (
            f".safepersist-{transaction_id}-{attempt}.tmp"
        )

        try:
            if os.name == "nt":
                fd = _windows_create_temp_fd(temp_path)
            else:
                fd = os.open(
                    temp_path,
                    flags,
                    TEMP_FILE_MODE,
                )

        except FileExistsError:
            # Retry only when the candidate name collides.
            continue

        try:
            file_stat = os.fstat(fd)
        except OSError as exc:
            # Identity capture failed. Close our descriptor but do not
            # perform pathname cleanup without verified ownership.
            try:
                os.close(fd)
            except OSError:
                pass

            raise TempCreationError(
                "Temp file created, but ownership identity "
                "could not be captured."
            ) from exc

        identity = (file_stat.st_dev, file_stat.st_ino)

        ownership = TempFileOwnership(
            transaction_id=transaction_id,
            path=temp_path,
            fd=fd,
            identity=identity,
            attempt=attempt,
        )

        _TEMP_OWNERSHIP_REGISTRY[transaction_id] = (
            TempOwnershipRegistryRecord(
                ownership=ownership,
            )
        )

        return ownership

    raise TempCreationError(
        f"Unable to reserve temp file after "
        f"{TEMP_CREATE_MAX_ATTEMPTS} attempts."
    )

def verify_temp_ownership(
    ownership: TempFileOwnership,
) -> OwnershipVerification:
    """
    Verify the temp pathname and retained descriptor against
    the identity captured at creation time.

    Path state is checked first so missing/replaced entries can
    be detected on platforms that restrict mutation of open files.

    This is an observation only. It does not perform cleanup
    and does not eliminate pathname TOCTOU races.
    """
    if not isinstance(ownership, TempFileOwnership):
        raise TypeError("ownership must be TempFileOwnership.")

    expected = ownership.identity

    try:
        path_stat = os.lstat(ownership.path)
    except FileNotFoundError:
        return OwnershipVerification(
            state=OwnershipVerificationState.MISSING,
            expected_identity=expected,
            descriptor_identity=None,
            path_identity=None,
            error=None,
        )
    except OSError as exc:
        return OwnershipVerification(
            state=OwnershipVerificationState.UNVERIFIABLE,
            expected_identity=expected,
            descriptor_identity=None,
            path_identity=None,
            error=f"Path identity verification failed: {exc}",
        )

    path_identity = (
        path_stat.st_dev,
        path_stat.st_ino,
    )

    if path_identity != expected:
        return OwnershipVerification(
            state=OwnershipVerificationState.MISMATCH,
            expected_identity=expected,
            descriptor_identity=None,
            path_identity=path_identity,
            error="Temp pathname no longer identifies the owned file.",
        )

    try:
        descriptor_stat = os.fstat(ownership.fd)
    except OSError as exc:
        return OwnershipVerification(
            state=OwnershipVerificationState.UNVERIFIABLE,
            expected_identity=expected,
            descriptor_identity=None,
            path_identity=path_identity,
            error=f"Descriptor verification failed: {exc}",
        )

    descriptor_identity = (
        descriptor_stat.st_dev,
        descriptor_stat.st_ino,
    )

    if descriptor_identity != expected:
        return OwnershipVerification(
            state=OwnershipVerificationState.MISMATCH,
            expected_identity=expected,
            descriptor_identity=descriptor_identity,
            path_identity=path_identity,
            error="Descriptor identity differs from captured identity.",
        )

    return OwnershipVerification(
        state=OwnershipVerificationState.VERIFIED,
        expected_identity=expected,
        descriptor_identity=descriptor_identity,
        path_identity=path_identity,
        error=None,
    )


def cleanup_owned_temp_file(
    ownership: TempFileOwnership,
    *,
    preserve_temp: bool = False,
) -> TempCleanupResult:
    if not isinstance(ownership, TempFileOwnership):
        raise TypeError("ownership must be TempFileOwnership.")

    file_removed = False
    fd_closed = False

    with _TEMP_OWNERSHIP_REGISTRY_LOCK:
        record = _TEMP_OWNERSHIP_REGISTRY.get(
            ownership.transaction_id
        )

        if (
            record is None
            or record.ownership is not ownership
            or record.lifecycle not in (
                TempOwnershipLifecycle.ACTIVE,
                TempOwnershipLifecycle.CONSUMED,
            )
            or (
                preserve_temp
                and record.lifecycle is not TempOwnershipLifecycle.ACTIVE
            )
        ):
            return TempCleanupResult(
                outcome=TempCleanupOutcome.FAILED,
                path=ownership.path,
                file_removed=False,
                fd_closed=False,
                error="Ownership is not active or registered.",
            )

        verification = verify_temp_ownership(ownership)

        if verification.state is OwnershipVerificationState.MISSING:
            outcome = TempCleanupOutcome.NOT_NEEDED

        elif preserve_temp:
            outcome = TempCleanupOutcome.PRESERVED

        elif verification.state is not OwnershipVerificationState.VERIFIED:
            outcome = TempCleanupOutcome.FAILED

        else:
            try:
                ownership.path.unlink()
                file_removed = True
                outcome = TempCleanupOutcome.CLEANED
            except FileNotFoundError:
                outcome = TempCleanupOutcome.NOT_NEEDED
            except OSError as exc:
                outcome = TempCleanupOutcome.FAILED
                error = f"Temp removal failed: {exc}"

        try:
            os.close(ownership.fd)
            fd_closed = True
        except OSError:
            fd_closed = False

        record.lifecycle = TempOwnershipLifecycle.REVOKED

    return TempCleanupResult(
        outcome=outcome,
        path=ownership.path,
        file_removed=file_removed,
        fd_closed=fd_closed,
        error=locals().get("error"),
    )

class TempWriteError(Exception):
    """Raised when writing bytes to an owned temp descriptor fails."""

    def __init__(self, message: str, *, bytes_written: int):
        super().__init__(message)
        self.bytes_written = bytes_written

class TempSyncError(Exception):
    """Raised when syncing an owned temp file fails."""


def write_owned_temp_bytes(
    ownership: TempFileOwnership,
    payload: bytes,
) -> int:
    interrupted_retries = 0

    if not isinstance(ownership, TempFileOwnership):
        raise TypeError("ownership must be TempFileOwnership")

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")

    with _TEMP_OWNERSHIP_REGISTRY_LOCK:
        registered = _TEMP_OWNERSHIP_REGISTRY.get(
            ownership.transaction_id
        )

        if (
            registered is None
            or registered.ownership is not ownership
            or registered.lifecycle is not TempOwnershipLifecycle.ACTIVE
        ):
            raise TempWriteError(
                "Temp ownership is not registered or active.",
                bytes_written=0,
            )

        # Invalidate previous evidence before starting a new write.
        registered.write_completed = False
        registered.bytes_written = 0
        registered.payload_digest = None
        registered.sync_completed = False

        verification = verify_temp_ownership(ownership)

        if verification.state is not OwnershipVerificationState.VERIFIED:
            raise TempWriteError(
                (
                    "Temp ownership verification failed before write: "
                    f"{verification.state.value}"
                ),
                bytes_written=0,
            )

        total_written = 0
        view = memoryview(payload)

        while total_written < len(payload):
            try:
                written = os.write(
                    ownership.fd,
                    view[total_written:],
                )
            except InterruptedError as exc:
                interrupted_retries += 1

                if interrupted_retries > TEMP_WRITE_MAX_INTERRUPTED_RETRIES:
                    registered.bytes_written = total_written
                    raise TempWriteError(
                        "Write interrupted too many times.",
                        bytes_written=total_written,
                    ) from exc

                continue

            except OSError as exc:
                registered.bytes_written = total_written
                raise TempWriteError(
                    "Failed while writing owned temp file.",
                    bytes_written=total_written,
                ) from exc

            if written <= 0:
                registered.bytes_written = total_written
                raise TempWriteError(
                    "Temp file write returned zero bytes.",
                    bytes_written=total_written,
                )

            total_written += written
            interrupted_retries = 0

        registered.write_completed = True
        registered.bytes_written = total_written
        registered.payload_digest = hashlib.sha256(payload).hexdigest()
        registered.sync_completed = False

        return total_written


def sync_owned_temp_file(
    ownership: TempFileOwnership,
) -> None:
    """
    Sync an owned temp file using its retained descriptor.

    Requires completed write evidence for the same active ownership.
    Does not write payload, close the descriptor, commit,
    or perform cleanup.
    """
    if not isinstance(ownership, TempFileOwnership):
        raise TypeError("ownership must be TempFileOwnership")

    with _TEMP_OWNERSHIP_REGISTRY_LOCK:
        registered = _TEMP_OWNERSHIP_REGISTRY.get(
            ownership.transaction_id
        )

        if (
            registered is None
            or registered.ownership is not ownership
            or registered.lifecycle is not TempOwnershipLifecycle.ACTIVE
        ):
            raise TempSyncError(
                "Temp ownership is not registered or active."
            )

        verification = verify_temp_ownership(ownership)

        if verification.state is not OwnershipVerificationState.VERIFIED:
            raise TempSyncError(
                (
                    "Temp ownership verification failed before sync: "
                    f"{verification.state.value}"
                )
            )

        if (
            not registered.write_completed
            or registered.payload_digest is None
        ):
            raise TempSyncError(
                "Sync requires completed write evidence."
            )

        if registered.sync_completed:
            raise TempSyncError(
                "Temp file has already been synced for this write."
            )

        try:
            descriptor_stat = os.fstat(ownership.fd)
        except OSError as exc:
            raise TempSyncError(
                "Failed to inspect owned temp file before sync."
            ) from exc

        if descriptor_stat.st_size != registered.bytes_written:
            raise TempSyncError(
                "Temp file size differs from registered write evidence."
            )

        try:
            os.fsync(ownership.fd)
        except OSError as exc:
            registered.sync_completed = False
            raise TempSyncError(
                "Failed to sync owned temp file."
            ) from exc

        registered.sync_completed = True



def sync_committed_directory(
    destination: Path,
) -> DirectorySyncResult:
    """
    Sync the parent directory after a successful atomic commit.

    POSIX: opens and fsyncs the destination directory.
    Windows: reports UNSUPPORTED rather than claiming durability.
    """
    if not isinstance(destination, Path):
        raise TypeError("destination must be a Path.")

    if not destination.is_absolute():
        raise ValueError("destination must be absolute.")

    if os.name == "nt":
        return DirectorySyncResult(
            state=DirectorySyncState.UNSUPPORTED,
            error="Directory fsync is unsupported by this implementation on Windows.",
        )

    directory_fd = None

    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)

        directory_fd = os.open(
            destination.parent,
            flags,
        )

        os.fsync(directory_fd)

        return DirectorySyncResult(
            state=DirectorySyncState.SYNCED,
        )

    except OSError as exc:
        return DirectorySyncResult(
            state=DirectorySyncState.FAILED,
            error=f"Directory sync failed: {exc}",
        )

    finally:
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                pass


class PreCommitGateState(Enum):
    PASSED = "passed"
    DENIED = "denied"


class PreCommitDenialReason(Enum):
    OWNERSHIP_NOT_VERIFIED = "ownership_not_verified"
    WRITE_NOT_COMPLETED = "write_not_completed"
    SYNC_NOT_COMPLETED = "sync_not_completed"


@dataclass(frozen=True)
class PreCommitGateResult:
    state: PreCommitGateState
    reason: PreCommitDenialReason | None = None
    transaction_id: str | None = None
    payload_digest: str | None = None
    authorization_token: str | None = None

    @property
    def allowed(self) -> bool:
        return self.state is PreCommitGateState.PASSED


def evaluate_pre_commit_gate(
    *,
    ownership: TempFileOwnership,
    expected_bytes: int,
    bytes_written: int,
    sync_completed: bool,
) -> PreCommitGateResult:
    """Issue transaction-bound commit authorization from trusted evidence."""

    if not isinstance(ownership, TempFileOwnership):
        raise TypeError("ownership must be TempFileOwnership")

    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise TypeError("expected_bytes must be an integer")

    if isinstance(bytes_written, bool) or not isinstance(bytes_written, int):
        raise TypeError("bytes_written must be an integer")

    if not isinstance(sync_completed, bool):
        raise TypeError("sync_completed must be bool")

    if expected_bytes < 0:
        raise ValueError("expected_bytes must be non-negative")

    if bytes_written < 0:
        raise ValueError("bytes_written must be non-negative")

    with _TEMP_OWNERSHIP_REGISTRY_LOCK:
        registered = _TEMP_OWNERSHIP_REGISTRY.get(
            ownership.transaction_id
        )

        if (
            registered is None
            or registered.ownership is not ownership
            or registered.lifecycle is not TempOwnershipLifecycle.ACTIVE
        ):
            return PreCommitGateResult(
                state=PreCommitGateState.DENIED,
                reason=PreCommitDenialReason.OWNERSHIP_NOT_VERIFIED,
            )

        verification = verify_temp_ownership(ownership)

        if verification.state is not OwnershipVerificationState.VERIFIED:
            return PreCommitGateResult(
                state=PreCommitGateState.DENIED,
                reason=PreCommitDenialReason.OWNERSHIP_NOT_VERIFIED,
            )

        if (
            not registered.write_completed
            or registered.payload_digest is None
            or registered.bytes_written != expected_bytes
            or bytes_written != registered.bytes_written
        ):
            return PreCommitGateResult(
                state=PreCommitGateState.DENIED,
                reason=PreCommitDenialReason.WRITE_NOT_COMPLETED,
            )

        if (
            not registered.sync_completed
            or not sync_completed
            or registered.authorization_consumed
            or registered.authorization_token is not None
        ):
            return PreCommitGateResult(
                state=PreCommitGateState.DENIED,
                reason=PreCommitDenialReason.SYNC_NOT_COMPLETED,
            )

        try:
            actual_size = os.fstat(ownership.fd).st_size
        except OSError:
            return PreCommitGateResult(
                state=PreCommitGateState.DENIED,
                reason=PreCommitDenialReason.WRITE_NOT_COMPLETED,
            )

        if actual_size != registered.bytes_written:
            return PreCommitGateResult(
                state=PreCommitGateState.DENIED,
                reason=PreCommitDenialReason.WRITE_NOT_COMPLETED,
            )

        token = secrets.token_urlsafe(32)
        registered.authorization_token = token

        return PreCommitGateResult(
            state=PreCommitGateState.PASSED,
            transaction_id=ownership.transaction_id,
            payload_digest=registered.payload_digest,
            authorization_token=token,
        )


class AtomicCommitMode(Enum):
    FAIL_IF_EXISTS = "fail_if_exists"
    REPLACE = "replace"


class AtomicCommitOutcome(Enum):
    COMMITTED = "committed"
    NOT_COMMITTED = "not_committed"
    OUTCOME_UNKNOWN = "outcome_unknown"

@dataclass(frozen=True)
class AtomicCommitResult:
    outcome: AtomicCommitOutcome
    mode: AtomicCommitMode
    destination: Path
    error: Exception | None = None

def commit_owned_temp_file(
    *,
    ownership: TempFileOwnership,
    destination: Path,
    mode: AtomicCommitMode,
    gate: PreCommitGateResult,
) -> AtomicCommitResult:
    """Commit an owned temp file using one atomic filesystem primitive.

    This function never retries, falls back, or performs cleanup.
    """

    if not isinstance(ownership, TempFileOwnership):
        raise TypeError("ownership must be TempFileOwnership")

    with _TEMP_OWNERSHIP_REGISTRY_LOCK:
        registered = _TEMP_OWNERSHIP_REGISTRY.get(
            ownership.transaction_id
        )

        if (
            registered is None
            or registered.ownership is not ownership
            or registered.lifecycle is not TempOwnershipLifecycle.ACTIVE
        ):
            return AtomicCommitResult(
                outcome=AtomicCommitOutcome.NOT_COMMITTED,
                mode=mode,
                destination=destination,
                error=RuntimeError(
                    "Temp ownership is not registered by trusted layer."
                ),
            )

        if not isinstance(destination, Path):
            raise TypeError("destination must be a Path")

        if not destination.is_absolute():
            raise ValueError("destination must be absolute")

        if not isinstance(mode, AtomicCommitMode):
            raise TypeError("mode must be AtomicCommitMode")

        if not isinstance(gate, PreCommitGateResult):
            raise TypeError("gate must be PreCommitGateResult")

        if not gate.allowed:
            return AtomicCommitResult(
                outcome=AtomicCommitOutcome.NOT_COMMITTED,
                mode=mode,
                destination=destination,
                error=RuntimeError("Pre-commit gate denied commit."),
            )

        if (
            gate.transaction_id != ownership.transaction_id
            or gate.payload_digest is None
            or gate.authorization_token is None
            or registered.payload_digest != gate.payload_digest
            or registered.authorization_token is None
            or registered.authorization_consumed
            or not secrets.compare_digest(
                gate.authorization_token,
                registered.authorization_token,
            )
            or not registered.write_completed
            or not registered.sync_completed
        ):
            return AtomicCommitResult(
                outcome=AtomicCommitOutcome.NOT_COMMITTED,
                mode=mode,
                destination=destination,
                error=RuntimeError(
                    "Pre-commit authorization is invalid, "
                    "foreign, or already consumed."
                ),
            )

        # Consume authorization before the atomic filesystem operation.
        # Any failure remains non-replayable by design.
        registered.authorization_consumed = True
        registered.authorization_token = None

        if ownership.path.parent != destination.parent:
            raise ValueError(
                "Temp file and destination must share the same parent directory."
            )

        verification = verify_temp_ownership(ownership)

        if verification.state is not OwnershipVerificationState.VERIFIED:
            return AtomicCommitResult(
                outcome=AtomicCommitOutcome.NOT_COMMITTED,
                mode=mode,
                destination=destination,
                error=RuntimeError(
                    "Temp ownership verification failed before commit: "
                    f"{verification.state.value}"
                ),
            )

        try:
            if mode is AtomicCommitMode.FAIL_IF_EXISTS:
                os.link(ownership.path, destination)
            else:
                os.replace(ownership.path, destination)

        except FileExistsError as exc:
            return AtomicCommitResult(
                outcome=AtomicCommitOutcome.NOT_COMMITTED,
                mode=mode,
                destination=destination,
                error=exc,
            )

        except FileNotFoundError as exc:
            return AtomicCommitResult(
                outcome=AtomicCommitOutcome.NOT_COMMITTED,
                mode=mode,
                destination=destination,
                error=exc,
            )

        except OSError as exc:
            # Conservatively classify other filesystem errors:
            # commit outcome may not be safely inferable.
            return AtomicCommitResult(
                outcome=AtomicCommitOutcome.OUTCOME_UNKNOWN,
                mode=mode,
                destination=destination,
                error=exc,
            )

        registered.lifecycle = TempOwnershipLifecycle.CONSUMED

        return AtomicCommitResult(
            outcome=AtomicCommitOutcome.COMMITTED,
            mode=mode,
            destination=destination,
        )