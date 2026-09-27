"""Integrity primitives for the fresh C70S100 score-control campaign.

The campaign is intentionally write-once: a canonical condition/seed run
directory is reserved atomically before training, and a completion manifest is
published atomically only after the fixed training and test rollout artifacts
exist.  This module uses only the Python standard library so both the launcher
and post-run analyzer can enforce the same contract without a GPU runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


CONDITIONS: Tuple[str, ...] = ("uniform", "aligned", "zeroed", "deranged")
SEEDS: Tuple[int, ...] = (42, 43, 44, 45)
COMPLETION_MANIFEST_FILENAME = "score_control_completion_manifest.json"
SOURCE_LOCK_FILENAME = "c70s100_score_control_source_lock.json"
SOURCE_LOCK_SCHEMA_VERSION = 1
SOURCE_LOCK_STUDY = "fresh C70S100 score-edge control study v2"

# These names are deliberately fixed in code as well as enumerated in the
# JSON lock.  In particular, a lock file cannot redirect validation to a
# similarly named file elsewhere in the workspace.
SOURCE_LOCK_SOURCE_FILES: Tuple[Tuple[str, str], ...] = (
    ("trainer", "train_gnn_fno_semibc.py"),
    ("score_edge_controls", "score_edge_controls.py"),
    ("case_split_controls", "case_split_controls.py"),
    ("integrity_contract", "c70_score_control_integrity.py"),
    ("locked_launcher", "train_our_gnn_fno_gnn_c70s100_score_controls.py"),
)
SOURCE_LOCK_SOURCE_NAMES: Tuple[str, ...] = tuple(
    name for name, _ in SOURCE_LOCK_SOURCE_FILES
)
SOURCE_LOCK_SOURCE_PATHS: Dict[str, str] = dict(SOURCE_LOCK_SOURCE_FILES)


class ScoreControlIntegrityError(RuntimeError):
    """Raised when a fresh score-control run cannot meet its write-once contract."""


@dataclass(frozen=True)
class VerifiedSourceLock:
    """The raw lock digest and fully verified source-member digests.

    The JSON lock intentionally does not contain a hash of itself.  Its raw
    SHA-256 is recorded by the run manifest, which avoids a self-referential
    digest cycle while still binding a result to the exact lock bytes used at
    launch time.
    """

    path: Path
    raw_sha256: str
    source_sha256: Dict[str, str]


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _source_root(source_root: Optional[Path] = None) -> Path:
    """Resolve the sole directory permitted to contain locked source files."""

    if source_root is None:
        return Path(__file__).resolve().parent
    return Path(source_root).resolve(strict=False)


def canonical_source_lock_path(source_root: Optional[Path] = None) -> Path:
    """Return the canonical lock location for a source root."""

    return _source_root(source_root) / SOURCE_LOCK_FILENAME


def _read_source_lock_json(path: Path) -> Mapping[str, Any]:
    """Read a regular, local source lock with an exact schema."""

    if path.is_symlink() or not path.is_file():
        raise ScoreControlIntegrityError(
            "source lock must be a non-symlink regular file: {}".format(path)
        )
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScoreControlIntegrityError(
            "cannot read valid source lock {}: {}".format(path, exc)
        )
    if not isinstance(value, dict):
        raise ScoreControlIntegrityError("source lock must be a JSON object: {}".format(path))
    expected_keys = {"schema_version", "study", "source_files", "source_sha256"}
    if set(value) != expected_keys:
        raise ScoreControlIntegrityError("source lock has an unexpected key inventory: {}".format(path))
    if value["schema_version"] != SOURCE_LOCK_SCHEMA_VERSION:
        raise ScoreControlIntegrityError("source lock schema is unsupported: {}".format(path))
    if value["study"] != SOURCE_LOCK_STUDY:
        raise ScoreControlIntegrityError("source lock study differs from C70S100: {}".format(path))
    if value["source_files"] != SOURCE_LOCK_SOURCE_PATHS:
        raise ScoreControlIntegrityError("source lock source-file inventory differs: {}".format(path))
    source_sha256 = value["source_sha256"]
    if not isinstance(source_sha256, dict) or set(source_sha256) != set(SOURCE_LOCK_SOURCE_NAMES):
        raise ScoreControlIntegrityError("source lock digest inventory differs: {}".format(path))
    if not all(_is_sha256(digest) for digest in source_sha256.values()):
        raise ScoreControlIntegrityError("source lock contains a malformed source digest: {}".format(path))
    return value


def verify_source_lock(
    source_root: Optional[Path] = None,
    source_lock_path: Optional[Path] = None,
    require_canonical_path: bool = True,
) -> VerifiedSourceLock:
    """Verify a lock and every one of its five fixed source members.

    Production callers use the defaults, which require the canonical lock
    beside this module.  Dependency-free tests may explicitly provide a
    fixture root and lock with ``require_canonical_path=False``; the same
    regular-file, exact-schema, and byte-digest checks still apply.
    """

    root = _source_root(source_root)
    canonical = canonical_source_lock_path(root)
    if source_lock_path is None:
        lock_path = canonical
    else:
        supplied_lock_path = Path(source_lock_path)
        # Resolve only after inspecting the supplied directory entry.  Calling
        # resolve first would turn a symlink fixture into its target and hide a
        # prohibited indirection from the regular-file guard below.
        if supplied_lock_path.is_symlink():
            raise ScoreControlIntegrityError(
                "source lock must not be a symlink: {}".format(supplied_lock_path)
            )
        lock_path = supplied_lock_path.resolve(strict=False)
    if require_canonical_path and lock_path != canonical:
        raise ScoreControlIntegrityError(
            "source lock path is not canonical: {}".format(lock_path)
        )
    value = _read_source_lock_json(lock_path)
    expected = value["source_sha256"]
    observed: Dict[str, str] = {}
    for name, filename in SOURCE_LOCK_SOURCE_FILES:
        source_path = root / filename
        observed[name] = sha256_file(source_path)
        if observed[name] != expected[name]:
            raise ScoreControlIntegrityError(
                "source lock digest differs for {}: {}".format(name, source_path)
            )
    return VerifiedSourceLock(
        path=lock_path,
        raw_sha256=sha256_file(lock_path),
        source_sha256=observed,
    )


def verify_canonical_source_lock() -> VerifiedSourceLock:
    """Verify the production C70S100 source lock and its fixed sources."""

    return verify_source_lock(require_canonical_path=True)


def canonical_run_name(condition: str, seed: int) -> str:
    """Return the sole permitted run-directory name for a campaign cell."""

    if condition not in CONDITIONS:
        raise ScoreControlIntegrityError("unknown score-control condition: {!r}".format(condition))
    if seed not in SEEDS:
        raise ScoreControlIntegrityError("unknown score-control seed: {!r}".format(seed))
    return "c70s100_score_{}_seed{}".format(condition, seed)


def canonical_run_names() -> Tuple[str, ...]:
    return tuple(canonical_run_name(condition, seed) for condition in CONDITIONS for seed in SEEDS)


def _lexists(path: Path) -> bool:
    """Treat dangling symlinks as existing artifacts too."""

    return os.path.lexists(str(path))


def find_preexisting_canonical_paths(runs_root: Path) -> Tuple[Path, ...]:
    """List every canonical campaign cell already present under ``runs_root``."""

    root = Path(runs_root)
    return tuple(root / name for name in canonical_run_names() if _lexists(root / name))


def assert_campaign_fresh(runs_root: Path) -> None:
    """Refuse a campaign launch if any canonical cell is already present.

    An old directory, file, or dangling symlink is evidence of a prior attempt
    and cannot be safely interpreted as a fresh matched run.  Noncanonical
    entries (for example a separately generated ``analysis`` directory) are
    deliberately outside this check.
    """

    existing = find_preexisting_canonical_paths(runs_root)
    if existing:
        raise ScoreControlIntegrityError(
            "fresh C70S100 campaign refused: canonical run artifact(s) already exist: {}".format(
                ", ".join(str(path) for path in existing)
            )
        )


def reserve_fresh_run_directory(run_dir: Path, condition: str, seed: int) -> Path:
    """Atomically reserve an otherwise absent canonical score-control run directory.

    ``mkdir`` is the actual reservation operation; a prior check improves the
    diagnostic, while ``FileExistsError`` closes the check--create race.  A
    failed run intentionally leaves its directory behind, so no later launch
    can silently resume or overwrite it.
    """

    run_path = Path(run_dir)
    expected_name = canonical_run_name(condition, seed)
    if run_path.name != expected_name:
        raise ScoreControlIntegrityError(
            "score-control run directory {!s} does not match canonical name {}".format(
                run_path, expected_name
            )
        )
    if _lexists(run_path):
        raise ScoreControlIntegrityError(
            "fresh C70S100 run refused: canonical run artifact already exists: {}".format(
                run_path
            )
        )
    run_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_path.mkdir()
    except FileExistsError:
        raise ScoreControlIntegrityError(
            "fresh C70S100 run refused: canonical run artifact appeared during reservation: {}".format(
                run_path
            )
        )
    return run_path


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a regular campaign artifact."""

    artifact = Path(path)
    if artifact.is_symlink() or not artifact.is_file():
        raise ScoreControlIntegrityError(
            "completion artifact must be a non-symlink regular file: {}".format(artifact)
        )
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_artifact(run_dir: Path, artifact_path: Path) -> Dict[str, str]:
    run_root = Path(run_dir).resolve()
    raw_artifact = Path(artifact_path)
    if raw_artifact.is_symlink():
        raise ScoreControlIntegrityError(
            "completion artifact must not be a symlink: {}".format(raw_artifact)
        )
    artifact = raw_artifact.resolve()
    try:
        relative = artifact.relative_to(run_root)
    except ValueError:
        raise ScoreControlIntegrityError(
            "completion artifact is outside its run directory: {}".format(artifact)
        )
    return {"path": relative.as_posix(), "sha256": sha256_file(artifact)}


def runtime_identity(torch_module: Any, physicsnemo_module: Optional[Any] = None) -> Dict[str, Any]:
    """Capture a compact, non-secret runtime identity after a successful run."""

    cuda_available = bool(torch_module.cuda.is_available())
    try:
        cuda_count = int(torch_module.cuda.device_count()) if cuda_available else 0
    except Exception:
        cuda_count = 0
    device_names = []
    for index in range(cuda_count):
        try:
            device_names.append(str(torch_module.cuda.get_device_name(index)))
        except Exception:
            device_names.append("unavailable")
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "torch": {
            "version": str(getattr(torch_module, "__version__", "unknown")),
            "cuda_version": str(getattr(getattr(torch_module, "version", None), "cuda", None)),
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_count,
            "cuda_device_names": device_names,
        },
        "physicsnemo": {
            "version": str(getattr(physicsnemo_module, "__version__", "unknown")),
        },
    }


def _atomic_write_json(destination: Path, payload: Mapping[str, Any]) -> None:
    """Publish one JSON artifact atomically, with no overwrite permission."""

    target = Path(destination)
    if _lexists(target):
        raise ScoreControlIntegrityError(
            "completion manifest already exists and will not be overwritten: {}".format(target)
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(target.name), suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        # Persist the directory entry when the filesystem permits it.  The
        # replace itself is atomic; this is durability hardening only.
        try:
            directory_fd = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_completion_manifest(
    run_dir: Path,
    condition: str,
    seed: int,
    start_epoch: int,
    end_epoch: int,
    checkpoint_path: Path,
    epoch_metrics_path: Path,
    rollout_metrics_path: Path,
    rollout_summary_path: Path,
    test_case: str,
    runtime: Mapping[str, Any],
    input_run_manifest_path: Path,
) -> Path:
    """Atomically publish the immutable end-of-run evidence manifest.

    The caller may invoke this only after the full rollout returns.  A missing
    or malformed artifact therefore prevents publication rather than creating
    an ambiguous partial-completion record.
    """

    canonical_run_name(condition, seed)
    if int(start_epoch) != 0:
        raise ScoreControlIntegrityError(
            "fresh score-control completion must start from epoch 0, got {}".format(start_epoch)
        )
    if int(end_epoch) < 1:
        raise ScoreControlIntegrityError("completion end epoch must be positive")
    if not isinstance(test_case, str) or not test_case:
        raise ScoreControlIntegrityError("completion test case must be a nonempty string")
    if not isinstance(runtime, Mapping):
        raise ScoreControlIntegrityError("completion runtime identity must be a mapping")

    root = Path(run_dir)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "condition": condition,
        "seed": int(seed),
        "training": {"start_epoch": 0, "end_epoch": int(end_epoch)},
        "test_case": test_case,
        "checkpoint": _relative_artifact(root, checkpoint_path),
        "artifacts": {
            "epoch_metrics_csv": _relative_artifact(root, epoch_metrics_path),
            "rollout_metrics_csv": _relative_artifact(root, rollout_metrics_path),
            "rollout_summary": _relative_artifact(root, rollout_summary_path),
            "input_run_manifest": _relative_artifact(root, input_run_manifest_path),
        },
        "runtime": dict(runtime),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    destination = root / COMPLETION_MANIFEST_FILENAME
    _atomic_write_json(destination, payload)
    return destination
