"""Validated, frozen case-split support for matched experiment arms.

The helpers avoid training/runtime dependencies so a campaign split can be
checked before a GPU job is launched.  An explicit manifest is necessary when
multiple random seeds must share one held-out configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SPLIT_KEYS = ("train_cases", "valid_cases", "test_cases")
C70_TRAINING_WINDOW_MANIFEST_FILENAME = "c70s100_score_control_training_windows.json"
C70_TRAINING_WINDOW_SCHEMA_VERSION = 1
C70_TRAINING_WINDOW_SEEDS = (42, 43, 44, 45)
C70_TRAINING_WINDOW_STRATA = ("early", "middle", "late")
_C70_FRAME_COUNT_RE = re.compile(r"__frames-(\d+)$")


def canonical_split_sha256(split: Mapping[str, Sequence[str]]) -> str:
    """Return a stable digest of just the three semantic case lists."""

    canonical = {}
    for key in SPLIT_KEYS:
        values = split.get(key)
        if not isinstance(values, (list, tuple)) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(f"{key} must be a list of strings")
        canonical[key] = list(values)
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value without whitespace or key-order ambiguity."""

    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _frame_count_from_case_name(case_name: str) -> int:
    match = _C70_FRAME_COUNT_RE.search(case_name)
    if match is None:
        raise ValueError(
            "C70 training-window manifest case has no terminal frames-NNN suffix: "
            + repr(case_name)
        )
    return int(match.group(1))


def _temporal_stratum_bounds(valid_start_count: int) -> Dict[str, Tuple[int, int]]:
    """Return non-overlapping early/middle/late half-open start-index ranges."""

    if valid_start_count < 3:
        raise ValueError("C70 training-window manifest needs at least three starts")
    first = (valid_start_count + 2) // 3
    second = (2 * valid_start_count + 2) // 3
    return {
        "early": (0, first),
        "middle": (first, second),
        "late": (second, valid_start_count),
    }


def _normalize_training_window_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the semantic part of a C70 frozen-window manifest.

    The prose-only ``purpose`` field is deliberately omitted from this digest;
    all executable selections, their seed mapping, the split digest, and the
    training unroll remain bound.
    """

    if not isinstance(manifest, Mapping):
        raise ValueError("C70 training-window manifest must be a JSON object")
    required = {
        "schema_version",
        "purpose",
        "split_semantic_sha256",
        "training_rollout_steps",
        "seeds",
    }
    if set(manifest) != required:
        raise ValueError("C70 training-window manifest has an unexpected key inventory")
    if manifest["schema_version"] != C70_TRAINING_WINDOW_SCHEMA_VERSION:
        raise ValueError("C70 training-window manifest schema is unsupported")
    if not isinstance(manifest["purpose"], str) or not manifest["purpose"].strip():
        raise ValueError("C70 training-window manifest purpose must be nonempty")
    split_sha = manifest["split_semantic_sha256"]
    if not isinstance(split_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", split_sha):
        raise ValueError("C70 training-window manifest split digest is malformed")
    rollout_steps = manifest["training_rollout_steps"]
    if isinstance(rollout_steps, bool) or not isinstance(rollout_steps, int):
        raise ValueError("C70 training-window manifest rollout steps must be an integer")
    seeds = manifest["seeds"]
    if not isinstance(seeds, Mapping):
        raise ValueError("C70 training-window manifest seeds must be an object")
    return {
        "schema_version": manifest["schema_version"],
        "split_semantic_sha256": split_sha,
        "training_rollout_steps": rollout_steps,
        "seeds": dict(seeds),
    }


def canonical_c70_training_window_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Return a stable digest of the executable frozen-window selection."""

    return _canonical_json_sha256(_normalize_training_window_manifest(manifest))


def c70_training_window_sample_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the ordered list of case/start/stratum records used by one seed."""

    normalized = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("C70 training-window record must be an object")
        if set(record) != {"case", "start", "stratum"}:
            raise ValueError("C70 training-window record has an unexpected key inventory")
        case = record["case"]
        start = record["start"]
        stratum = record["stratum"]
        if not isinstance(case, str) or not case:
            raise ValueError("C70 training-window record case must be nonempty")
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("C70 training-window record start must be nonnegative")
        if stratum not in C70_TRAINING_WINDOW_STRATA:
            raise ValueError("C70 training-window record has an unknown stratum")
        normalized.append({"case": case, "start": start, "stratum": stratum})
    return _canonical_json_sha256(normalized)


def _validate_c70_window_environment_coverage(train_cases: Sequence[str]) -> None:
    """Enforce the declared 14-trajectory multi-room/long-trace study scope."""

    if len(train_cases) != 14 or len(set(train_cases)) != 14:
        raise ValueError("C70 training-window manifest requires exactly 14 unique training cases")
    # A directory ending in ``frames-201`` can be an exact prefix of a
    # same-condition ``frames-401`` directory.  Case-directory names alone
    # must never be used to count such a prefix twice as independent training
    # evidence.  The v2 split has a byte-level audit for the known historical
    # duplicate and rejects this whole class of duplicated case identities.
    trajectory_ids = [re.sub(r"__frames-\d+$", "", name) for name in train_cases]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError(
            "C70 training-window manifest cannot count a frame-prefix duplicate as a distinct source trajectory"
        )
    big = [name for name in train_cases if name.startswith("big_room__")]
    small = [name for name in train_cases if name.startswith("small_room__")]
    if len(big) != 2 or len(small) != 12:
        raise ValueError("C70 training-window manifest requires exactly two big-room and twelve small-room cases")
    frames = {name: _frame_count_from_case_name(name) for name in train_cases}
    if sorted(frames.values()) != [201] * 12 + [401] * 2:
        raise ValueError("C70 training-window manifest requires twelve 201-state and two 401-state trajectories")
    long_cases = [name for name, count in frames.items() if count == 401]
    if sum(name.startswith("big_room__") for name in long_cases) != 1 or sum(
        name.startswith("small_room__") for name in long_cases
    ) != 1:
        raise ValueError("C70 training-window manifest requires one long trace in each room configuration")


def validate_c70_training_window_manifest(
    manifest: Mapping[str, Any],
    expected_train_cases: Sequence[str],
    expected_split_semantic_sha256: str,
) -> Dict[int, Dict[str, Any]]:
    """Validate every frozen C70 seed selection and return normalized records.

    Each of the 12 201-state trajectories contributes 14 windows (5/5/4 over
    early/middle/late thirds); each of the two 401-state trajectories
    contributes 16 (6/5/5).  That makes the fixed 200-window budget cover all
    cases and all temporal thirds without inventing physical seconds.
    """

    normalized = _normalize_training_window_manifest(manifest)
    if normalized["split_semantic_sha256"] != expected_split_semantic_sha256:
        raise ValueError("C70 training-window manifest is bound to a different case split")
    if normalized["training_rollout_steps"] != 3:
        raise ValueError("C70 training-window manifest requires the fixed three-step training rollout")
    train_cases = list(expected_train_cases)
    if not all(isinstance(name, str) and name for name in train_cases):
        raise ValueError("C70 training-window expected training cases must be nonempty strings")
    _validate_c70_window_environment_coverage(train_cases)

    seeds = normalized["seeds"]
    expected_seed_keys = {str(seed) for seed in C70_TRAINING_WINDOW_SEEDS}
    if set(seeds) != expected_seed_keys:
        raise ValueError("C70 training-window manifest seed inventory differs from the locked study")

    selections: Dict[int, Dict[str, Any]] = {}
    for seed in C70_TRAINING_WINDOW_SEEDS:
        per_case = seeds[str(seed)]
        if not isinstance(per_case, Mapping) or set(per_case) != set(train_cases):
            raise ValueError(
                "C70 training-window manifest case inventory differs for seed {}".format(seed)
            )
        records: List[Dict[str, Any]] = []
        per_case_counts: Dict[str, int] = {}
        per_stratum_counts = {stratum: 0 for stratum in C70_TRAINING_WINDOW_STRATA}
        room_sample_counts = {"big_room": 0, "small_room": 0}
        state_count_sample_counts = {"201": 0, "401": 0}
        seen = set()
        for case in train_cases:
            per_stratum = per_case[case]
            if not isinstance(per_stratum, Mapping) or set(per_stratum) != set(
                C70_TRAINING_WINDOW_STRATA
            ):
                raise ValueError(
                    "C70 training-window manifest strata differ for seed {} case {}".format(
                        seed, case
                    )
                )
            frame_count = _frame_count_from_case_name(case)
            valid_start_count = frame_count - normalized["training_rollout_steps"]
            bounds = _temporal_stratum_bounds(valid_start_count)
            expected_count = 14 if frame_count == 201 else 16
            expected_sorted_stratum_counts = [4, 5, 5] if frame_count == 201 else [5, 5, 6]
            case_count = 0
            observed_stratum_counts = []
            for stratum in C70_TRAINING_WINDOW_STRATA:
                starts = per_stratum[stratum]
                if not isinstance(starts, list) or any(
                    isinstance(start, bool) or not isinstance(start, int) for start in starts
                ):
                    raise ValueError(
                        "C70 training-window starts must be integer lists for seed {} case {}".format(
                            seed, case
                        )
                    )
                if starts != sorted(starts) or len(set(starts)) != len(starts):
                    raise ValueError(
                        "C70 training-window starts must be sorted and unique for seed {} case {}".format(
                            seed, case
                        )
                    )
                lo, hi = bounds[stratum]
                if any(start < lo or start >= hi for start in starts):
                    raise ValueError(
                        "C70 training-window start lies outside its temporal stratum for seed {} case {}".format(
                            seed, case
                        )
                    )
                observed_stratum_counts.append(len(starts))
                per_stratum_counts[stratum] += len(starts)
                case_count += len(starts)
                for start in starts:
                    key = (case, start)
                    if key in seen:
                        raise ValueError(
                            "C70 training-window manifest duplicates a case/start selection"
                        )
                    seen.add(key)
                    records.append({"case": case, "start": start, "stratum": stratum})
            if case_count != expected_count or sorted(observed_stratum_counts) != expected_sorted_stratum_counts:
                raise ValueError(
                    "C70 training-window manifest has an invalid temporal quota for seed {} case {}".format(
                        seed, case
                    )
                )
            per_case_counts[case] = case_count
            room = "big_room" if case.startswith("big_room__") else "small_room"
            room_sample_counts[room] += case_count
            state_count_sample_counts[str(frame_count)] += case_count
        if len(records) != 200 or len(seen) != 200:
            raise ValueError("C70 training-window manifest must select exactly 200 unique windows per seed")
        selections[seed] = {
            "records": records,
            "sample_sha256": c70_training_window_sample_sha256(records),
            "summary": {
                "sample_count": len(records),
                "case_count": len(per_case_counts),
                "per_case_counts": per_case_counts,
                "per_stratum_counts": per_stratum_counts,
                "room_sample_counts": room_sample_counts,
                "state_count_sample_counts": state_count_sample_counts,
            },
        }
    return selections


def load_c70_training_window_selection(
    manifest_path: str,
    expected_train_cases: Sequence[str],
    expected_split_semantic_sha256: str,
    seed: int,
) -> Dict[str, Any]:
    """Load one locked seed selection and its raw/semantic provenance."""

    if seed not in C70_TRAINING_WINDOW_SEEDS:
        raise ValueError("C70 training-window selection uses an unapproved seed")
    path = os.path.abspath(manifest_path)
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise ValueError("cannot read C70 training-window manifest: {}".format(path)) from exc
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("C70 training-window manifest is not UTF-8 JSON") from exc
    selections = validate_c70_training_window_manifest(
        manifest, expected_train_cases, expected_split_semantic_sha256
    )
    selection = dict(selections[seed])
    selection.update(
        {
            "filename": os.path.basename(path),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "semantic_sha256": canonical_c70_training_window_manifest_sha256(manifest),
        }
    )
    return selection


def resolve_c70_training_window_samples(
    records: Sequence[Mapping[str, Any]],
    candidate_samples: Sequence[Tuple[str, int]],
) -> List[Tuple[str, int]]:
    """Map a validated frozen record list onto the dataset's absolute paths."""

    available: Dict[str, str] = {}
    candidates = set()
    for case_dir, start in candidate_samples:
        case = os.path.basename(os.path.normpath(case_dir))
        existing = available.get(case)
        if existing is not None and existing != case_dir:
            raise ValueError("C70 training-window candidates have ambiguous case paths")
        available[case] = case_dir
        candidates.add((case, int(start)))
    resolved = []
    for record in records:
        case = str(record["case"])
        start = int(record["start"])
        if case not in available or (case, start) not in candidates:
            raise ValueError(
                "C70 training-window selection is unavailable in the live dataset: {} @ {}".format(
                    case, start
                )
            )
        resolved.append((available[case], start))
    if len(resolved) != 200 or len(set(resolved)) != 200:
        raise ValueError("C70 training-window selection did not resolve to 200 unique samples")
    return resolved


def load_explicit_case_split(
    case_dirs: Iterable[str],
    manifest_path: str,
    expected_counts: Mapping[str, int] | None = None,
) -> Tuple[List[str], List[str], List[str], Dict[str, str]]:
    """Resolve and validate an explicit JSON split against available cases.

    The manifest must provide ``train_cases``, ``valid_cases``, and
    ``test_cases`` lists.  Each listed basename must exist exactly once under
    ``case_dirs`` and partitions must be disjoint.  Extra cases in the source
    root are allowed so a frozen split remains valid if unrelated data are
    present later.
    """

    if not isinstance(manifest_path, str) or not manifest_path:
        raise ValueError("explicit case-split manifest path must be nonempty")
    manifest_path = os.path.abspath(manifest_path)
    try:
        with open(manifest_path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise ValueError(f"cannot read explicit case-split manifest: {manifest_path}") from exc
    try:
        split = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("explicit case-split manifest is not UTF-8 JSON") from exc
    if not isinstance(split, dict):
        raise ValueError("explicit case-split manifest must be a JSON object")

    resolved: Dict[str, List[str]] = {}
    names: List[str] = []
    for key in SPLIT_KEYS:
        values = split.get(key)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"explicit case-split field {key} must be a list of strings")
        resolved[key] = list(values)
        names.extend(values)

    if len(set(names)) != len(names):
        raise ValueError("explicit case-split partitions are not disjoint")
    if expected_counts is not None:
        for key in SPLIT_KEYS:
            if key in expected_counts and len(resolved[key]) != int(expected_counts[key]):
                raise ValueError(
                    f"explicit case-split {key} has {len(resolved[key])} cases, "
                    f"expected {int(expected_counts[key])}"
                )

    available: Dict[str, str] = {}
    for case_dir in case_dirs:
        name = os.path.basename(os.path.normpath(case_dir))
        if name in available:
            raise ValueError(f"ambiguous available case basename: {name}")
        available[name] = case_dir
    unknown = sorted(set(names).difference(available))
    if unknown:
        raise ValueError("explicit case-split refers to unavailable cases: " + ", ".join(unknown))

    metadata = {
        "split_mode": "explicit-manifest",
        "split_manifest_path": manifest_path,
        "split_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "split_semantic_sha256": canonical_split_sha256(resolved),
    }
    return (
        [available[name] for name in resolved["train_cases"]],
        [available[name] for name in resolved["valid_cases"]],
        [available[name] for name in resolved["test_cases"]],
        metadata,
    )
