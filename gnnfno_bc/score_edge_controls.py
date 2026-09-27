"""Deterministic score-side edge controls for 3-D graph simulators.

The functions here deliberately distinguish a message's edge attributes from
the edge attributes presented to an attention scorer.  They are dependency
light so their invariants can be tested without the 3-D VTK/PhysicsNeMo stack.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch


SCORE_EDGE_MODES = frozenset(("aligned", "zeroed", "deranged"))
DERANGEMENT_NAMESPACE = "serverroom-3d-score-edge-derangement-v1"


def validate_score_edge_mode(mode: str) -> str:
    mode = str(mode)
    if mode not in SCORE_EDGE_MODES:
        allowed = ", ".join(sorted(SCORE_EDGE_MODES))
        raise ValueError(f"score_edge_mode must be one of {allowed}, got {mode!r}")
    return mode


def destination_local_derangement(
    edge_index: np.ndarray,
    node_ids: np.ndarray,
    case_key: str,
) -> np.ndarray:
    """Map each directed edge to another edge into the same destination.

    The mapping is a deterministic, destination-local cyclic derangement.
    ``node_ids`` are immutable IDs before any training-node subsampling, which
    makes the mapping independent of incidental local index order.  Incoming
    edges are ordered by immutable source ID; a case/destination hash chooses
    a nonzero cyclic offset.  The output ``permutation[e]`` selects the edge
    attribute presented at scorer row ``e``.
    """

    edge_index = np.asarray(edge_index, dtype=np.int64)
    node_ids = np.asarray(node_ids, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if node_ids.ndim != 1:
        raise ValueError("node_ids must be one-dimensional")
    if np.unique(node_ids).size != node_ids.size:
        raise ValueError("node_ids must be unique")
    if not isinstance(case_key, str) or not case_key:
        raise ValueError("case_key must be a nonempty string")
    src, dst = edge_index
    if src.size == 0:
        raise ValueError("edge_index must be nonempty")
    if np.any(src < 0) or np.any(dst < 0) or np.any(src >= node_ids.size) or np.any(dst >= node_ids.size):
        raise ValueError("edge_index refers to an out-of-range node")
    pair_order = np.lexsort((dst, src))
    if pair_order.size > 1 and np.any(
        (src[pair_order[1:]] == src[pair_order[:-1]])
        & (dst[pair_order[1:]] == dst[pair_order[:-1]])
    ):
        raise ValueError("directed edge identities must be unique")

    # One lexicographic sort puts each destination in a contiguous block with
    # immutable source IDs in canonical order.  This avoids an O(N*E) scan of
    # all E edges for every destination on room-scale full meshes.
    ordered = np.lexsort((node_ids[src], dst))
    ordered_dst = dst[ordered]
    starts = np.r_[0, np.flatnonzero(ordered_dst[1:] != ordered_dst[:-1]) + 1]
    ends = np.r_[starts[1:], ordered.size]
    lengths = ends - starts
    if np.any(lengths < 2):
        raise ValueError("every destination requires at least two incoming edges")

    # Mix the immutable destination ID with a case-specific seed without a
    # Python loop.  Each group therefore receives a deterministic nonzero
    # cyclic offset, preserving destination locality and bijectivity.
    case_seed = np.uint64(
        int.from_bytes(
            hashlib.sha256(
                f"{DERANGEMENT_NAMESPACE}|{case_key}".encode("utf-8")
            ).digest()[:8],
            "little",
        )
    )
    destination_ids = node_ids[ordered_dst[starts]].astype(np.uint64, copy=False)
    mixed = destination_ids + case_seed
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    offsets = (
        mixed % (lengths.astype(np.uint64) - np.uint64(1))
    ).astype(np.int64) + 1

    group_index = np.repeat(np.arange(starts.size, dtype=np.int64), lengths)
    position = np.arange(ordered.size, dtype=np.int64)
    local_position = position - starts[group_index]
    source_position = starts[group_index] + (
        (local_position + offsets[group_index]) % lengths[group_index]
    )
    permutation = np.empty(src.size, dtype=np.int64)
    permutation[ordered] = ordered[source_position]

    if np.any(permutation == np.arange(src.size, dtype=np.int64)):
        raise ValueError("score-edge derangement contains a fixed point")
    if not np.array_equal(dst, dst[permutation]):
        raise ValueError("score-edge derangement crossed destination segments")
    if not np.array_equal(np.sort(permutation), np.arange(src.size, dtype=np.int64)):
        raise ValueError("score-edge derangement is not bijective")
    return permutation


def validate_score_edge_permutation(
    edge_index: torch.Tensor,
    permutation: torch.Tensor,
    *,
    require_unique_directed_edges: bool = False,
) -> None:
    """Validate the scorer-only permutation carried with a graph sample.

    ``destination_local_derangement`` establishes the full identity contract at
    data construction time.  This standalone validator is retained for
    preflight and unit tests; the model consumes only permutations created by
    that constructor, avoiding a repeated O(E log E) sort in every rollout
    layer.  The optional duplicate-edge test is useful in unit tests.
    """

    if not isinstance(edge_index, torch.Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be a tensor with shape [2, E]")
    if not isinstance(permutation, torch.Tensor) or permutation.ndim != 1:
        raise ValueError("edge permutation must be a one-dimensional tensor")
    edge_count = edge_index.shape[1]
    if permutation.numel() != edge_count:
        raise ValueError("edge permutation has the wrong shape")
    if permutation.dtype != torch.long:
        raise ValueError("edge permutation must have torch.long dtype")
    if permutation.device != edge_index.device:
        raise ValueError("edge permutation and edge_index must share a device")
    expected = torch.arange(edge_count, device=edge_index.device, dtype=torch.long)
    if permutation.numel() and (
        int(permutation.min()) < 0 or int(permutation.max()) >= edge_count
    ):
        raise ValueError("edge permutation contains an out-of-range value")
    if not torch.equal(torch.sort(permutation).values, expected):
        raise ValueError("edge permutation is not bijective")
    if torch.any(permutation == expected):
        raise ValueError("edge permutation contains a fixed point")
    if not torch.equal(edge_index[1], edge_index[1].index_select(0, permutation)):
        raise ValueError("edge permutation crossed destination segments")
    if require_unique_directed_edges:
        pairs = edge_index.t().contiguous()
        if torch.unique(pairs, dim=0).shape[0] != edge_count:
            raise ValueError("directed edge identities must be unique")


def score_edge_attributes(
    edge_attr: torch.Tensor,
    mode: str,
    permutation: torch.Tensor | None = None,
    *,
    validate_permutation: bool = True,
) -> torch.Tensor:
    """Return scorer-only edge attributes without changing message attributes."""

    mode = validate_score_edge_mode(mode)
    if edge_attr.ndim != 2:
        raise ValueError("edge_attr must have shape [E, F]")
    if mode == "aligned":
        return edge_attr
    if mode == "zeroed":
        return torch.zeros_like(edge_attr)
    if permutation is None:
        raise ValueError("deranged score mode requires an edge permutation")
    if not isinstance(permutation, torch.Tensor) or permutation.ndim != 1 or permutation.numel() != edge_attr.shape[0]:
        raise ValueError("edge permutation has the wrong shape")
    if permutation.device != edge_attr.device:
        raise ValueError("edge permutation and edge_attr must share a device")
    if permutation.dtype != torch.long:
        raise ValueError("edge permutation must have torch.long dtype")
    if validate_permutation:
        if permutation.numel() and (int(permutation.min()) < 0 or int(permutation.max()) >= edge_attr.shape[0]):
            raise ValueError("edge permutation contains an out-of-range value")
        if not torch.equal(torch.sort(permutation).values, torch.arange(edge_attr.shape[0], device=edge_attr.device)):
            raise ValueError("edge permutation is not bijective")
    return edge_attr.index_select(0, permutation)
