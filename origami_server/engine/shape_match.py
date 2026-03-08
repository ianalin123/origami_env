"""Shape matching for reward computation.

Computes similarity between the LLM's folded shape and the target shape.
Like AlphaFold's RMSD but for origami vertex positions.
"""

import numpy as np
from scipy.spatial.distance import cdist


def compute_shape_match(
    predicted: np.ndarray,
    target: np.ndarray,
) -> float:
    """Compute shape similarity between predicted and target positions.

    Uses chamfer distance normalized by bounding box diagonal.
    Aligns shapes by centering before comparison.

    Args:
        predicted: (N, 3) predicted vertex positions.
        target: (M, 3) target vertex positions.

    Returns:
        Similarity score in [0, 1]. 1.0 = perfect match.
    """
    if len(predicted) == 0 or len(target) == 0:
        return 0.0

    # Center both point clouds
    pred_centered = predicted - predicted.mean(axis=0)
    target_centered = target - target.mean(axis=0)

    # Try multiple rotations and pick best match
    # (the LLM's pattern might produce a rotated version of the target)
    best_score = 0.0
    for rotation in _get_alignment_rotations():
        rotated = pred_centered @ rotation.T
        score = _chamfer_similarity(rotated, target_centered)
        best_score = max(best_score, score)

    return best_score


def _chamfer_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Chamfer distance converted to similarity score.

    Chamfer = average nearest-neighbor distance (bidirectional).
    Similarity = 1 - (chamfer / diagonal), clamped to [0, 1].
    """
    d = cdist(a, b)

    # Forward: for each point in a, min distance to b
    forward = d.min(axis=1).mean()
    # Backward: for each point in b, min distance to a
    backward = d.min(axis=0).mean()
    chamfer = (forward + backward) / 2.0

    # Normalize by bounding box diagonal of target
    all_pts = np.vstack([a, b])
    bbox_diag = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))
    if bbox_diag < 1e-12:
        return 1.0 if chamfer < 1e-12 else 0.0

    similarity = max(0.0, 1.0 - chamfer / bbox_diag)
    return similarity


def _get_alignment_rotations() -> list[np.ndarray]:
    """Generate rotation matrices for alignment search.

    We check identity + 90° rotations around each axis (24 orientations).
    This handles cases where the LLM's fold produces a rotated version.
    """
    I = np.eye(3)
    rotations = [I]

    # 90° rotations around Z axis
    for k in range(1, 4):
        angle = k * np.pi / 2
        c, s = np.cos(angle), np.sin(angle)
        rotations.append(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]))

    # 90° rotations around X axis
    for k in range(1, 4):
        angle = k * np.pi / 2
        c, s = np.cos(angle), np.sin(angle)
        rotations.append(np.array([[1, 0, 0], [0, c, -s], [0, s, c]]))

    # 90° rotations around Y axis
    for k in range(1, 4):
        angle = k * np.pi / 2
        c, s = np.cos(angle), np.sin(angle)
        rotations.append(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))

    # Flip (mirror)
    rotations.append(np.diag([-1, 1, 1]))
    rotations.append(np.diag([1, -1, 1]))
    rotations.append(np.diag([1, 1, -1]))

    return rotations
