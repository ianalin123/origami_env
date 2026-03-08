"""Task definitions for origami RL training.

Each task defines a target shape as a reference FOLD crease pattern.
The LLM must discover a crease pattern that folds into the same shape.

Starting simple (triangle) and progressing to harder folds.
"""

TASKS: dict[str, dict] = {
    "triangle": {
        "name": "triangle",
        "description": "Fold the paper in half diagonally to make a triangle",
        "difficulty": 1,
        "paper": {"width": 1.0, "height": 1.0},
        "target_fold": {
            "vertices_coords": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "edges_vertices": [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
            "edges_assignment": ["B", "B", "B", "B", "V"],
            "edges_foldAngle": [0, 0, 0, 0, 180],
            "faces_vertices": [[0, 1, 2], [0, 2, 3]],
        },
    },
    "half_fold": {
        "name": "half_fold",
        "description": "Fold the paper in half horizontally",
        "difficulty": 1,
        "paper": {"width": 1.0, "height": 1.0},
        "target_fold": {
            "vertices_coords": [
                [0, 0], [1, 0], [1, 1], [0, 1], [0, 0.5], [1, 0.5],
            ],
            "edges_vertices": [
                [0, 1], [1, 5], [5, 2], [2, 3], [3, 4], [4, 0],
                [4, 5],
            ],
            "edges_assignment": ["B", "B", "B", "B", "B", "B", "V"],
            "edges_foldAngle": [0, 0, 0, 0, 0, 0, 180],
            "faces_vertices": [[0, 1, 5, 4], [4, 5, 2, 3]],
        },
    },
    "quarter_fold": {
        "name": "quarter_fold",
        "description": "Fold the paper into quarters (two perpendicular folds)",
        "difficulty": 2,
        "paper": {"width": 1.0, "height": 1.0},
        "target_fold": {
            "vertices_coords": [
                [0, 0], [0.5, 0], [1, 0],
                [0, 0.5], [0.5, 0.5], [1, 0.5],
                [0, 1], [0.5, 1], [1, 1],
            ],
            "edges_vertices": [
                # Boundary
                [0, 1], [1, 2], [2, 5], [5, 8], [8, 7], [7, 6], [6, 3], [3, 0],
                # Fold lines
                [1, 4], [4, 7],  # vertical fold
                [3, 4], [4, 5],  # horizontal fold
            ],
            "edges_assignment": [
                "B", "B", "B", "B", "B", "B", "B", "B",
                "V", "V", "V", "V",
            ],
            "edges_foldAngle": [
                0, 0, 0, 0, 0, 0, 0, 0,
                180, 180, 180, 180,
            ],
            "faces_vertices": [
                [0, 1, 4, 3],  # bottom-left
                [1, 2, 5, 4],  # bottom-right
                [3, 4, 7, 6],  # top-left
                [4, 5, 8, 7],  # top-right
            ],
        },
    },
    "letter_fold": {
        "name": "letter_fold",
        "description": "Tri-fold the paper like a letter (two parallel folds)",
        "difficulty": 2,
        "paper": {"width": 1.0, "height": 1.0},
        "target_fold": {
            "vertices_coords": [
                [0, 0], [1, 0],
                [0, 1/3], [1, 1/3],
                [0, 2/3], [1, 2/3],
                [0, 1], [1, 1],
            ],
            "edges_vertices": [
                # Boundary
                [0, 1], [1, 3], [3, 5], [5, 7], [7, 6], [6, 4], [4, 2], [2, 0],
                # Fold lines
                [2, 3],  # first fold (valley)
                [4, 5],  # second fold (mountain)
            ],
            "edges_assignment": [
                "B", "B", "B", "B", "B", "B", "B", "B",
                "V", "M",
            ],
            "edges_foldAngle": [
                0, 0, 0, 0, 0, 0, 0, 0,
                180, -180,
            ],
            "faces_vertices": [
                [0, 1, 3, 2],  # bottom strip
                [2, 3, 5, 4],  # middle strip
                [4, 5, 7, 6],  # top strip
            ],
        },
    },
}


def get_task(name: str | None = None) -> dict:
    """Get a task by name. Defaults to 'triangle'."""
    if name is None:
        name = "triangle"
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASKS.keys())}")
    return TASKS[name]


def list_tasks() -> list[str]:
    """List all available task names."""
    return list(TASKS.keys())
