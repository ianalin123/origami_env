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
        "max_folds": 1,
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
        "max_folds": 1,
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
        "max_folds": 2,
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
        "max_folds": 2,
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
    "waterbomb_base": {
        "name": "waterbomb_base",
        "description": "Create a waterbomb base with four valley folds: both diagonals and both center lines",
        "difficulty": 3,
        "max_folds": 4,
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
                # Diagonal folds
                [0, 8], [2, 6],
                # Center folds
                [1, 7], [3, 5],
            ],
            "edges_assignment": [
                "B", "B", "B", "B", "B", "B", "B", "B",
                "V", "V",
                "V", "V",
            ],
            "edges_foldAngle": [
                0, 0, 0, 0, 0, 0, 0, 0,
                180, 180,
                180, 180,
            ],
            "faces_vertices": [
                [0, 1, 4, 3], [1, 2, 5, 4],
                [3, 4, 7, 6], [4, 5, 8, 7],
            ],
        },
    },
    "map_fold": {
        "name": "map_fold",
        "description": "Accordion fold into a 4x4 grid with alternating mountain and valley creases along both axes",
        "difficulty": 4,
        "max_folds": 6,
        "paper": {"width": 1.0, "height": 1.0},
        "target_fold": {
            "vertices_coords": [
                [0, 0], [0.25, 0], [0.5, 0], [0.75, 0], [1, 0],
                [0, 0.25], [0.25, 0.25], [0.5, 0.25], [0.75, 0.25], [1, 0.25],
                [0, 0.5], [0.25, 0.5], [0.5, 0.5], [0.75, 0.5], [1, 0.5],
                [0, 0.75], [0.25, 0.75], [0.5, 0.75], [0.75, 0.75], [1, 0.75],
                [0, 1], [0.25, 1], [0.5, 1], [0.75, 1], [1, 1],
            ],
            "edges_vertices": [
                # Boundary
                [0, 1], [1, 2], [2, 3], [3, 4],
                [4, 9], [9, 14], [14, 19], [19, 24],
                [24, 23], [23, 22], [22, 21], [21, 20],
                [20, 15], [15, 10], [10, 5], [5, 0],
                # Horizontal folds (y = 0.25, 0.5, 0.75)
                [5, 9],   # y=0.25 valley
                [10, 14], # y=0.5  mountain
                [15, 19], # y=0.75 valley
                # Vertical folds (x = 0.25, 0.5, 0.75)
                [1, 21],  # x=0.25 valley
                [2, 22],  # x=0.5  mountain
                [3, 23],  # x=0.75 valley
            ],
            "edges_assignment": [
                "B", "B", "B", "B",
                "B", "B", "B", "B",
                "B", "B", "B", "B",
                "B", "B", "B", "B",
                "V", "M", "V",
                "V", "M", "V",
            ],
            "edges_foldAngle": [
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                180, -180, 180,
                180, -180, 180,
            ],
            "faces_vertices": [
                [0, 1, 6, 5], [1, 2, 7, 6], [2, 3, 8, 7], [3, 4, 9, 8],
                [5, 6, 11, 10], [6, 7, 12, 11], [7, 8, 13, 12], [8, 9, 14, 13],
                [10, 11, 16, 15], [11, 12, 17, 16], [12, 13, 18, 17], [13, 14, 19, 18],
                [15, 16, 21, 20], [16, 17, 22, 21], [17, 18, 23, 22], [18, 19, 24, 23],
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


def get_task_for_step_mode(name: str) -> dict:
    """Get a task, validating it has max_folds set (required for step mode)."""
    task = get_task(name)
    if "max_folds" not in task:
        raise ValueError(f"Task '{name}' missing max_folds — not compatible with step mode")
    return task
