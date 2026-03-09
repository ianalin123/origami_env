CURRICULUM = [
    {"steps": (0, 200), "tasks": ["triangle", "half_fold"]},
    {"steps": (200, 500), "tasks": ["triangle", "half_fold", "quarter_fold", "letter_fold"]},
    {"steps": (500, 900), "tasks": ["quarter_fold", "letter_fold", "waterbomb_base"]},
    {"steps": (900, 1500), "tasks": ["waterbomb_base", "map_fold"]},
]


def get_task_pool(global_step: int) -> list[str]:
    for phase in CURRICULUM:
        if phase["steps"][0] <= global_step < phase["steps"][1]:
            return phase["tasks"]
    return CURRICULUM[-1]["tasks"]
