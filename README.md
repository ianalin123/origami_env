---
title: Origami Env Environment Server
emoji: 🦢
colorFrom: red
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Origami Env

RL environment for origami folding — LLM generates FOLD crease patterns, physics simulates them, reward = shape similarity to target.

## Usage

```python
from origami_env.client import OrigamiEnv
from origami_env.models import OrigamiAction

with OrigamiEnv(base_url="http://localhost:8000") as env:
    result = env.reset(task_name="triangle")
    result = env.step(OrigamiAction(fold_data={
        "vertices_coords": [[0,0],[1,0],[1,1],[0,1]],
        "edges_vertices": [[0,1],[1,2],[2,3],[3,0],[0,2]],
        "edges_assignment": ["B","B","B","B","V"],
        "edges_foldAngle": [0,0,0,0,180]
    }))
    print(result.observation.shape_similarity)
```

## Tasks

- **triangle** — diagonal valley fold
- **half_fold** — horizontal valley fold at y=0.5
- **quarter_fold** — two perpendicular valley folds
- **letter_fold** — two parallel valley folds at y=1/3 and y=2/3
