"""FastAPI entry point — OpenEnv create_app() + custom viewer."""

import os
from pathlib import Path

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from openenv.core.env_server.http_server import create_app

from .environment import OrigamiEnvironment
from .models import OrigamiAction, OrigamiObservation

app = create_app(
    OrigamiEnvironment,
    OrigamiAction,
    OrigamiObservation,
    env_name="origami_env",
    max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", 1)),
)

from .tasks import TASKS


@app.get("/tasks")
def get_tasks():
    return {
        name: {
            "name": task["name"],
            "description": task["description"],
            "difficulty": task["difficulty"],
            "paper": task["paper"],
            "target_fold": task["target_fold"],
        }
        for name, task in TASKS.items()
    }


@app.get("/tasks/{task_name}")
def get_task_detail(task_name: str):
    if task_name not in TASKS:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Task '{task_name}' not found")
    task = TASKS[task_name]
    return {
        "name": task["name"],
        "description": task["description"],
        "difficulty": task["difficulty"],
        "paper": task["paper"],
        "target_fold": task["target_fold"],
    }


_VIEWER_DIR = Path(__file__).resolve().parent.parent / "viewer"
if _VIEWER_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_VIEWER_DIR), html=True), name="renderer")
else:

    @app.get("/", response_class=HTMLResponse)
    def no_viewer():
        return HTMLResponse("<h3>Viewer not found</h3><p>API docs at <a href='/docs'>/docs</a></p>")


def main():
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
