"""Tests for origami RL environment."""

import numpy as np
import pytest

from origami_server.engine.fold_parser import parse_fold, validate_fold
from origami_server.engine.shape_match import compute_shape_match
from origami_server.engine.simulate import simulate
from origami_server.environment import OrigamiEnvironment
from origami_server.models import OrigamiAction
from origami_server.tasks import TASKS, get_task, list_tasks
from training.reward import extract_fold_json, shape_match, valid_fold

# --- Fixtures ---

TRIANGLE_FOLD = {
    "vertices_coords": [[0, 0], [1, 0], [1, 1], [0, 1]],
    "edges_vertices": [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
    "edges_assignment": ["B", "B", "B", "B", "V"],
    "edges_foldAngle": [0, 0, 0, 0, 180],
}

HALF_FOLD = {
    "vertices_coords": [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0.5], [1, 0.5]],
    "edges_vertices": [[0, 1], [1, 5], [5, 4], [4, 0], [4, 3], [3, 2], [2, 5], [4, 5]],
    "edges_assignment": ["B", "B", "B", "B", "B", "B", "B", "V"],
    "edges_foldAngle": [0, 0, 0, 0, 0, 0, 0, 180],
}


# --- Validation ---


class TestValidation:
    def test_valid_fold_accepted(self):
        valid, err = validate_fold(TRIANGLE_FOLD)
        assert valid, err

    def test_missing_field_rejected(self):
        valid, _ = validate_fold({"vertices_coords": [[0, 0]]})
        assert not valid

    def test_no_creases_rejected(self):
        valid, _ = validate_fold(
            {
                "vertices_coords": [[0, 0], [1, 0], [1, 1]],
                "edges_vertices": [[0, 1], [1, 2], [2, 0]],
                "edges_assignment": ["B", "B", "B"],
            }
        )
        assert not valid

    def test_bad_vertex_index_rejected(self):
        valid, _ = validate_fold(
            {
                "vertices_coords": [[0, 0], [1, 0], [1, 1]],
                "edges_vertices": [[0, 1], [1, 2], [2, 99]],
                "edges_assignment": ["B", "B", "V"],
            }
        )
        assert not valid

    def test_degenerate_edge_rejected(self):
        valid, _ = validate_fold(
            {
                "vertices_coords": [[0, 0], [1, 0], [1, 1]],
                "edges_vertices": [[0, 1], [1, 1], [1, 2]],
                "edges_assignment": ["B", "V", "B"],
            }
        )
        assert not valid


# --- Parsing ---


class TestParsing:
    def test_parse_creates_3d_vertices(self):
        p = parse_fold(TRIANGLE_FOLD)
        assert p["vertices"].shape == (4, 3)
        assert np.allclose(p["vertices"][:, 2], 0)

    def test_parse_computes_faces(self):
        p = parse_fold(TRIANGLE_FOLD)
        assert len(p["faces"]) >= 2

    def test_parse_angles_in_radians(self):
        p = parse_fold(TRIANGLE_FOLD)
        assert abs(p["fold_angles"][4] - np.pi) < 0.01


# --- Physics ---


class TestPhysics:
    def test_flat_stays_flat(self):
        r = simulate(TRIANGLE_FOLD, crease_percent=0.0, max_steps=100)
        assert np.max(np.abs(r.positions[:, 2])) < 0.01

    def test_fold_creates_z_displacement(self):
        # A partial fold (90°) creates z displacement; full 180° folds flat
        r = simulate(TRIANGLE_FOLD, crease_percent=0.5, max_steps=2000)
        z_range = r.positions[:, 2].max() - r.positions[:, 2].min()
        assert z_range > 0.1

    def test_valley_fold_brings_vertices_together(self):
        r = simulate(TRIANGLE_FOLD, crease_percent=1.0, max_steps=2000)
        dist = np.linalg.norm(r.positions[1] - r.positions[3])
        assert dist < 0.1

    def test_half_fold_works(self):
        # Full fold: top vertices should overlap bottom vertices
        r = simulate(HALF_FOLD, crease_percent=1.0, max_steps=2000)
        # v2=[1,1] should fold onto v1=[1,0], v3=[0,1] onto v0=[0,0]
        dist = np.linalg.norm(r.positions[2] - r.positions[1])
        assert dist < 0.1, f"v2 didn't fold onto v1 (dist={dist})"

    def test_all_tasks_fold(self):
        for name, task in TASKS.items():
            r = simulate(task["target_fold"], crease_percent=1.0, max_steps=2000)
            assert r.converged, f"Task {name} did not converge"
            assert r.max_strain < 0.01, f"Task {name} has high strain ({r.max_strain})"
            # Partial fold should produce z displacement
            r_half = simulate(task["target_fold"], crease_percent=0.5)
            z_range = r_half.positions[:, 2].max() - r_half.positions[:, 2].min()
            assert z_range > 0.01, f"Task {name} partial fold no z (z_range={z_range})"


# --- Shape Match ---


class TestShapeMatch:
    def test_same_shape_perfect_match(self):
        r = simulate(TRIANGLE_FOLD, crease_percent=1.0, max_steps=2000)
        sim = compute_shape_match(r.positions, r.positions)
        assert sim > 0.99

    def test_different_shapes_lower_match(self):
        target = simulate(TRIANGLE_FOLD, crease_percent=1.0, max_steps=2000)
        wrong = simulate(HALF_FOLD, crease_percent=1.0, max_steps=2000)
        sim = compute_shape_match(wrong.positions, target.positions)
        assert sim < 0.95

    def test_flat_vs_folded_lower_match(self):
        target = simulate(TRIANGLE_FOLD, crease_percent=1.0, max_steps=2000)
        flat = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        sim = compute_shape_match(flat, target.positions)
        assert sim < 0.95


# --- Environment ---


class TestEnvironment:
    def test_reset(self):
        env = OrigamiEnvironment()
        obs = env.reset(task_name="triangle")
        assert not obs.done
        assert obs.reward is None
        assert len(obs.target_positions) > 0

    def test_step_correct_fold(self):
        env = OrigamiEnvironment()
        env.reset(task_name="triangle")
        obs = env.step(OrigamiAction(fold_data=TRIANGLE_FOLD))
        assert obs.done
        assert obs.reward == 20.0
        assert obs.shape_similarity == 1.0

    def test_step_invalid_fold(self):
        env = OrigamiEnvironment()
        env.reset(task_name="triangle")
        obs = env.step(OrigamiAction(fold_data={"bad": True}))
        assert obs.done
        assert obs.reward == -2.0
        assert obs.error is not None

    def test_state(self):
        env = OrigamiEnvironment()
        env.reset(task_name="triangle")
        assert env.state.task_name == "triangle"


# --- Tasks ---


class TestTasks:
    def test_four_tasks(self):
        assert len(list_tasks()) == 4

    def test_get_task(self):
        task = get_task("triangle")
        assert task["name"] == "triangle"

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            get_task("nonexistent")


# --- Rewards ---


class TestRewards:
    def test_extract_json_fenced(self):
        text = '```json\n{"vertices_coords": [[0, 0]]}\n```'
        assert extract_fold_json(text) is not None

    def test_extract_json_raw(self):
        text = '{"vertices_coords": [[0, 0]]}'
        assert extract_fold_json(text) is not None

    def test_extract_none_garbage(self):
        assert extract_fold_json("no json here") is None

    def test_valid_fold_reward(self):
        import json

        good = [[{"content": json.dumps(TRIANGLE_FOLD)}]]
        bad = [[{"content": "nope"}]]
        scores = valid_fold(good + bad)
        assert scores[0] == 1.0
        assert scores[1] == -2.0

    def test_shape_match_reward(self):
        import json

        good = [[{"content": json.dumps(TRIANGLE_FOLD)}]]
        bad = [[{"content": "nope"}]]
        scores = shape_match(good + bad, task_name="triangle")
        assert scores[0] == 20.0
        assert scores[1] == -2.0


# --- API ---


class TestAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from origami_server.app import app

        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_tasks_endpoint(self, client):
        r = client.get("/tasks")
        assert r.status_code == 200
        tasks = r.json()
        assert "triangle" in tasks
        assert "half_fold" in tasks
        assert len(tasks) == 4

    def test_task_detail_endpoint(self, client):
        r = client.get("/tasks/triangle")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "triangle"
        assert "target_fold" in data

    def test_task_not_found(self, client):
        r = client.get("/tasks/nonexistent")
        assert r.status_code == 404

    def test_websocket_reset_step(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {"task_name": "triangle"}})
            resp = ws.receive_json()
            assert resp["type"] == "observation"
            assert resp["data"]["done"] is False

            ws.send_json({"type": "step", "data": {"fold_data": TRIANGLE_FOLD}})
            resp = ws.receive_json()
            assert resp["type"] == "observation"
            assert resp["data"]["reward"] == 20.0
            assert resp["data"]["done"] is True

    def test_websocket_all_tasks(self, client):
        for task_name in ("triangle", "half_fold", "quarter_fold", "letter_fold"):
            r = client.get(f"/tasks/{task_name}")
            fold_data = r.json()["target_fold"]
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "reset", "data": {"task_name": task_name}})
                ws.receive_json()
                ws.send_json({"type": "step", "data": {"fold_data": fold_data}})
                resp = ws.receive_json()
                assert resp["data"]["reward"] == 20.0, f"{task_name} failed"
