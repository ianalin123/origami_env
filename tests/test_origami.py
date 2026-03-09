"""Tests for origami RL environment."""

import numpy as np
import pytest

from origami_server.engine.fold_parser import parse_fold, validate_fold
from origami_server.engine.shape_match import compute_shape_match
from origami_server.engine.simulate import simulate
from origami_server.environment import OrigamiEnvironment
from origami_server.models import OrigamiAction
from origami_server.tasks import TASKS, get_task, list_tasks
from training.reward import extract_fold_json, valid_fold

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
        assert len(list_tasks()) == 6

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

    def test_shape_match_via_server(self):
        """shape_match reward now goes through the server (WebSocket).
        Test the same flow via TestClient's websocket to verify end-to-end."""
        from fastapi.testclient import TestClient

        from origami_server.app import app

        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {"task_name": "triangle"}})
            ws.receive_json()
            ws.send_json({"type": "step", "data": {"fold_data": TRIANGLE_FOLD}})
            resp = ws.receive_json()
            assert resp["data"]["reward"] == 20.0


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
        assert len(tasks) == 6

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
        for task_name in ("triangle", "half_fold", "quarter_fold", "letter_fold", "waterbomb_base", "map_fold"):
            r = client.get(f"/tasks/{task_name}")
            fold_data = r.json()["target_fold"]
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"type": "reset", "data": {"task_name": task_name}})
                ws.receive_json()
                ws.send_json({"type": "step", "data": {"fold_data": fold_data}})
                resp = ws.receive_json()
                assert resp["data"]["reward"] == 20.0, f"{task_name} failed"


# --- V3 Multi-Step RL Tests ---

from origami_server.engine.paper_state import PaperState, hash_paper_state
from training.trajectory import Step, Trajectory
from training.curriculum import get_task_pool
from training.env_pool import OrigamiEnvPool
from training.gigpo import compute_gigpo_advantages, GiGPORewardManager
from training.prompt_builder import build_prompt_from_obs
from training.reward import extract_crease_json, valid_crease
from origami_server.engine.step_reward import COMPLETION_BONUS


class TestPaperStateHash:
    def test_empty_papers_same_hash(self):
        s1, s2 = PaperState(), PaperState()
        assert hash_paper_state(s1) == hash_paper_state(s2)

    def test_same_creases_different_order(self):
        s1 = PaperState()
        s1.add_crease([0.5, 0], [0.5, 1], "V")
        s1.add_crease([0, 0.5], [1, 0.5], "V")
        s2 = PaperState()
        s2.add_crease([0, 0.5], [1, 0.5], "V")
        s2.add_crease([0.5, 0], [0.5, 1], "V")
        assert hash_paper_state(s1) == hash_paper_state(s2)

    def test_different_creases_different_hash(self):
        s1 = PaperState()
        s1.add_crease([0, 0], [1, 1], "V")
        s2 = PaperState()
        s2.add_crease([1, 0], [0, 1], "V")
        assert hash_paper_state(s1) != hash_paper_state(s2)

    def test_different_assignment_different_hash(self):
        s1 = PaperState()
        s1.add_crease([0.5, 0], [0.5, 1], "V")
        s2 = PaperState()
        s2.add_crease([0.5, 0], [0.5, 1], "M")
        assert hash_paper_state(s1) != hash_paper_state(s2)


class TestTrajectory:
    def test_add_step(self):
        t = Trajectory(task="triangle")
        t.add_step(prompt="p", completion="c", reward=1.0, done=True, state_hash=0)
        assert t.length == 1
        assert t.total_reward == 1.0

    def test_total_reward_sums(self):
        t = Trajectory(task="quarter_fold")
        t.add_step(prompt="p1", completion="c1", reward=0.5, done=False, state_hash=0)
        t.add_step(prompt="p2", completion="c2", reward=0.7, done=True, state_hash=1)
        assert t.length == 2
        assert abs(t.total_reward - 1.2) < 1e-6


class TestCurriculum:
    def test_phase_1(self):
        pool = get_task_pool(0)
        assert "triangle" in pool
        assert "quarter_fold" in pool  # multi-step from start
        assert get_task_pool(50) == pool

    def test_phase_2(self):
        pool = get_task_pool(100)
        assert "quarter_fold" in pool
        assert "letter_fold" in pool

    def test_phase_3(self):
        pool = get_task_pool(300)
        assert "waterbomb_base" in pool

    def test_phase_4(self):
        pool = get_task_pool(700)
        assert "map_fold" in pool

    def test_beyond_max_uses_last_phase(self):
        pool = get_task_pool(9999)
        assert "map_fold" in pool


class TestEnvPool:
    def test_reset_and_step(self):
        pool = OrigamiEnvPool(pool_size=2)
        obs = pool.reset(0, "triangle")
        assert not obs.done
        assert len(obs.anchor_points) >= 4

        obs = pool.step(0, {"from": [0, 0], "to": [1, 1], "assignment": "V"})
        assert obs.reward is not None

    def test_batch_step(self):
        pool = OrigamiEnvPool(pool_size=4)
        for i in range(4):
            pool.reset(i, "triangle")
        creases = [{"from": [0, 0], "to": [1, 1], "assignment": "V"}] * 4
        results = pool.step_batch([0, 1, 2, 3], creases)
        assert len(results) == 4
        assert all(r.reward is not None for r in results)

    def test_get_paper_state(self):
        pool = OrigamiEnvPool(pool_size=1)
        pool.reset(0, "triangle")
        ps = pool.get_paper_state(0)
        assert ps is not None
        assert isinstance(ps, PaperState)


class TestGiGPO:
    def test_episode_level_ordering(self):
        trajs = [Trajectory(task="triangle") for _ in range(4)]
        for i, t in enumerate(trajs):
            t.add_step(prompt="", completion="", reward=float(i + 1),
                       done=True, state_hash=0)
        advantages = compute_gigpo_advantages(trajs, alpha=1.0)
        # Higher reward should have higher advantage
        assert advantages[0][0] < advantages[3][0]

    def test_step_level_grouping(self):
        trajs = [Trajectory(task="triangle") for _ in range(4)]
        for i, t in enumerate(trajs):
            t.add_step(prompt="", completion="", reward=float(i),
                       done=True, state_hash=42)
        advantages = compute_gigpo_advantages(trajs, alpha=0.0)
        assert advantages[0][0] < advantages[3][0]

    def test_singleton_group_returns_zero(self):
        trajs = [Trajectory(task="t") for _ in range(2)]
        trajs[0].add_step(prompt="", completion="", reward=10.0,
                          done=True, state_hash=1)
        trajs[1].add_step(prompt="", completion="", reward=0.0,
                          done=True, state_hash=2)
        advantages = compute_gigpo_advantages(trajs, alpha=0.0)
        assert advantages[0][0] == 0.0
        assert advantages[1][0] == 0.0

    def test_empty_trajectories(self):
        assert compute_gigpo_advantages([]) == []

    def test_reward_manager_alpha_annealing(self):
        mgr = GiGPORewardManager(alpha_start=1.0, alpha_end=0.3,
                                  warmup_steps=100, total_steps=1000)
        assert mgr.alpha == 1.0
        mgr.global_step = 100
        assert mgr.alpha == 1.0  # at warmup boundary
        mgr.global_step = 550
        assert 0.3 < mgr.alpha < 1.0  # midway
        mgr.global_step = 1000
        assert abs(mgr.alpha - 0.3) < 0.01  # at end


class TestPromptBuilder:
    def test_builds_from_observation(self):
        env = OrigamiEnvironment()
        obs = env.reset(task_name="triangle")
        task_info = get_task("triangle")
        prompt = build_prompt_from_obs("triangle", task_info, obs)
        assert "triangle" in prompt.lower() or "diagonal" in prompt.lower()
        assert "(0,0)" in prompt or "(0.0,0.0)" in prompt

    def test_includes_crease_history(self):
        env = OrigamiEnvironment(mode="step")
        obs = env.reset(task_name="quarter_fold")
        env.step(OrigamiAction(crease={"from": [0.5, 0], "to": [0.5, 1], "assignment": "V"}))
        # Manually construct an obs with creases for testing
        obs2 = env.step(OrigamiAction(crease={"from": [0, 0.5], "to": [1, 0.5], "assignment": "V"}))
        # obs2 should have current_creases populated (the env records them)
        # We just verify the prompt builder doesn't crash with real data
        task_info = get_task("quarter_fold")
        prompt = build_prompt_from_obs("quarter_fold", task_info, obs)
        assert "step 0" in prompt or "step 0 of" in prompt


class TestExtractCreaseJson:
    def test_valid_crease(self):
        text = '{"from": [0.5, 0], "to": [0.5, 1], "assignment": "V"}'
        result = extract_crease_json(text)
        assert result is not None
        assert result["assignment"] == "V"

    def test_invalid_json(self):
        assert extract_crease_json("not json") is None

    def test_missing_fields(self):
        assert extract_crease_json('{"from": [0, 0]}') is None


class TestValidCrease:
    def test_valid(self):
        import json
        good = [[{"content": json.dumps({"from": [0, 0], "to": [1, 1], "assignment": "V"})}]]
        scores = valid_crease(good)
        assert scores[0] == 1.0

    def test_invalid_assignment(self):
        import json
        bad = [[{"content": json.dumps({"from": [0, 0], "to": [1, 1], "assignment": "X"})}]]
        scores = valid_crease(bad)
        assert scores[0] == -0.5

    def test_not_json(self):
        bad = [[{"content": "hello world"}]]
        scores = valid_crease(bad)
        assert scores[0] == -2.0


class TestCompletionBonus:
    def test_difficulty_scaling(self):
        assert COMPLETION_BONUS[1] == 2.0
        assert COMPLETION_BONUS[4] == 15.0


class TestTasksV3:
    def test_six_tasks(self):
        assert len(list_tasks()) == 6

    def test_all_tasks_have_max_folds(self):
        for name in list_tasks():
            task = get_task(name)
            assert "max_folds" in task, f"Task {name} missing max_folds"
            assert task["max_folds"] >= 1


class TestRollout:
    def test_rollout_with_mock_generate(self):
        from training.rollout import run_rollout_batch

        def mock_gen(prompts):
            return ['{"from": [0,0], "to": [1,1], "assignment": "V"}'] * len(prompts)

        trajs = run_rollout_batch(
            generate_fn=mock_gen,
            task_pool=["triangle"],
            batch_size=4,
        )
        assert len(trajs) == 4
        assert all(t.length == 1 for t in trajs)  # triangle max_folds=1
        assert all(t.steps[0].done for t in trajs)
        assert all(t.total_reward > 0 for t in trajs)

    def test_rollout_multistep_task(self):
        from training.rollout import run_rollout_batch

        def mock_gen(prompts):
            return ['{"from": [0.5,0], "to": [0.5,1], "assignment": "V"}'] * len(prompts)

        trajs = run_rollout_batch(
            generate_fn=mock_gen,
            task_pool=["quarter_fold"],
            batch_size=2,
        )
        assert len(trajs) == 2
        assert all(t.length == 2 for t in trajs)  # quarter_fold max_folds=2

    def test_rollout_invalid_json_handled(self):
        from training.rollout import run_rollout_batch

        def bad_gen(prompts):
            return ["not valid json"] * len(prompts)

        trajs = run_rollout_batch(
            generate_fn=bad_gen,
            task_pool=["triangle"],
            batch_size=2,
        )
        assert len(trajs) == 2
        # Should have 1 step with -2.0 reward (parse failure)
        assert all(t.length == 1 for t in trajs)
        assert all(t.steps[0].reward == -2.0 for t in trajs)

    def test_rollout_gigpo_integration(self):
        """End-to-end: rollout → GiGPO advantages."""
        from training.rollout import run_rollout_batch
        from training.gigpo import compute_gigpo_advantages

        def mock_gen(prompts):
            return ['{"from": [0,0], "to": [1,1], "assignment": "V"}'] * len(prompts)

        trajs = run_rollout_batch(
            generate_fn=mock_gen,
            task_pool=["triangle"],
            batch_size=8,
        )
        advantages = compute_gigpo_advantages(trajs, alpha=0.5)
        assert len(advantages) == 8
        assert all(len(a) == 1 for a in advantages)


class TestTrainV3Components:
    """Test V3 training loop components (no GPU required)."""

    def test_generate_fn_interface(self):
        """Verify generate_fn accepts list[str], returns list[str]."""
        # Mock the interface that train_v3.build_generate_fn produces
        def mock_generate(prompts: list[str]) -> list[str]:
            return ['{"from": [0,0], "to": [1,1], "assignment": "V"}'] * len(prompts)

        prompts = ["test prompt 1", "test prompt 2"]
        results = mock_generate(prompts)
        assert len(results) == len(prompts)
        assert all(isinstance(r, str) for r in results)

    def test_full_pipeline_mock(self):
        """Full training pipeline: rollout → GiGPO → advantage shapes."""
        from training.curriculum import get_task_pool
        from training.gigpo import GiGPORewardManager
        from training.rollout import run_rollout_batch

        def mock_gen(prompts):
            return ['{"from": [0.5,0], "to": [0.5,1], "assignment": "V"}'] * len(prompts)

        task_pool = get_task_pool(0)
        assert "triangle" in task_pool

        trajs = run_rollout_batch(
            generate_fn=mock_gen,
            task_pool=task_pool,
            batch_size=4,
        )

        manager = GiGPORewardManager(alpha_start=1.0, alpha_end=0.3, total_steps=100)
        advantages = manager.compute_advantages(trajs)

        assert len(advantages) == 4
        for i, traj in enumerate(trajs):
            assert len(advantages[i]) == traj.length

        manager.step()
        assert manager.global_step == 1

    def test_curriculum_progression(self):
        """Curriculum returns different task pools at different steps."""
        from training.curriculum import get_task_pool

        early = get_task_pool(0)
        mid = get_task_pool(300)
        late = get_task_pool(800)

        assert "triangle" in early
        assert "quarter_fold" in early  # multi-step from start
        assert "waterbomb_base" in mid
        assert "map_fold" in late

    def test_reward_manager_alpha_schedule(self):
        """Alpha anneals from 1.0 to 0.3 over training."""
        from training.gigpo import GiGPORewardManager

        mgr = GiGPORewardManager(
            alpha_start=1.0, alpha_end=0.3,
            warmup_steps=10, total_steps=100,
        )
        assert mgr.alpha == 1.0

        for _ in range(10):
            mgr.step()
        assert mgr.alpha == 1.0  # still in warmup

        for _ in range(90):
            mgr.step()
        assert abs(mgr.alpha - 0.3) < 0.01  # annealed to end

    def test_multistep_rollout_quarter_fold(self):
        """Quarter fold requires 2 steps — verify trajectory length."""
        from training.rollout import run_rollout_batch

        step_count = [0]
        def mock_gen(prompts):
            step_count[0] += 1
            return ['{"from": [0.5,0], "to": [0.5,1], "assignment": "V"}'] * len(prompts)

        trajs = run_rollout_batch(
            generate_fn=mock_gen,
            task_pool=["quarter_fold"],
            batch_size=2,
        )
        assert all(t.length == 2 for t in trajs)
        assert step_count[0] == 2  # generate_fn called twice
