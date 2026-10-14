"""Fixed-seed PyBullet scene for BCI-noise robustness experiments."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pybullet as p
import pybullet_data as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_LIMITS = np.asarray([[0.0, 10.0], [-9.0, 9.0]], dtype=np.float32)
FIXED_GOAL_POSITIONS = np.asarray(
    [[6.0, -7.0, 0.0], [8.5, -3.5, 0.0], [9.2, 0.0, 0.0], [8.5, 3.5, 0.0], [6.0, 7.0, 0.0]],
    dtype=np.float32,
)


class NoiseRobustnessEnv:
    """Fixed-seed shared-control scene for BCI-noise robustness experiments."""

    success_radius = 1.0
    collision_radius = 1.3

    def __init__(self, control_dt: float = 0.1, gui: bool = False, scene_layout: str = "random_occluded"):
        if scene_layout not in {"fixed_five_target", "random_occluded"}:
            raise ValueError(f"Unknown scene layout: {scene_layout}")
        self.control_dt = control_dt
        self.gui = gui
        self.scene_layout = scene_layout
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pd.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        if gui:
            p.resetDebugVisualizerCamera(18, -90, -70, [5, 0, 0], physicsClientId=self.client_id)
        self.flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.init_orn = [0.0, 0.0, 0.0, 1.0]
        p.loadURDF("plane.urdf", [0, 0, -0.6], physicsClientId=self.client_id)
        self.avatar = p.loadURDF("sphere_small.urdf", [0, 0, 0], self.init_orn, useFixedBase=True, flags=self.flags, globalScaling=20, physicsClientId=self.client_id)
        self.goal_ids: list[int] = []
        self.obstacle_ids: list[int] = []
        self.true_goal_idx: int | None = None

    def reset(self, system_seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(system_seed)
        self.clear_table()
        p.resetBasePositionAndOrientation(self.avatar, [0, 0, 0], self.init_orn, physicsClientId=self.client_id)
        if self.scene_layout == "fixed_five_target":
            goals, obstacles = self._sample_fixed_five_target_scene(rng)
        else:
            goals, obstacles = self._sample_occluded_scene(rng)
        for index, goal in enumerate(goals):
            path = "true_goal_small.urdf" if index == 0 else "goal_small.urdf"
            body_id = p.loadURDF(str(REPOSITORY_ROOT / "objects" / path), goal, self.init_orn, useFixedBase=True, flags=self.flags, globalScaling=30, physicsClientId=self.client_id)
            self.goal_ids.append(body_id)
            if index == 0:
                self.true_goal_idx = body_id
        for obstacle in obstacles:
            body_id = p.loadURDF(str(REPOSITORY_ROOT / "objects" / "obstacle_small.urdf"), obstacle, self.init_orn, useFixedBase=True, flags=self.flags, globalScaling=30, physicsClientId=self.client_id)
            self.obstacle_ids.append(body_id)
        return goals[:, :2].copy(), obstacles[:, :2].copy()

    @staticmethod
    def _sample_fixed_five_target_scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        goals = FIXED_GOAL_POSITIONS[rng.permutation(len(FIXED_GOAL_POSITIONS))]
        obstacle = goals[0] / rng.uniform(1.6, 2.4)
        return goals, obstacle.reshape(1, 3)

    def _sample_occluded_scene(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        occupied = np.empty((0, 2), dtype=np.float32)
        obstacles: list[np.ndarray] = []
        while len(obstacles) < rng.integers(3, 6):
            point = rng.uniform(WORKSPACE_LIMITS[:, 0], WORKSPACE_LIMITS[:, 1])
            if 3.0 < np.linalg.norm(point) < 6.0 and (not occupied.size or np.linalg.norm(point - occupied, axis=1).min() >= 1.0):
                obstacles.append(point)
                occupied = np.vstack((occupied, point))

        goals: list[np.ndarray] = []
        while len(goals) < rng.integers(2, 4):
            point = rng.uniform(WORKSPACE_LIMITS[:, 0], WORKSPACE_LIMITS[:, 1])
            occluded = any(np.all(np.minimum(0.0, point) <= obstacle) and np.all(obstacle <= np.maximum(0.0, point)) for obstacle in obstacles)
            if np.linalg.norm(point) > 6.0 and occluded and np.linalg.norm(point - occupied, axis=1).min() >= 1.0:
                goals.append(point)
                occupied = np.vstack((occupied, point))
        return np.column_stack((np.asarray(goals), np.zeros(len(goals)))), np.column_stack((np.asarray(obstacles), np.zeros(len(obstacles))))

    def position(self) -> np.ndarray:
        position, _ = p.getBasePositionAndOrientation(self.avatar, physicsClientId=self.client_id)
        return np.asarray(position[:2], dtype=np.float32)

    def move(self, velocity: np.ndarray) -> None:
        position, orientation = p.getBasePositionAndOrientation(self.avatar, physicsClientId=self.client_id)
        next_position = np.asarray(position, dtype=np.float32)
        next_position[:2] += self.control_dt * np.asarray(velocity, dtype=np.float32)
        p.resetBasePositionAndOrientation(self.avatar, next_position, orientation, physicsClientId=self.client_id)
        if self.gui:
            time.sleep(self.control_dt)

    def is_collision(self) -> bool:
        return any(np.linalg.norm(self.position() - np.asarray(p.getBasePositionAndOrientation(body_id, physicsClientId=self.client_id)[0][:2])) < self.collision_radius for body_id in self.obstacle_ids)

    def is_true_goal_achieved(self) -> bool:
        assert self.true_goal_idx is not None
        target, _ = p.getBasePositionAndOrientation(self.true_goal_idx, physicsClientId=self.client_id)
        return np.linalg.norm(self.position() - np.asarray(target[:2])) < self.success_radius

    def clear_table(self) -> None:
        for body_id in self.goal_ids + self.obstacle_ids:
            p.removeBody(body_id, physicsClientId=self.client_id)
        self.goal_ids, self.obstacle_ids, self.true_goal_idx = [], [], None

    def close(self) -> None:
        self.clear_table()
        p.disconnect(self.client_id)