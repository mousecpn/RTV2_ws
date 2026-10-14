"""Fixed-five-target PyBullet scene for late-BCI-influence experiments."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pybullet as p
import pybullet_data as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GOALS = np.asarray(
    [[6.0, -7.0, 0.0], [8.5, -3.5, 0.0], [9.2, 0.0, 0.0], [8.5, 3.5, 0.0], [6.0, 7.0, 0.0]],
    dtype=np.float32,
)


class LateInfluenceEnv:
    success_radius = 1.0
    collision_radius = 1.3

    def __init__(self, control_dt: float = 0.1, gui: bool = False):
        self.control_dt = control_dt
        self.gui = gui
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
        self.true_goal_id: int | None = None

    def reset(self, system_seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(system_seed)
        self.clear_table()
        p.resetBasePositionAndOrientation(self.avatar, [0, 0, 0], self.init_orn, physicsClientId=self.client_id)
        goals = GOALS[rng.permutation(len(GOALS))]
        obstacle = (goals[0] / rng.uniform(1.6, 2.4)).reshape(1, 3)
        for index, goal in enumerate(goals):
            path = "true_goal_small.urdf" if index == 0 else "goal_small.urdf"
            body_id = p.loadURDF(str(REPOSITORY_ROOT / "objects" / path), goal, self.init_orn, useFixedBase=True, flags=self.flags, globalScaling=30, physicsClientId=self.client_id)
            self.goal_ids.append(body_id)
            if index == 0:
                self.true_goal_id = body_id
        body_id = p.loadURDF(str(REPOSITORY_ROOT / "objects" / "obstacle_small.urdf"), obstacle[0], self.init_orn, useFixedBase=True, flags=self.flags, globalScaling=30, physicsClientId=self.client_id)
        self.obstacle_ids.append(body_id)
        return goals[:, :2].copy(), obstacle[:, :2].copy()

    def position(self) -> np.ndarray:
        position, _ = p.getBasePositionAndOrientation(self.avatar, physicsClientId=self.client_id)
        return np.asarray(position[:2], dtype=np.float32)

    def move(self, action: np.ndarray) -> None:
        position, orientation = p.getBasePositionAndOrientation(self.avatar, physicsClientId=self.client_id)
        next_position = np.asarray(position, dtype=np.float32)
        next_position[:2] += self.control_dt * np.asarray(action, dtype=np.float32)
        p.resetBasePositionAndOrientation(self.avatar, next_position, orientation, physicsClientId=self.client_id)
        if self.gui:
            time.sleep(self.control_dt)

    def terminal_outcome(self) -> str | None:
        position = self.position()
        if any(np.linalg.norm(position - np.asarray(p.getBasePositionAndOrientation(body_id, physicsClientId=self.client_id)[0][:2])) < self.collision_radius for body_id in self.obstacle_ids):
            return "collision"
        assert self.true_goal_id is not None
        target, _ = p.getBasePositionAndOrientation(self.true_goal_id, physicsClientId=self.client_id)
        if np.linalg.norm(position - np.asarray(target[:2])) < self.success_radius:
            return "success"
        return None

    def clear_table(self) -> None:
        for body_id in self.goal_ids + self.obstacle_ids:
            p.removeBody(body_id, physicsClientId=self.client_id)
        self.goal_ids, self.obstacle_ids, self.true_goal_id = [], [], None

    def close(self) -> None:
        self.clear_table()
        p.disconnect(self.client_id)