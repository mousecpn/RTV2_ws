"""Headless batch adapter for the shared-control scene in ``sim_env.py``."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pybullet as p
import pybullet_data as pd
from scipy.spatial.transform import Rotation


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GOAL_POSITIONS = np.array(
    [[6.0, -7.0, 0.0], [8.5, -3.5, 0.0], [9.2, 0.0, 0.0], [8.5, 3.5, 0.0], [6.0, 7.0, 0.0]],
    dtype=np.float32,
)
WORKSPACE_LIMITS = np.asarray([[0.0, 10.0], [-9.0, 9.0]], dtype=np.float32)


class SharedControlBatchEnv:
    """The ``sim_env.py`` five-target scene without keyboard or ZMQ setup."""

    success_radius = 1.0
    collision_radius = 1.3

    def __init__(self, control_dt: float = 0.1, gui: bool = False, scene_layout: str = "fixed_five_target"):
        if scene_layout not in {"fixed_five_target", "random_occluded"}:
            raise ValueError(f"Unknown scene layout: {scene_layout}")
        self.control_dt = control_dt
        self.gui = gui
        self.scene_layout = scene_layout
        connection_mode = p.GUI if gui else p.DIRECT
        self.client_id = p.connect(connection_mode)
        p.setAdditionalSearchPath(pd.getDataPath(), physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client_id)
        if gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=18,
                cameraYaw=-90,
                cameraPitch=-70,
                cameraTargetPosition=[5, 0, 0],
                physicsClientId=self.client_id,
            )
        self.flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.init_orn = [0.0, 0.0, 0.0, 1.0]
        p.loadURDF("plane.urdf", [0, 0, -0.6], physicsClientId=self.client_id)
        self.avatar = p.loadURDF(
            "sphere_small.urdf",
            [0, 0, 0],
            self.init_orn,
            useFixedBase=True,
            flags=self.flags,
            globalScaling=20,
            physicsClientId=self.client_id,
        )
        self.goal_ids: list[int] = []
        self.obstacle_ids: list[int] = []
        self.true_goal_idx: int | None = None

    def reset(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        self.clear_table()
        p.resetBasePositionAndOrientation(self.avatar, [0, 0, 0], self.init_orn, physicsClientId=self.client_id)
        if self.scene_layout == "fixed_five_target":
            shuffled_goals, obstacle_positions = self._fixed_scene(rng)
        else:
            shuffled_goals, obstacle_positions = self._random_occluded_scene(rng)
        self.goal_ids = []
        for index, goal_position in enumerate(shuffled_goals):
            urdf = "true_goal_small.urdf" if index == 0 else "goal_small.urdf"
            goal_id = p.loadURDF(
                str(REPOSITORY_ROOT / "objects" / urdf),
                goal_position,
                self.init_orn,
                useFixedBase=True,
                flags=self.flags,
                globalScaling=30,
                physicsClientId=self.client_id,
            )
            self.goal_ids.append(goal_id)
            if index == 0:
                self.true_goal_idx = goal_id

        self.obstacle_ids = []
        for obstacle_position in obstacle_positions:
            obstacle_id = p.loadURDF(
                str(REPOSITORY_ROOT / "objects" / "obstacle_small.urdf"),
                obstacle_position,
                self.init_orn,
                useFixedBase=True,
                flags=self.flags,
                globalScaling=30,
                physicsClientId=self.client_id,
            )
            self.obstacle_ids.append(obstacle_id)
        return shuffled_goals[:, :2].copy(), obstacle_positions[:, :2].copy()

    def _fixed_scene(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        shuffled_goals = GOAL_POSITIONS[rng.permutation(len(GOAL_POSITIONS))]
        true_goal = shuffled_goals[0]
        distance_scale = rng.uniform(1.6, 2.4)
        return shuffled_goals, (true_goal / distance_scale).reshape(1, 3)

    def _random_occluded_scene(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Reproduce the commented random object setup in sim_env.py."""
        num_goals = rng.integers(2, 4)
        num_obstacles = rng.integers(3, 6)
        banned_radius = 3.0
        obstacle_extension = 3.0
        occupied = np.empty((0, 2), dtype=np.float32)
        obstacles = []

        while len(obstacles) < num_obstacles:
            candidate = rng.uniform(WORKSPACE_LIMITS[:, 0], WORKSPACE_LIMITS[:, 1])
            if occupied.size and np.linalg.norm(candidate - occupied, axis=1).min() < 1.0:
                continue
            if banned_radius < np.linalg.norm(candidate) < banned_radius + obstacle_extension:
                obstacles.append(candidate)
                occupied = np.vstack((occupied, candidate))

        goals = []
        while len(goals) < num_goals:
            candidate = rng.uniform(WORKSPACE_LIMITS[:, 0], WORKSPACE_LIMITS[:, 1])
            if occupied.size and np.linalg.norm(candidate - occupied, axis=1).min() < 1.0:
                continue
            if np.linalg.norm(candidate) > banned_radius + obstacle_extension and self._is_goal_occluded(obstacles, candidate):
                goals.append(candidate)
                occupied = np.vstack((occupied, candidate))

        return np.column_stack((np.asarray(goals), np.zeros(num_goals))), np.column_stack((np.asarray(obstacles), np.zeros(num_obstacles)))

    @staticmethod
    def _is_goal_occluded(obstacles: list[np.ndarray], goal: np.ndarray) -> bool:
        """Use the original sim_env.py axis-aligned occlusion test."""
        lower = np.minimum(np.zeros(2), goal)
        upper = np.maximum(np.zeros(2), goal)
        return any(np.all((lower <= obstacle) & (obstacle <= upper)) for obstacle in obstacles)

    def position(self) -> np.ndarray:
        position, _ = p.getBasePositionAndOrientation(self.avatar, physicsClientId=self.client_id)
        return np.asarray(position[:2], dtype=np.float32)

    def move(self, planar_velocity: np.ndarray) -> None:
        """Kinematically advance the fixed-base avatar at the environment control rate."""
        position, orientation = p.getBasePositionAndOrientation(self.avatar, physicsClientId=self.client_id)
        next_position = np.asarray(position, dtype=np.float32)
        next_position[:2] += self.control_dt * np.asarray(planar_velocity, dtype=np.float32)
        p.resetBasePositionAndOrientation(self.avatar, next_position, orientation, physicsClientId=self.client_id)
        if self.gui:
            time.sleep(self.control_dt)

    def is_collision(self) -> bool:
        position = self.position()
        return any(
            np.linalg.norm(position - np.asarray(p.getBasePositionAndOrientation(obstacle_id, physicsClientId=self.client_id)[0][:2]))
            < self.collision_radius
            for obstacle_id in self.obstacle_ids
        )

    def is_true_goal_achieved(self) -> bool:
        assert self.true_goal_idx is not None
        goal_position, _ = p.getBasePositionAndOrientation(self.true_goal_idx, physicsClientId=self.client_id)
        return np.linalg.norm(self.position() - np.asarray(goal_position[:2])) < self.success_radius

    def clear_table(self) -> None:
        for body_id in self.goal_ids + self.obstacle_ids:
            p.removeBody(body_id, physicsClientId=self.client_id)
        self.goal_ids = []
        self.obstacle_ids = []

    def close(self) -> None:
        self.clear_table()
        p.disconnect(self.client_id)