"""PyBullet environment for variable candidate-target experiments."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pybullet as p
import pybullet_data as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_LIMITS = np.asarray([[0.0, 10.0], [-9.0, 9.0]], dtype=np.float32)


class CandidateTargetEnv:
    """Headless or GUI scene containing one true target and distractor targets."""

    success_radius = 1.0
    min_target_distance = 1.0
    min_start_distance = 6.0

    def __init__(self, control_dt: float = 0.1, gui: bool = False):
        self.control_dt = control_dt
        self.gui = gui
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
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
        self.target_ids: list[int] = []
        self.true_target_id: int | None = None

    def reset(self, seed: int, candidate_count: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        self.clear_targets()
        p.resetBasePositionAndOrientation(self.avatar, [0, 0, 0], self.init_orn, physicsClientId=self.client_id)
        target_positions = self._sample_targets(rng, candidate_count)
        self.target_ids = []
        for index, target_position in enumerate(target_positions):
            urdf = "true_goal_small.urdf" if index == 0 else "goal_small.urdf"
            target_id = p.loadURDF(
                str(REPOSITORY_ROOT / "objects" / urdf),
                target_position,
                self.init_orn,
                useFixedBase=True,
                flags=self.flags,
                globalScaling=30,
                physicsClientId=self.client_id,
            )
            self.target_ids.append(target_id)
            if index == 0:
                self.true_target_id = target_id
        return target_positions[:, :2].copy()

    def _sample_targets(self, rng: np.random.Generator, candidate_count: int) -> np.ndarray:
        targets: list[np.ndarray] = []
        attempts = 0
        max_attempts = 100_000
        while len(targets) < candidate_count:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Could not place {candidate_count} targets with {self.min_target_distance} m spacing")
            candidate = rng.uniform(WORKSPACE_LIMITS[:, 0], WORKSPACE_LIMITS[:, 1])
            if np.linalg.norm(candidate) <= self.min_start_distance:
                continue
            if targets and np.linalg.norm(candidate - np.asarray(targets), axis=1).min() < self.min_target_distance:
                continue
            targets.append(candidate)
        return np.column_stack((np.asarray(targets), np.zeros(candidate_count)))

    def position(self) -> np.ndarray:
        position, _ = p.getBasePositionAndOrientation(self.avatar, physicsClientId=self.client_id)
        return np.asarray(position[:2], dtype=np.float32)

    def move(self, planar_velocity: np.ndarray) -> None:
        position, orientation = p.getBasePositionAndOrientation(self.avatar, physicsClientId=self.client_id)
        next_position = np.asarray(position, dtype=np.float32)
        next_position[:2] += self.control_dt * np.asarray(planar_velocity, dtype=np.float32)
        p.resetBasePositionAndOrientation(self.avatar, next_position, orientation, physicsClientId=self.client_id)
        if self.gui:
            time.sleep(self.control_dt)

    def reached_target_index(self) -> int | None:
        position = self.position()
        for index, target_id in enumerate(self.target_ids):
            target_position, _ = p.getBasePositionAndOrientation(target_id, physicsClientId=self.client_id)
            if np.linalg.norm(position - np.asarray(target_position[:2])) < self.success_radius:
                return index
        return None

    def clear_targets(self) -> None:
        for target_id in self.target_ids:
            p.removeBody(target_id, physicsClientId=self.client_id)
        self.target_ids = []
        self.true_target_id = None

    def close(self) -> None:
        self.clear_targets()
        p.disconnect(self.client_id)