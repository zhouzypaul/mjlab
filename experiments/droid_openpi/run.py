"""Run the original OpenPI pi05_droid velocity policy in the tabletop scene."""

import contextlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import imageio.v3 as iio
import mujoco
import mujoco.viewer
import numpy as np
import tyro
from mjviser import Viewer

from experiments.droid_openpi.client import OpenpiClient
from experiments.droid_openpi.policy_adapter import (
  DROID_CONTROL_HZ,
  DroidPolicyAdapter,
  InferencePolicy,
)
from experiments.droid_openpi.visualizer import PolicyCameraPanels
from mjlab.scene import Scene
from mjlab.tasks.manipulation.config.droid.cube_scene import get_droid_cube_scene_cfg
from mjlab.viewer.viser.shading import set_glb_flat_shading


@dataclass
class DroidInferenceConfig:
  host: str = "localhost"
  port: int = 8000
  timeout: float = 300.0
  """Connection/inference timeout, allowing for initial JAX compilation."""
  prompt: str = "pick up the red cube"
  external_camera: Literal["left", "right"] = "left"
  viewer: Literal["viser", "native", "none"] = "viser"
  max_steps: int = 600
  """15 Hz control steps; 600 steps is 40 simulated seconds."""
  open_loop_horizon: int = 8
  """Actions to execute before querying again, capped by returned chunk length."""
  height: int = 360
  width: int = 640
  output_dir: Path = Path("logs/droid_pi05")


class DroidRollout:
  """Schedule 15 Hz commands without accumulating physics-step rounding drift."""

  def __init__(
    self,
    cfg: DroidInferenceConfig,
    adapter: DroidPolicyAdapter,
    policy: InferencePolicy,
    data: mujoco.MjData,
    *,
    on_observation: Callable[[dict[str, Any], float], None] | None = None,
    on_reset: Callable[[], None] | None = None,
  ) -> None:
    self.cfg = cfg
    self.adapter = adapter
    self.policy = policy
    self.on_observation = on_observation
    self.on_reset = on_reset
    self.cube_id = adapter.model.body("cube/cube").id
    self.reset(adapter.model, data)

  def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if hasattr(self, "control_steps") and self.control_steps > 0:
      self.save(data)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    self.control_steps = 0
    self.inference_calls = 0
    self.chunk = np.empty((0, 8), dtype=np.float32)
    self.chunk_index = 0
    self.timestamps: list[float] = []
    self.actions: list[np.ndarray] = []
    self.joint_positions: list[np.ndarray] = []
    self.gripper_positions: list[float] = []
    self.cube_positions: list[np.ndarray] = []
    self.output_dir = self.cfg.output_dir / f"run_{uuid4().hex[:8]}"
    self.output_dir.mkdir(parents=True, exist_ok=True)
    (self.output_dir / "server_metadata.json").write_text(
      json.dumps(getattr(self.policy, "metadata", {}), indent=2)
    )
    if self.on_reset is not None:
      self.on_reset()

  def done(self, data: mujoco.MjData) -> bool:
    return data.time + 1e-9 >= self.cfg.max_steps / DROID_CONTROL_HZ

  def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if self.done(data):
      return
    if data.time + 1e-9 >= self.control_steps / DROID_CONTROL_HZ:
      if self.chunk_index >= min(self.cfg.open_loop_horizon, len(self.chunk)):
        observation = self.adapter.observation(data, self.cfg.prompt)
        if self.on_observation is not None:
          self.on_observation(observation, float(data.time))
        start = time.monotonic()
        result = self.policy.infer(observation)
        chunk = np.asarray(result.get("actions"), dtype=np.float32)
        if (
          chunk.ndim != 2
          or chunk.shape[1] != 8
          or len(chunk) == 0
          or not np.all(np.isfinite(chunk))
        ):
          raise ValueError("OpenPI must return a nonempty finite action chunk [N, 8]")
        self.chunk = chunk
        self.chunk_index = 0
        np.savez_compressed(
          self.output_dir / f"inference_{self.inference_calls:04d}.npz",
          exterior_rgb=observation["observation/exterior_image_1_left"],
          wrist_rgb=observation["observation/wrist_image_left"],
          joint_position=observation["observation/joint_position"],
          gripper_position=observation["observation/gripper_position"],
          actions=chunk,
          simulation_time=data.time,
        )
        if self.inference_calls == 0:
          iio.imwrite(
            self.output_dir / "initial_views.png",
            np.concatenate(
              (
                observation["observation/exterior_image_1_left"],
                observation["observation/wrist_image_left"],
              ),
              axis=1,
            ),
          )
        self.inference_calls += 1
        print(
          f"t={data.time:.2f}s: chunk {chunk.shape}, "
          f"inference {time.monotonic() - start:.2f}s",
          flush=True,
        )
      applied = self.adapter.apply_action(data, self.chunk[self.chunk_index])
      self.timestamps.append(data.time)
      self.actions.append(applied)
      self.joint_positions.append(data.qpos[self.adapter.arm_qpos_ids].copy())
      self.gripper_positions.append(self.adapter.gripper_closure(data))
      self.cube_positions.append(data.xpos[self.cube_id].copy())
      self.chunk_index += 1
      self.control_steps += 1
    mujoco.mj_step(model, data)

  def save(self, data: mujoco.MjData) -> None:
    observation = self.adapter.observation(data, self.cfg.prompt)
    iio.imwrite(
      self.output_dir / "final_views.png",
      np.concatenate(
        (
          observation["observation/exterior_image_1_left"],
          observation["observation/wrist_image_left"],
        ),
        axis=1,
      ),
    )
    np.savez_compressed(
      self.output_dir / "rollout.npz",
      time=np.asarray(self.timestamps),
      actions=np.asarray(self.actions),
      joint_position=np.asarray(self.joint_positions),
      gripper_position=np.asarray(self.gripper_positions),
      cube_position=np.asarray(self.cube_positions),
    )
    final_z = float(data.xpos[self.cube_id, 2])
    summary = {
      "prompt": self.cfg.prompt,
      "external_camera": self.cfg.external_camera,
      "control_steps": self.control_steps,
      "inference_calls": self.inference_calls,
      "simulation_seconds": float(data.time),
      "max_cube_center_height": max([final_z, *[p[2] for p in self.cube_positions]]),
      "final_cube_center_height": final_z,
    }
    (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Rollout saved to {self.output_dir.resolve()}", flush=True)
    print(
      f"Cube center: final z={final_z:.3f}m, "
      f"max z={summary['max_cube_center_height']:.3f}m",
      flush=True,
    )


def run_droid_inference(cfg: DroidInferenceConfig) -> None:
  if cfg.max_steps <= 0 or cfg.open_loop_horizon <= 0 or cfg.timeout <= 0:
    raise ValueError("Step count, open-loop horizon, and timeout must be positive")
  scene = Scene(
    get_droid_cube_scene_cfg(camera_height=cfg.height, camera_width=cfg.width),
    device="cpu",
  )
  model = scene.compile()
  data = mujoco.MjData(model)
  uri = f"ws://{cfg.host}:{cfg.port}"
  print(f"Connecting to pi05_droid server at {uri}", flush=True)
  with contextlib.ExitStack() as stack:
    policy = stack.enter_context(contextlib.closing(OpenpiClient(uri, cfg.timeout)))
    adapter = stack.enter_context(
      contextlib.closing(
        DroidPolicyAdapter(
          model,
          external_camera=cfg.external_camera,
          height=cfg.height,
          width=cfg.width,
        )
      )
    )
    rollout = DroidRollout(cfg, adapter, policy, data)
    print(f"Task: {cfg.prompt}", flush=True)
    try:
      if cfg.viewer == "viser":
        viewer = Viewer(model, data, step_fn=rollout.step, reset_fn=rollout.reset)
        set_glb_flat_shading(viewer.scene.server, "/bodies/cube/cube/group0")
        panels = PolicyCameraPanels(viewer.scene.server, cfg.external_camera)
        rollout.on_observation = panels.update
        rollout.on_reset = panels.reset
        viewer.run()
      elif cfg.viewer == "native":
        with mujoco.viewer.launch_passive(model, data) as viewer:
          viewer.opt.geomgroup[3] = 0
          viewer.cam.lookat[:] = (0.4, 0.0, 0.25)
          viewer.cam.distance = 1.8
          while viewer.is_running():
            start = time.monotonic()
            rollout.step(model, data)
            viewer.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.monotonic() - start)))
      else:
        while not rollout.done(data):
          rollout.step(model, data)
    finally:
      rollout.save(data)


def main() -> None:
  run_droid_inference(tyro.cli(DroidInferenceConfig))


if __name__ == "__main__":
  main()
