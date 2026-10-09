"""Run the DROID tabletop simulation on CPU, with optional camera snapshots."""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import imageio.v3 as iio
import mujoco
import mujoco.viewer
import numpy as np
import tyro
from mjviser import Viewer

from mjlab.asset_zoo.robots.droid.droid_constants import DROID_CAMERA_NAMES
from mjlab.asset_zoo.robots.droid.scene import get_droid_scene_cfg
from mjlab.scene import Scene


@dataclass
class DroidConfig:
  viewer: Literal["viser", "native", "none"] = "viser"
  """Browser viewer, native MuJoCo viewer, or a finite headless run."""
  duration: float = 5.0
  """Simulation seconds for a headless run."""
  animate: bool = False
  """Move the arm gently and cycle the gripper (overrides control sliders)."""
  output_dir: Path | None = None
  """Write a portable scene and final RGB/depth images for all three cameras."""
  height: int = 180
  width: int = 320


def run_droid(cfg: DroidConfig) -> None:
  if cfg.duration <= 0:
    raise ValueError("Duration must be positive")
  scene = Scene(
    get_droid_scene_cfg(camera_height=cfg.height, camera_width=cfg.width),
    device="cpu",
  )
  model = scene.compile()
  data = mujoco.MjData(model)

  def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

  reset(model, data)

  def step(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if cfg.animate:
      data.ctrl[:7] = model.key_ctrl[0, :7]
      data.ctrl[0] += 0.15 * math.sin(data.time)
      data.ctrl[6] += 0.2 * math.sin(data.time)
      data.ctrl[7] = 0.5 * (1 - math.cos(math.pi * data.time / 2))
    mujoco.mj_step(model, data)

  if cfg.viewer == "viser":
    Viewer(model, data, step_fn=step, reset_fn=reset).run()
  elif cfg.viewer == "native":
    with mujoco.viewer.launch_passive(model, data) as viewer:
      viewer.opt.geomgroup[2] = 1
      viewer.opt.geomgroup[3] = 0
      viewer.cam.lookat[:] = (0.4, 0.0, 0.25)
      viewer.cam.distance = 1.8
      viewer.cam.azimuth = 135
      viewer.cam.elevation = -25
      while viewer.is_running():
        start = time.monotonic()
        step(model, data)
        viewer.sync()
        time.sleep(max(0.0, model.opt.timestep - (time.monotonic() - start)))
  else:
    for _ in range(math.ceil(cfg.duration / model.opt.timestep)):
      step(model, data)
    print(f"Simulated {data.time:.3f}s; {model.nu} actuators, {model.ncam} cameras")

  if cfg.output_dir is not None:
    scene.write(cfg.output_dir)
    mujoco.mj_forward(model, data)
    options = mujoco.MjvOption()
    options.geomgroup[2] = 1  # Menagerie visual meshes.
    options.geomgroup[3] = 0  # Hide overlapping collision meshes.
    options.sitegroup[:] = 0  # Keep debug sites out of camera observations.
    with mujoco.Renderer(model, height=cfg.height, width=cfg.width) as renderer:
      for name in DROID_CAMERA_NAMES:
        camera = f"{'robot' if name == 'wrist_cam' else 'table'}/{name}"
        renderer.update_scene(data, camera=camera, scene_option=options)
        iio.imwrite(cfg.output_dir / f"{name}.png", renderer.render())
        renderer.enable_depth_rendering()
        np.save(cfg.output_dir / f"{name}_depth.npy", renderer.render())
        renderer.disable_depth_rendering()
    print(f"Wrote scene and camera images to {cfg.output_dir.resolve()}")


def main() -> None:
  run_droid(tyro.cli(DroidConfig))


if __name__ == "__main__":
  main()
