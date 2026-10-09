"""DROID cube observations and velocity actions for the pi05_droid experiment."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Protocol

import mujoco
import numpy as np
from PIL import Image

# DROID RobotIKSolver: normalized joint_velocity_to_delta multiplies by 0.2.
# This is radians per 15 Hz command, rather than radians per second.
DROID_CONTROL_HZ = 15
DROID_MAX_JOINT_DELTA = 0.2
DROID_MAX_GRIPPER_WIDTH = 0.085


class InferencePolicy(Protocol):
  def infer(self, observation: dict[str, Any]) -> dict[str, Any]: ...


def resize_with_pad(image: np.ndarray) -> np.ndarray:
  """Match OpenPI's bilinear 224x224 RGB resize and centered black padding."""
  source = Image.fromarray(image)
  width, height = source.size
  ratio = max(width / 224, height / 224)
  resized = source.resize(
    (int(width / ratio), int(height / ratio)), Image.Resampling.BILINEAR
  )
  result = Image.new("RGB", (224, 224), 0)
  result.paste(resized, ((224 - resized.width) // 2, (224 - resized.height) // 2))
  return np.asarray(result)


class DroidPolicyAdapter:
  def __init__(
    self,
    model: mujoco.MjModel,
    *,
    external_camera: Literal["left", "right"] = "left",
    height: int = 360,
    width: int = 640,
  ) -> None:
    self.model = model
    joints = [model.joint(f"robot/joint{i}") for i in range(1, 8)]
    self.arm_qpos_ids = np.array([joint.qposadr[0] for joint in joints])
    self.arm_ctrl_ids = np.array(
      [model.actuator(f"robot/actuator{i}").id for i in range(1, 8)]
    )
    self.gripper_ctrl_id = model.actuator("robot/robotiq_fingers_actuator").id
    self._base_id = model.body("robot/robotiq_base").id
    self._pad_ids = [
      model.geom(f"robot/robotiq_{side}_pad1").id for side in ("left", "right")
    ]
    self._external_camera = f"table/over_shoulder_{external_camera}_camera"
    self._options = mujoco.MjvOption()
    self._options.geomgroup[2] = 1
    self._options.geomgroup[3] = 0
    self._options.sitegroup[:] = 0
    # EGL contexts must stay on their owning thread. Viser invokes Reset and
    # paused Step on GUI workers, while normal steps run on the main thread.
    self._render_thread = ThreadPoolExecutor(
      max_workers=1, thread_name_prefix="droid-render"
    )
    self._closed = False
    try:
      self._renderer = self._render_thread.submit(
        mujoco.Renderer, model, height=height, width=width
      ).result()
    except BaseException:
      self._render_thread.shutdown(wait=True)
      raise

  def gripper_closure(self, data: mujoco.MjData) -> float:
    """Read pad separation: DROID measures 1 - width / maximum_width."""
    axis = data.xmat[self._base_id].reshape(3, 3)[:, 1]
    left, right = self._pad_ids
    width = abs((data.geom_xpos[left] - data.geom_xpos[right]) @ axis)
    for pad in self._pad_ids:
      projected_axes = data.geom_xmat[pad].reshape(3, 3).T @ axis
      width -= np.abs(projected_axes) @ self.model.geom_size[pad]
    return float(np.clip(1 - width / DROID_MAX_GRIPPER_WIDTH, 0.0, 1.0))

  def observation(self, data: mujoco.MjData, prompt: str) -> dict[str, Any]:
    mujoco.mj_forward(self.model, data)
    # Wait for rendering before returning, so the caller keeps its simulation
    # lock held and cannot advance/reset data while the renderer reads it.
    images = self._render_thread.submit(self._render_images, data).result()
    return {
      "observation/exterior_image_1_left": images[0],
      "observation/wrist_image_left": images[1],
      "observation/joint_position": data.qpos[self.arm_qpos_ids].astype(np.float32),
      "observation/gripper_position": np.array(
        [self.gripper_closure(data)], dtype=np.float32
      ),
      "prompt": prompt,
    }

  def _render_images(self, data: mujoco.MjData) -> list[np.ndarray]:
    images = []
    for camera in (self._external_camera, "robot/wrist_cam"):
      self._renderer.update_scene(data, camera=camera, scene_option=self._options)
      images.append(resize_with_pad(self._renderer.render()))
    return images

  def apply_action(self, data: mujoco.MjData, action: np.ndarray) -> np.ndarray:
    if action.shape != (8,) or not np.all(np.isfinite(action)):
      raise ValueError("DROID action must contain eight finite values")
    applied = np.clip(action, -1.0, 1.0).copy()
    applied[7] = float(action[7] > 0.5)
    # Match DROID's measured-position-relative command. The MuJoCo position
    # actuators approximate the hardware impedance controller for each interval.
    targets = data.qpos[self.arm_qpos_ids] + applied[:7] * DROID_MAX_JOINT_DELTA
    limits = self.model.actuator_ctrlrange[self.arm_ctrl_ids]
    data.ctrl[self.arm_ctrl_ids] = np.clip(targets, limits[:, 0], limits[:, 1])
    data.ctrl[self.gripper_ctrl_id] = applied[7]
    return applied

  def close(self) -> None:
    if self._closed:
      return
    try:
      self._render_thread.submit(self._renderer.close).result()
    finally:
      self._closed = True
      self._render_thread.shutdown(wait=True)
