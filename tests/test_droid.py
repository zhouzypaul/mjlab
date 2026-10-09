"""Check the assembled DROID mechanism, camera frames, and portable scene."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.droid.droid_constants import DROID_HOME, get_spec
from mjlab.asset_zoo.robots.droid.scene import get_droid_scene_cfg
from mjlab.scene import Scene


def test_droid_gripper_closes_and_reopens() -> None:
  model = get_spec().compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  drivers = [
    model.joint(f"robotiq_{side}_driver_joint").qposadr[0] for side in ("left", "right")
  ]
  assert model.nu == 8
  for closure in (0.0, 1.0, 0.0):
    data.ctrl[7] = closure
    for _ in range(1000):
      mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))
    assert not np.any(data.warning.number)
    assert np.allclose(data.qpos[:7], DROID_HOME, atol=0.02)
    assert abs(data.qpos[drivers[0]] - data.qpos[drivers[1]]) < 0.005
    if closure:
      assert np.all(data.qpos[drivers] > 0.7)
    else:
      assert np.all(data.qpos[drivers] < 0.01)


def test_droid_camera_frames_and_export(tmp_path: Path) -> None:
  scene = Scene(get_droid_scene_cfg(camera_sensors=True), device="cpu")
  model = scene.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  left = model.camera("table/over_shoulder_left_camera").id
  right = model.camera("table/over_shoulder_right_camera").id
  wrist = model.camera("robot/wrist_cam").id
  base = model.site("robot/gripper_base").id
  link7 = model.body("robot/link7").id
  assert model.ncam == 3
  assert len(scene.sensors) == 3
  assert np.allclose(data.cam_xpos[left], (0.05, 0.57, 0.66))
  assert np.allclose(data.cam_xpos[right], (0.05, -0.57, 0.66))
  base_rot = data.site_xmat[base].reshape(3, 3)
  assert np.allclose(
    base_rot.T @ (data.cam_xpos[wrist] - data.site_xpos[base]),
    (0.011, -0.031, -0.074),
    atol=1e-6,
  )
  # USD-derived mount frame: +X points along link7 +Z, +Y along link7 -Y.
  assert np.allclose(
    data.xmat[link7].reshape(3, 3).T @ base_rot,
    ((0, 0, 1), (0, -1, 0), (1, 0, 0)),
    atol=1e-6,
  )
  assert model.cam_fovy[wrist] == pytest.approx(56.7381, abs=0.01)
  wrist_pos = data.cam_xpos[wrist].copy()
  data.qpos[0] += 0.3
  mujoco.mj_forward(model, data)
  assert not np.allclose(data.cam_xpos[wrist], wrist_pos)
  assert np.allclose(data.cam_xpos[left], (0.05, 0.57, 0.66))
  scene.write(tmp_path)
  exported = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
  assert (exported.nq, exported.nu, exported.ncam) == (22, 8, 3)
