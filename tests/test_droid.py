"""Check the assembled DROID mechanism, camera frames, and portable scene."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.droid.droid_constants import DROID_HOME, get_spec
from mjlab.scene import Scene
from mjlab.tasks.manipulation.config.droid.cube_scene import get_droid_cube_scene_cfg


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


def test_droid_cube_resists_table_penetration_under_load() -> None:
  model = Scene(get_droid_cube_scene_cfg(), device="cpu").compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  cube = model.geom("cube/cube_geom").id
  table = model.geom("table/tabletop").id
  # Press on the 50 g cube with an additional 10 N, like a descending gripper.
  data.xfrc_applied[model.body("cube/cube").id, 2] = -10.0
  max_penetration = 0.0
  for _ in range(500):
    mujoco.mj_step(model, data)
    for contact in data.contact:
      if {contact.geom1, contact.geom2} == {cube, table}:
        max_penetration = max(max_penetration, -contact.dist)
  contacts = [c for c in data.contact if {c.geom1, c.geom2} == {cube, table}]
  assert contacts
  assert max_penetration < 0.001
  assert max(-c.dist for c in contacts) < 0.0001
  assert not np.any(data.warning.number)


def test_droid_gripper_blocks_on_cube_and_releases() -> None:
  model = Scene(get_droid_cube_scene_cfg(), device="cpu").compile()
  # Isolate finger/cube contact from falling and table contact.
  model.opt.gravity[:] = 0.0
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  cube = model.geom("cube/cube_geom").id
  cube_qpos = model.joint("cube/cube_joint").qposadr[0]
  data.qpos[cube_qpos : cube_qpos + 3] = data.site("robot/grasp_site").xpos
  for closure in (1.0, 0.0):
    data.ctrl[7] = closure
    for _ in range(500):
      mujoco.mj_step(model, data)
    contacts = [c for c in data.contact if cube in (c.geom1, c.geom2)]
    assert np.all(np.isfinite(data.qpos))
    assert not np.any(data.warning.number)
    if closure:
      touched = {g for c in contacts for g in (c.geom1, c.geom2)}
      for side in ("left", "right"):
        assert model.geom(f"robot/robotiq_{side}_pad1").id in touched
      assert max(-c.dist for c in contacts) < 0.001
    else:
      assert not contacts


def test_droid_mount_matches_reference_and_collides_with_objects() -> None:
  spec = get_spec()
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  link7 = model.body("link7").id
  mount = model.geom("gripper_adapter").id
  collision = model.geom("gripper_adapter_collision").id
  assert model.geom_type[mount] == mujoco.mjtGeom.mjGEOM_CYLINDER
  assert np.allclose(model.geom_size[mount, :2], (0.0375, 0.009))
  assert np.allclose(model.geom_pos[mount], (0.0, 0.0, 0.116))
  assert model.geom_group[mount] == 2
  assert model.geom_contype[mount] == model.geom_conaffinity[mount] == 0
  assert model.geom_group[collision] == 3
  assert model.geom_contype[collision] == model.geom_conaffinity[collision] == 1
  assert model.geom_bodyid[mount] == model.geom_bodyid[collision] == link7
  # All mounting parts are welded; their intentional overlap is not a contact.
  assert (
    model.body_weldid[link7] == model.body_weldid[model.body("robotiq_base_mount").id]
  )
  assert not any(collision in (c.geom1, c.geom2) for c in data.contact)
  rotation = data.xmat[link7].reshape(3, 3)
  base = model.site("gripper_base").id
  assert np.allclose(
    rotation.T @ (data.site_xpos[base] - data.xpos[link7]),
    (0.0, 0.0, 0.125174),
  )
  # Place an object at the adapter's side in the previously empty wrist gap.
  probe_pos = data.xpos[link7] + rotation @ np.array((0.038, 0.0, 0.111))
  spec.worldbody.add_geom(
    name="mount_probe",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    pos=probe_pos.tolist(),
    size=(0.003, 0.0, 0.0),
  )
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  pair = {model.geom("gripper_adapter_collision").id, model.geom("mount_probe").id}
  assert any({c.geom1, c.geom2} == pair and c.dist < 0 for c in data.contact)


def test_droid_camera_frames_and_export(tmp_path: Path) -> None:
  scene = Scene(get_droid_cube_scene_cfg(camera_sensors=True), device="cpu")
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
  for name in ("robot/gripper_adapter", "robot/gripper_adapter_collision"):
    assert np.allclose(exported.geom(name).size, model.geom(name).size)
    assert np.allclose(exported.geom(name).pos, model.geom(name).pos)
