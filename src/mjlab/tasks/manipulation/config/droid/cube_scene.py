"""DROID cube workspace, with a table and the RoboLab three-camera preset."""

import math

import mujoco

from mjlab.asset_zoo.robots.droid.droid_constants import get_droid_robot_cfg
from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensorCfg

DROID_CUBE_CAMERA_NAMES = (
  "over_shoulder_left_camera",
  "over_shoulder_right_camera",
  "wrist_cam",
)


def _get_table_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table")
  # Robot-root coordinates: table surface at z=0, floor at z=-0.697.
  body.add_geom(
    name="tabletop",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.45, 0.0, -0.025),
    size=(0.75, 0.65, 0.025),
    rgba=(0.65, 0.55, 0.42, 1.0),
  )
  for x in (-0.2, 1.1):
    for y in (-0.55, 0.55):
      body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(x, y, -0.3735),
        size=(0.035, 0.035, 0.3235),
        rgba=(0.25, 0.25, 0.28, 1.0),
      )
  _add_external_cameras(body)
  return spec


def _get_cube_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.025,) * 3,
    mass=0.05,
    friction=(1.0, 0.005, 0.0001),
    rgba=(0.8, 0.15, 0.1, 1.0),
  )
  return spec


def _edit_scene(spec: mujoco.MjSpec) -> None:
  spec.option.timestep = 0.002
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  spec.option.impratio = 10
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    pos=(0.0, 0.0, -0.697),
    size=(3.0, 3.0, 0.1),
    rgba=(0.2, 0.25, 0.3, 1.0),
  )
  spec.worldbody.add_light(pos=(0.5, 0.0, 2.0), dir=(0.0, 0.0, -1.0))


def _add_external_cameras(body: mujoco.MjsBody) -> None:
  # RoboLab's OpenGL quaternion convention is identical to MuJoCo's.
  for name, pos, quat in (
    (
      "over_shoulder_left_camera",
      (0.05, 0.57, 0.66),
      (-0.393, -0.195, 0.399, 0.805),
    ),
    (
      "over_shoulder_right_camera",
      (0.05, -0.57, 0.66),
      (0.805, 0.399, -0.195, -0.393),
    ),
  ):
    body.add_camera(
      name=name,
      pos=pos,
      quat=quat,
      fovy=math.degrees(2 * math.atan(3.024 / (2 * 2.1))),
    )


def get_droid_cube_scene_cfg(
  *,
  num_envs: int = 1,
  camera_height: int = 180,
  camera_width: int = 320,
  camera_sensors: bool = False,
) -> SceneCfg:
  """Build a scene usable with native MuJoCo or mjlab's batched Simulation.

  Camera elements always exist. Enable camera_sensors to also create mjlab RGB
  and depth sensors; the smaller default resolution keeps batched rendering cheap.
  Use height=720 and width=1280 to match RoboLab's image resolution.
  """
  if camera_height <= 0 or camera_width <= 0:
    raise ValueError("Camera dimensions must be positive")
  sensors = (
    tuple(
      CameraSensorCfg(
        name=name,
        camera_name=f"{'robot' if name == 'wrist_cam' else 'table'}/{name}",
        height=camera_height,
        width=camera_width,
        data_types=("rgb", "depth"),
        enabled_geom_groups=(0, 2),
      )
      for name in DROID_CUBE_CAMERA_NAMES
    )
    if camera_sensors
    else ()
  )
  return SceneCfg(
    num_envs=num_envs,
    env_spacing=3.0,
    entities={
      "robot": get_droid_robot_cfg(),
      "table": EntityCfg(spec_fn=_get_table_spec),
      "cube": EntityCfg(
        spec_fn=_get_cube_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0.5, 0.0, 0.027)),
      ),
    },
    sensors=sensors,
    spec_fn=_edit_scene,
  )
