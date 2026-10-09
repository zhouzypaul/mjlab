"""Menagerie DROID model with RoboLab camera and end-effector frames."""

import math
from pathlib import Path

import mujoco
import numpy as np

from mjlab.actuator import XmlActuatorCfg
from mjlab.actuator.actuator import TransmissionType
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_XML_DIR = Path(__file__).parent / "xmls"
DROID_HOME = (0.0, -math.pi / 5, 0.0, -4 * math.pi / 5, 0.0, 3 * math.pi / 5, 0.0)
DROID_CAMERA_NAMES = (
  "over_shoulder_left_camera",
  "over_shoulder_right_camera",
  "wrist_cam",
)

# RoboLab's base_link uses +X along the fingers; Menagerie uses +Z.
# Rotation taking vectors in RoboLab base_link into Menagerie robotiq_base.
_BASE_LINK_QUAT = (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)


def _load_spec(directory: str, filename: str) -> mujoco.MjSpec:
  """Embed meshes with distinct paths so attachment and export are portable."""
  root = _XML_DIR / directory
  spec = mujoco.MjSpec.from_file(str(root / filename))
  assets = {}
  for mesh in spec.meshes:
    path = root / "assets" / mesh.file
    mesh.file = f"{directory}/{path.name}"
    assets[mesh.file] = path.read_bytes()
  spec.meshdir = ""
  spec.assets = assets
  return spec


def get_spec() -> mujoco.MjSpec:
  """Build the arm, physical gripper linkage, and moving wrist camera."""
  spec = _load_spec("panda", "panda_nohand.xml")
  gripper = _load_spec("robotiq", "2f85.xml")
  spec.modelname = "droid"
  spec.option.timestep = 0.002
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  for key in list(spec.keys):
    spec.delete(key)
  for light in list(spec.lights):
    spec.delete(light)

  # RoboLab USD: gripper base is 0.125174 m along link7's +Z.
  # Menagerie base_mount + base offsets contribute 0.0108 m of that distance.
  mount = spec.body("attachment")
  assert mount is not None
  mount.pos[:] = (0.0, 0.0, 0.125174 - 0.0108)
  mount.quat[:] = (math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5))
  spec.attach(gripper, site=spec.site("attachment_site"), prefix="robotiq_")

  base = spec.body("robotiq_base")
  assert base is not None
  base.add_site(name="gripper_base", quat=_BASE_LINK_QUAT, size=(0.005,) * 3)
  eef_quat = np.empty(4)
  mujoco.mju_mulQuat(
    eef_quat, np.array(_BASE_LINK_QUAT), np.array((0.5, -0.5, 0.5, -0.5))
  )
  base.add_site(name="eef_frame", quat=eef_quat.tolist(), size=(0.005,) * 3)
  base.add_site(name="grasp_site", pos=(0.0, 0.0, 0.145), size=(0.005,) * 3)

  # Both simulators use OpenGL cameras: local -Z looks forward, +Y points up.
  wrist_quat = np.empty(4)
  mujoco.mju_mulQuat(
    wrist_quat, np.array(_BASE_LINK_QUAT), np.array((-0.420, 0.570, 0.576, -0.409))
  )
  base.add_camera(
    name="wrist_cam",
    pos=(0.074, -0.031, 0.011),
    quat=wrist_quat.tolist(),
    fovy=math.degrees(2 * math.atan(3.024 / (2 * 2.8))),
  )

  # Normalize Menagerie's [0, 255] control to DROID's [0, 1] closure command.
  gripper_actuator = spec.actuator("robotiq_fingers_actuator")
  assert gripper_actuator is not None
  gripper_actuator.ctrlrange[:] = (0.0, 1.0)
  gripper_actuator.gainprm[0] = 80.0

  spec.add_key(name="home", qpos=(*DROID_HOME, *([0.0] * 8)), ctrl=(*DROID_HOME, 0.0))
  return spec


def _get_entity_spec() -> mujoco.MjSpec:
  spec = get_spec()
  # Entity attachment does not transfer global options. Set those on the scene
  # (or SimulationCfg.mujoco) instead, and avoid attachment conflict warnings.
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
  spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
  spec.option.impratio = 1.0
  return spec


def get_droid_robot_cfg() -> EntityCfg:
  """Fresh configuration retaining the Menagerie position and tendon actuators.

  Arm targets are joint angles in radians. The gripper tendon position target
  is normalized closure (0=open, 1=closed), rather than a tendon length.
  """
  return EntityCfg(
    spec_fn=_get_entity_spec,
    init_state=EntityCfg.InitialStateCfg(joint_pos=None),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        XmlActuatorCfg(target_names_expr=("joint[1-7]",)),
        XmlActuatorCfg(
          target_names_expr=("robotiq_split",),
          transmission_type=TransmissionType.TENDON,
          command_field="position",
        ),
      ),
    ),
  )
