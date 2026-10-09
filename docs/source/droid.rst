DROID simulation
================

The DROID asset combines the Menagerie Franka Panda arm and Robotiq 2F-85
gripper, including the gripper's tendon and four-bar linkage constraints.
The tabletop scene includes a movable cube, two exterior cameras, and a wrist
camera that moves with the gripper.

Run from the repository root:

.. code-block:: bash

   uv run scripts/demos/droid.py

Open the Viser URL printed in the terminal. Use the actuator controls to command
the seven arm joints in radians and the gripper with normalized closure:
``0`` is open and ``1`` is closed. Joint-position sliders directly change the
simulation state; actuator sliders drive the physical mechanism.
Pass ``--animate`` to move the arm gently and cycle the gripper automatically.
Use ``--viewer native`` for the MuJoCo desktop viewer.

To run without a viewer and export a portable MJCF scene, RGB images, and metric
depth arrays:

.. code-block:: bash

   uv run scripts/demos/droid.py --viewer none --animate --duration 5 \
     --output-dir /tmp/droid --height 720 --width 1280

On machines without EGL, prefix that command with ``MUJOCO_GL=osmesa`` (requires
the system OSMesa library). Rendering uses the visual meshes and hides collision
meshes. Depth files are NumPy arrays in meters. The exported ``scene.xml`` and
``assets/`` directory can be loaded by MuJoCo independently of mjlab and Menagerie.

Using the asset in mjlab
-----------------------

.. code-block:: python

   from mjlab.asset_zoo.robots import get_droid_robot_cfg
   from mjlab.tasks.manipulation.config.droid.cube_scene import get_droid_cube_scene_cfg
   from mjlab.scene import Scene

   robot_cfg = get_droid_robot_cfg()
   scene_cfg = get_droid_cube_scene_cfg(num_envs=16, camera_sensors=True)
   scene = Scene(scene_cfg, device="cuda")
   model = scene.compile()

Camera elements always exist; ``camera_sensors=True`` also wires RGB and depth
into mjlab's sensor system. The default resolution is 320 by 180. Use
``camera_width=1280, camera_height=720`` for RoboLab's resolution. Keep the 16:9
aspect ratio to retain both horizontal and vertical fields of view.

The robot asset contains the mechanism and mounted wrist camera. Workspaces
live in ``mjlab.tasks.manipulation.config.droid``: ``cube_scene.py`` defines
the table, cube, and exterior cameras. Add other workspaces as sibling
``<name>_scene.py`` modules with their own scene configuration factories.
The standalone demo lives in ``scripts/demos/droid.py``. OpenPI inference code
lives separately in ``experiments/droid_openpi``; see the top-level
``DROID_sim.md`` for model-server and inference commands.

For batched simulation, use ``SimulationCfg`` with
``MujocoCfg(timestep=0.002, integrator="implicitfast", cone="elliptic", impratio=10)``.
The arm uses XML position actuators and the gripper uses a tendon transmission.
Set gripper targets via ``robot.set_tendon_position_target`` for ``robotiq_split``;
its target is normalized closure, rather than a physical tendon length.
If using an action manager, select the tendon transmission and disable the
default position offset for the gripper.

Frames and cameras
------------------

The robot base is at the origin; +X points into the workspace and +Z points up.
The table surface is at z=0 and the floor is at z=-0.697 m. The table is a simple
box fixture with legs, rather than RoboLab's USD table mesh.

The exterior camera positions are (0.05, +0.57, 0.66) and
(0.05, -0.57, 0.66) m, with RoboLab's OpenGL orientations. The wrist camera
uses RoboLab's calibrated 2.8 focal length and 5.376 by 3.024 aperture; the
exterior cameras use focal length 2.1 with the same aperture. Camera names are
``table/over_shoulder_left_camera``, ``table/over_shoulder_right_camera``, and
``robot/wrist_cam``. These are monocular views of the three-camera preset.

RoboLab's Robotiq ``base_link`` points +X along the fingers, while Menagerie's
gripper base points +Z along them. The wrist pose is converted between these
frames, including its translation. The mount places the gripper base 0.125174 m
along Panda link7's +Z, as measured from RoboLab's USD.

The wrist adapter matches the PolaRiS/RoboLab cylinder: 75 mm diameter,
18 mm thickness, centered 9 mm beyond the Panda flange (116 mm along link7's
+Z). It has separate visual and collision geoms, rigidly attached to link7.
It overlaps Menagerie's gripper mounting plate; MuJoCo excludes contacts
between these welded parts. The existing Menagerie link inertias are retained.

The robot exposes ``gripper_base`` (RoboLab base_link axes), ``eef_frame``
(RoboLab's rotated control frame), and ``grasp_site`` (the fingertip pinch center).
Scene site names have the ``robot/`` prefix. The home arm pose follows RoboLab.
Arm actuator gains, force limits, gripper dynamics, and self-collision behavior
come from Menagerie; the simulation does not reproduce Isaac Sim dynamics or
an individually calibrated physical DROID rig.

Asset provenance
----------------

The unmodified XML files and referenced mesh assets were copied from local
MuJoCo Menagerie commit ``0059d4335f8156206f63a35662313385f7ad6d74``:

* `Franka Emika Panda <https://github.com/google-deepmind/mujoco_menagerie/tree/0059d4335f8156206f63a35662313385f7ad6d74/franka_emika_panda>`_
  (Apache-2.0; license included under ``xmls/panda/LICENSE``).
* `Robotiq 2F-85 <https://github.com/google-deepmind/mujoco_menagerie/tree/0059d4335f8156206f63a35662313385f7ad6d74/robotiq_2f85>`_
  (BSD-2-Clause; license included under ``xmls/robotiq/LICENSE``).

Assembly, normalized gripper control, home keyframe, sites, and cameras are
applied in Python, leaving those source XML files intact. Camera poses,
intrinsics, and control frames are based on RoboLab commit
``ad45d4f974725d020f82c2b0d77d78533aeba2b3``:

* `DROID configuration <https://github.com/NVlabs/RoboLab/blob/ad45d4f974725d020f82c2b0d77d78533aeba2b3/robolab/robots/droid.py>`_.
* `Camera presets <https://github.com/NVlabs/RoboLab/blob/ad45d4f974725d020f82c2b0d77d78533aeba2b3/robolab/variations/camera.py>`_.
* `Robot USD <https://github.com/NVlabs/RoboLab/blob/ad45d4f974725d020f82c2b0d77d78533aeba2b3/assets/robots/franka_robotiq_2f_85_flattened.usd>`_.
