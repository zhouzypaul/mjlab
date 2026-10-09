# DROID simulation and pi0.5 inference

Run commands from the mjlab repository root unless a different directory is
shown. The simulator uses a Menagerie Franka Panda and Robotiq 2F-85, a table,
and a red cube. The simulation runs on CPU; the OpenPI model runs in a separate
GPU process. This keeps OpenPI's JAX/Python/NumPy dependencies separate from
mjlab's dependencies.

## Code layout and adding scenes

- `src/mjlab/asset_zoo/robots/droid/` contains the robot assembly, mounted wrist
  camera, XML files, meshes, and licenses.
- `src/mjlab/tasks/manipulation/config/droid/cube_scene.py` defines the cube
  workspace: table, movable cube, exterior cameras, and physics settings.
- `scripts/demos/droid.py` runs or exports that workspace without a policy.
- `experiments/droid_openpi/` contains the inference runner, policy observation
  and action adapter, and OpenPI transport client. It runs from this checkout
  and is separate from the installed `mjlab` package.
- `tests/test_droid.py` covers the simulation; `tests/experiments/` covers
  experimental integrations.

Import the cube workspace explicitly:

```python
from mjlab.tasks.manipulation.config.droid.cube_scene import get_droid_cube_scene_cfg

scene_cfg = get_droid_cube_scene_cfg()
```

Add future DROID workspaces as sibling `<name>_scene.py` modules in the same
task directory, each with its own `get_droid_<name>_scene_cfg` factory. Keep
object placement, exterior camera layouts, and scene settings in those modules;
share the robot asset across them. Scene modules should not import experiments.
The demo and OpenPI experiment currently select the cube workspace explicitly.

## Install and run the simulator

```sh
uv sync --extra cpu --group dev
uv run --extra cpu scripts/demos/droid.py
```

Open the Viser URL printed in the terminal, normally `http://localhost:8080`.
Actuator controls command seven arm joint angles in radians and gripper closure
from `0` (open) to `1` (closed). Joint-position sliders directly change state.

```sh
# Move the arm and cycle the gripper automatically.
uv run --extra cpu scripts/demos/droid.py --animate

# Use the native MuJoCo desktop viewer.
uv run --extra cpu scripts/demos/droid.py --viewer native

# Run headlessly and export MJCF, RGB images, and depth arrays in meters.
uv run --extra cpu scripts/demos/droid.py --viewer none --animate \
  --duration 5 --output-dir /tmp/droid --height 720 --width 1280
```

Linux defaults to EGL for camera rendering. If EGL is unavailable, prefix a
command with `MUJOCO_GL=osmesa`; this requires the system OSMesa library.
The exported `scene.xml` and `assets/` directory are portable MuJoCo assets.
See [the model guide](docs/source/droid.rst) for frames and asset provenance.
The wrist includes PolaRiS/RoboLab's 75 mm diameter, 18 mm thick cylindrical
gripper adapter, with visual and collision geometry. The gripper and wrist
camera retain their calibrated placement.

## Start the pi0.5-DROID model server

Use the original `pi05_droid` configuration and checkpoint. This checkpoint
predicts joint **velocity** commands; joint-position fine-tuned DROID models
need a different action adapter.

On the machine with the NVIDIA GPU, install OpenPI separately following its
[installation instructions](https://github.com/Physical-Intelligence/openpi#installation):

```sh
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git ~/repos/openpi
cd ~/repos/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 uv run scripts/serve_policy.py \
  --port 8000 policy:checkpoint \
  --policy.config pi05_droid \
  --policy.dir gs://openpi-assets/checkpoints/pi05_droid
```

Wait for the server to report that it is listening on port 8000. The first run
downloads roughly 12.4 GB of checkpoint files, normally cached under
`~/.cache/openpi`. `OPENPI_DATA_HOME` changes that cache directory. The initial
inference request also compiles JAX kernels. The simulator allows 300 seconds
for an inference response by default. An RTX 4090 with 24 GB is sufficient for
the verified setup; a working NVIDIA driver is required.

For an existing local checkpoint, replace the `gs://` path with its directory.
The server can run locally or on another machine. There is no need to install
the OpenPI model or `openpi-client` into the mjlab environment: the experiment
implements the same NumPy MessagePack/WebSocket protocol. Its client dependencies
are provided by the `droid-openpi` dependency group, enabled in the commands below.
The `dev` group also includes them so development checks cover the experiment.

### Local setup used during this session

The live 4090 test used an isolated OpenPI checkout at `/tmp/mjlab-openpi`
and the downloaded checkpoint at `/tmp/mjlab-pi05-droid`. The server was started
on port 8000. To restart that setup without downloading the checkpoint again:

```sh
cd /tmp/mjlab-openpi
UV_NO_SYNC=1 OPENPI_DATA_HOME=/tmp/mjlab-openpi-data \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 uv run scripts/serve_policy.py \
  --port 8000 policy:checkpoint \
  --policy.config pi05_droid --policy.dir /tmp/mjlab-pi05-droid
```

These `/tmp` directories are temporary. For a permanent installation, use the
checkout and cache locations in the instructions above.

## Run "pick up the red cube"

In a second terminal, return to the mjlab repository:

```sh
uv run --extra cpu --group droid-openpi -m experiments.droid_openpi.run \
  --host localhost --port 8000 \
  --prompt "pick up the red cube"
```

The default prompt is already `pick up the red cube`. The browser viewer lets
you watch the policy move the robot. For a finite rollout without a viewer:

```sh
uv run --extra cpu --group droid-openpi -m experiments.droid_openpi.run \
  --viewer none --max-steps 600 --open-loop-horizon 8 \
  --prompt "pick up the red cube"
```

`600` control steps cover 40 simulated seconds at 15 Hz. Eight actions from
each chunk are executed before requesting another chunk. The simulator pauses
while waiting for inference, so network latency does not execute stale actions.
The physical integration uses 2 ms steps with control boundaries rounded by
less than one physics step, without accumulating timing drift.

Useful options:

| Option | Meaning |
| --- | --- |
| `--host HOST --port PORT` | Connect to a remote or local OpenPI server. |
| `--external-camera left` | Use the left shoulder camera; this is the default. |
| `--external-camera right` | Use the right shoulder camera instead. |
| `--viewer native` | Watch in the MuJoCo desktop viewer. |
| `--viewer none` | Exit automatically when `--max-steps` is reached. |
| `--open-loop-horizon 1` | Request new feedback on every control step. |
| `--height 720 --width 1280` | Render source cameras at RoboLab resolution. |
| `--output-dir logs/droid_pi05` | Parent directory for rollout records. |

Interactive viewers hold the final state when the step budget is exhausted.
Exit the viewer (Ctrl+C for Viser) to finish writing the final record. Reset in
the viewer starts a new episode and clears the cached action chunk.

## Model observations

Each request follows OpenPI's `DroidInputs` and DROID runtime:

| Request key | Value |
| --- | --- |
| `observation/exterior_image_1_left` | One exterior RGB image, `uint8`, `(224, 224, 3)`. |
| `observation/wrist_image_left` | Wrist RGB image, `uint8`, `(224, 224, 3)`. |
| `observation/joint_position` | Seven measured Panda joint angles in joint1–joint7 order, `float32`, `(7,)`, radians. |
| `observation/gripper_position` | Measured normalized closure, `float32`, `(1,)`, open=0, closed=1. |
| `prompt` | The task instruction string. |

The source cameras render at 640×360 by default. They are resized bilinearly
and padded with black to 224×224 without changing aspect ratio, matching
OpenPI's image helper. MuJoCo already returns RGB, so there is no BGR conversion.
Collision meshes and debug sites are excluded from camera images.

Only **one** exterior view and the wrist view are sent. `left` in the request
key refers to the left image of a stereo pair; it does not require choosing the
left shoulder camera. The right shoulder view can occupy the same request key.
The camera elements are `table/over_shoulder_left_camera`,
`table/over_shoulder_right_camera`, and `robot/wrist_cam`.

The gripper observation is derived from actual fingertip pad separation:
`closure = clip(1 - width / 0.085, 0, 1)`. It is not the commanded actuator
value: the fingers may remain partly open while holding an object.
No Cartesian pose, joint velocity, depth image, or second exterior image is
required. State normalization and the masked unused image slot are handled by
the OpenPI server.

## Actions and simulation control

The current `pi05_droid` configuration returns `actions` with shape `(15, 8)`:

| Columns | Meaning |
| --- | --- |
| `0:7` | Normalized Panda joint-velocity commands, joint1–joint7. |
| `7` | Absolute normalized gripper closure. |

The client accepts finite `(N, 8)` chunks and replans after the smaller of `N`
and `--open-loop-horizon`. It does not assume the older example's 10-step
horizon, and it rejects malformed or non-finite actions.

Following OpenPI's DROID runtime, arm commands are clipped to `[-1, 1]` and
gripper output is binarized at `>0.5`. Following DROID's `RobotIKSolver`, each
normalized arm command becomes a target relative to the **measured** position:

```text
joint_target = measured_joint_position + 0.2 * clipped_joint_velocity
```

The target is clipped to the Panda joint limits and held for the 15 Hz control
interval. These velocities are not absolute joint angles or Cartesian deltas,
and they are not raw rad/s. The existing MuJoCo position actuators approximate
the hardware's impedance-controller response. The physical Robotiq tendon and
linkage receive the binary closure target.

## Rollout records and limits

Each episode creates a new `run_...` directory under `logs/droid_pi05`:

- `initial_views.png` and `final_views.png`: exterior and wrist model inputs.
- `inference_0000.npz`, etc.: images, measured state, raw action chunk, and time
  for every policy request; load with `numpy.load`.
- `rollout.npz`: control times, applied actions, joint positions, measured
  gripper closure, and cube positions.
- `summary.json`: prompt, camera, step count, inference count, and observed cube
  center heights in meters.
- `server_metadata.json`: the upstream server's metadata.

This is inference from a real-robot checkpoint in a simplified simulation.
The input/action contract is matched, but camera appearance, contacts, and
controller dynamics differ from the training hardware. Sending the prompt
does not guarantee a successful grasp. Inspect the camera records and cube
trajectory to assess the rollout; cube center height at rest is about 0.025 m.

Live verification on the local 4090 returned `(15, 8)` chunks at approximately
80 ms per request after compilation. Two 600-step runs, with open-loop horizons
of eight and one, approached the cube but did not lift it. Their maximum cube
center height was 0.027 m, including its initial settling height. This verifies
the inference/control connection, not successful task completion.

## Upstream references

The integration was checked against OpenPI commit
`215abfb217dbac7d5f1273282331b9b1866c0479`:

- [DROID inference instructions](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/README.md).
- [Input/output transforms](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/droid_policy.py).
- [pi05_droid configuration and 15-step horizon](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py).
- [DROID runtime: 15 Hz, velocity actions, binary gripper](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/main.py).
- [Original checkpoint uses velocity actions](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/README_train.md).
- [DROID velocity-to-position scaling](https://github.com/droid-dataset/droid/blob/main/droid/robot_ik/robot_ik_solver.py).
