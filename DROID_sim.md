# DROID simulation and pi0.5 inference

Run commands from the mjlab repository root unless shown otherwise. The scene
contains a Franka Panda, Robotiq 2F-85, table, and red cube. Physics runs on CPU;
camera rendering uses the GPU through EGL on Linux. OpenPI runs separately on
an NVIDIA GPU.

## Run the simulator

```sh
uv sync --extra cpu --group dev
uv run --extra cpu scripts/demos/droid.py

# Animate the arm and gripper.
uv run --extra cpu scripts/demos/droid.py --animate

# Use the native MuJoCo viewer instead of Viser.
uv run --extra cpu scripts/demos/droid.py --viewer native

# Export MJCF, RGB images, and depth arrays without a viewer.
uv run --extra cpu scripts/demos/droid.py --viewer none --animate \
  --duration 5 --output-dir /tmp/droid --height 720 --width 1280
```

Open the Viser URL printed in the terminal, usually `http://localhost:8080`.
Arm controls use radians; gripper closure is `0` (open) to `1` (closed).
If EGL is unavailable, prefix commands with `MUJOCO_GL=osmesa` for CPU
rendering; this requires the system OSMesa library.

## Start the model server

Install OpenPI in its own environment on the GPU machine:

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

Use the original `pi05_droid` velocity-action checkpoint. Joint-position
fine-tuned models require a different adapter. Wait for the server to listen
on port 8000; the first launch downloads about 12.4 GB, and the first inference
request compiles JAX kernels.

To restart the existing local 4090 setup using its downloaded checkpoint:

```sh
cd /tmp/mjlab-openpi
UV_NO_SYNC=1 OPENPI_DATA_HOME=/tmp/mjlab-openpi-data \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 uv run scripts/serve_policy.py \
  --port 8000 policy:checkpoint \
  --policy.config pi05_droid --policy.dir /tmp/mjlab-pi05-droid
```

The `/tmp` setup is temporary; use the installation above for a permanent setup.

## Run "pick up the red cube"

In a second terminal, from the mjlab repository root:

```sh
uv run --extra cpu --group droid-openpi -m experiments.droid_openpi.run \
  --host localhost --port 8000 \
  --prompt "pick up the red cube"

# Run a finite rollout without a viewer.
uv run --extra cpu --group droid-openpi -m experiments.droid_openpi.run \
  --viewer none --max-steps 600 --open-loop-horizon 8 \
  --prompt "pick up the red cube"
```

Viser's **Policy camera inputs** folder displays the exact 224×224 exterior
and wrist images sent to OpenPI. Images update on each policy request. **Reset**
starts a new episode and clears the images and cached actions. Use **Ctrl+C**
to exit Viser and finish saving the rollout.

Useful options:

| Option | Purpose |
| --- | --- |
| `--host HOST --port PORT` | Connect to another model server. |
| `--external-camera right` | Use the right shoulder camera instead of the default left. |
| `--viewer native` / `--viewer none` | Use the desktop viewer / run headlessly. |
| `--open-loop-horizon 1` | Request feedback every control step instead of every eight. |
| `--height 720 --width 1280` | Render source images at RoboLab resolution. |
| `--output-dir logs/droid_pi05` | Set the parent directory for rollout records. |

The policy receives one exterior image, the wrist image, seven measured joint
angles, measured gripper closure, and the prompt. It returns 15×8 action chunks:
seven normalized joint-velocity commands and gripper closure. Control runs at
15 Hz, so 600 steps cover 40 simulated seconds. The simulator pauses while
waiting for inference.

Each episode saves camera images, inference/state/action arrays, and a summary
under `logs/droid_pi05/run_...`. The real-robot checkpoint can run in this
simplified scene, but successful grasping is not guaranteed; verified rollouts
approached the cube without lifting it.

## Code layout

- `src/mjlab/asset_zoo/robots/droid/`: robot assets and mounted wrist camera.
- `src/mjlab/tasks/manipulation/config/droid/cube_scene.py`: cube workspace,
  exterior cameras, and physics settings.
- `scripts/demos/droid.py`: standalone simulator and export commands.
- `experiments/droid_openpi/`: policy adapter, client, inference runner, and UI.

Add future scenes beside `cube_scene.py` as `<name>_scene.py`, with a
`get_droid_<name>_scene_cfg` factory. The current runners explicitly use the cube
scene. Keep scene definitions separate from experimental integrations.

See the [model guide](docs/source/droid.rst) for asset details and
[OpenPI's DROID guide](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/droid/README.md)
for the upstream policy interface.
