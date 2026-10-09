"""DROID policy observations, velocity commands, chunk timing, and wire format."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import get_ident
from typing import Any

import msgpack
import mujoco
import numpy as np
import pytest

from experiments.droid_openpi import client as openpi
from experiments.droid_openpi.policy_adapter import DroidPolicyAdapter
from experiments.droid_openpi.run import DroidInferenceConfig, DroidRollout
from mjlab.asset_zoo.robots.droid.droid_constants import DROID_HOME
from mjlab.scene import Scene
from mjlab.tasks.manipulation.config.droid.cube_scene import get_droid_cube_scene_cfg


@pytest.fixture
def adapter_and_data(monkeypatch):
  class Renderer:
    def __init__(self, model, height, width):
      self.height = height
      self.width = width
      self.owner = get_ident()

    def update_scene(self, data, camera, scene_option):
      assert get_ident() == self.owner
      self.camera = camera

    def render(self):
      assert get_ident() == self.owner
      color = (0, 255, 0) if self.camera == "robot/wrist_cam" else (255, 0, 0)
      return np.broadcast_to(color, (self.height, self.width, 3)).astype(np.uint8)

    def close(self):
      assert get_ident() == self.owner

  monkeypatch.setattr(mujoco, "Renderer", Renderer)
  model = Scene(get_droid_cube_scene_cfg(), device="cpu").compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)
  adapter = DroidPolicyAdapter(model, height=180, width=320)
  yield adapter, data
  adapter.close()


def test_droid_observation_contract(adapter_and_data):
  adapter, data = adapter_and_data
  obs = adapter.observation(data, "pick up the red cube")
  assert set(obs) == {
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
    "observation/joint_position",
    "observation/gripper_position",
    "prompt",
  }
  for key, color in (
    ("observation/exterior_image_1_left", (255, 0, 0)),
    ("observation/wrist_image_left", (0, 255, 0)),
  ):
    image = obs[key]
    assert image.shape == (224, 224, 3)
    assert image.dtype == np.uint8
    assert np.all(image[:49] == 0)
    assert np.all(image[49:175] == color)
  assert np.allclose(obs["observation/joint_position"], DROID_HOME)
  assert obs["observation/gripper_position"].shape == (1,)
  assert obs["observation/gripper_position"][0] < 0.01
  assert obs["prompt"] == "pick up the red cube"


def test_velocity_action_uses_measured_position_and_binary_gripper(adapter_and_data):
  adapter, data = adapter_and_data
  before = data.qpos[adapter.arm_qpos_ids].copy()
  action = np.array((2.0, -2.0, 0.25, 0.0, -0.25, 0.5, -0.5, 0.6))
  applied = adapter.apply_action(data, action)
  assert np.allclose(
    data.ctrl[adapter.arm_ctrl_ids] - before,
    (0.2, -0.2, 0.05, 0.0, -0.05, 0.1, -0.1),
  )
  assert applied[7] == data.ctrl[adapter.gripper_ctrl_id] == 1.0
  adapter.apply_action(data, np.zeros(8))
  assert np.allclose(data.ctrl[adapter.arm_ctrl_ids], before)
  assert data.ctrl[adapter.gripper_ctrl_id] == 0.0
  # A blocked gripper can be commanded closed but still physically be open.
  assert adapter.gripper_closure(data) < 0.01
  action[1] = np.nan
  with pytest.raises(ValueError, match="finite"):
    adapter.apply_action(data, action)


def test_chunk_replanning_and_exact_control_rate(adapter_and_data, tmp_path: Path):
  adapter, data = adapter_and_data

  class Policy:
    def __init__(self):
      self.calls = 0

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
      self.calls += 1
      # Shorter chunks are accepted instead of hard-coding an obsolete horizon.
      return {"actions": np.zeros((3, 8), dtype=np.float32)}

  policy = Policy()
  cfg = DroidInferenceConfig(max_steps=15, open_loop_horizon=8, output_dir=tmp_path)
  rollout = DroidRollout(cfg, adapter, policy, data)
  while not rollout.done(data):
    rollout.step(adapter.model, data)
  assert rollout.control_steps == 15
  assert policy.calls == 5
  assert data.time == pytest.approx(1.0)
  times = np.asarray(rollout.timestamps)
  assert np.all(np.abs(times - np.arange(15) / 15) < 0.0021)
  assert len(list(rollout.output_dir.glob("inference_*.npz"))) == 5
  previous_output = rollout.output_dir
  rollout.reset(adapter.model, data)
  assert (previous_output / "rollout.npz").exists()
  assert (previous_output / "summary.json").exists()
  assert rollout.output_dir != previous_output
  assert rollout.chunk_index == rollout.control_steps == 0
  assert data.time == 0.0
  rollout.step(adapter.model, data)
  assert policy.calls == 6


def test_viser_worker_reset_and_step_keep_renderer_on_owner_thread(
  adapter_and_data, tmp_path: Path
):
  adapter, data = adapter_and_data

  class Policy:
    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
      return {"actions": np.zeros((1, 8), dtype=np.float32)}

  rollout = DroidRollout(
    DroidInferenceConfig(output_dir=tmp_path), adapter, Policy(), data
  )
  rollout.step(adapter.model, data)
  previous_output = rollout.output_dir
  # Viser serializes GUI callbacks against physics, but runs them on a worker.
  with ThreadPoolExecutor(max_workers=1) as gui_worker:
    gui_worker.submit(rollout.reset, adapter.model, data).result()
    assert (previous_output / "final_views.png").exists()
    with np.load(previous_output / "rollout.npz") as saved:
      assert len(saved["actions"]) == 1
    assert rollout.control_steps == 0
    assert data.time == 0.0
    assert rollout.output_dir != previous_output
    gui_worker.submit(rollout.step, adapter.model, data).result()
  # Return to the main simulation thread for the next inference and final save.
  while rollout.control_steps < 2:
    rollout.step(adapter.model, data)
  rollout.save(data)
  adapter.close()
  assert (rollout.output_dir / "final_views.png").exists()


def test_openpi_client_matches_upstream_numpy_wire_format(monkeypatch):
  action = np.arange(120, dtype=np.float32).reshape(15, 8)
  # Upstream openpi_client.msgpack_numpy encodes arrays as this binary-key map.
  reply = msgpack.packb(
    {
      "actions": {
        b"__ndarray__": True,
        b"data": action.tobytes(),
        b"dtype": "<f4",
        b"shape": (15, 8),
      }
    }
  )

  class Connection:
    def __init__(self):
      self.messages = iter((msgpack.packb({"model": "pi05_droid"}), reply))
      self.sent = b""
      self.closed = False

    def recv(self, timeout):
      return next(self.messages)

    def send(self, message):
      self.sent = message

    def close(self):
      self.closed = True

  connection = Connection()
  monkeypatch.setattr(openpi, "connect", lambda *a, **kw: connection)
  client = openpi.OpenpiClient("ws://localhost:8000")
  assert client.metadata["model"] == "pi05_droid"
  result = client.infer({"state": np.ones(8, dtype=np.float32), "prompt": "pick up"})
  assert np.array_equal(result["actions"], action)
  sent = msgpack.unpackb(connection.sent)
  assert sent["state"][b"shape"] == [8]
  assert sent["state"][b"dtype"] == "<f4"
  assert sent["state"][b"data"] == np.ones(8, dtype=np.float32).tobytes()
  client.close()
  assert connection.closed
