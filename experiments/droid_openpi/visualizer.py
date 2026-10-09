"""Viser panels showing the exact RGB observations sent to the policy."""

from typing import Any, Literal

import numpy as np
import viser


class PolicyCameraPanels:
  def __init__(
    self,
    server: viser.ViserServer,
    external_camera: Literal["left", "right"],
  ) -> None:
    self._server = server
    with server.gui.add_folder("Policy camera inputs", order=-1):
      self.status = server.gui.add_markdown("Waiting for a policy observation.")
      placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
      self.exterior = server.gui.add_image(
        placeholder,
        label=f"Exterior ({external_camera} shoulder)",
        format="png",
        visible=False,
      )
      self.wrist = server.gui.add_image(
        placeholder, label="Wrist", format="png", visible=False
      )

  def update(self, observation: dict[str, Any], simulation_time: float) -> None:
    with self._server.atomic():
      self.exterior.image = observation["observation/exterior_image_1_left"]
      self.wrist.image = observation["observation/wrist_image_left"]
      self.exterior.visible = self.wrist.visible = True
      self.status.content = (
        f"Policy inputs at **{simulation_time:.2f} s** (simulation time).\n\n"
        "Images update with each policy request."
      )

  def reset(self) -> None:
    with self._server.atomic():
      self.exterior.visible = self.wrist.visible = False
      self.status.content = "Waiting for a policy observation."
