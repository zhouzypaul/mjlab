"""Check that flat shading survives the GLB round trip used by Viser."""

import io
from unittest.mock import MagicMock

import numpy as np
import pytest
import trimesh
import viser
from trimesh.visual import ColorVisuals

from mjlab.viewer.viser.shading import set_glb_flat_shading


@pytest.mark.parametrize("handle_type", [viser.GlbHandle, viser.BatchedGlbHandle])
def test_glb_flat_shading_preserves_box_geometry_and_color(handle_type):
  original = trimesh.creation.box(extents=(0.05,) * 3)
  color = (204, 38, 25, 255)
  assert isinstance(original.visual, ColorVisuals)
  original.visual.vertex_colors = np.tile(color, (len(original.vertices), 1))
  handle = MagicMock(spec=handle_type)
  handle.glb_data = original.export(file_type="glb")
  server = MagicMock()
  server.scene.get_handle_by_name.return_value = handle

  set_glb_flat_shading(server, "/cube")

  scene = trimesh.load_scene(
    io.BytesIO(handle.glb_data), file_type="glb", process=False
  )
  mesh = next(iter(scene.geometry.values()))
  assert isinstance(mesh, trimesh.Trimesh)
  assert np.allclose(mesh.triangles, original.triangles)
  assert isinstance(mesh.visual, ColorVisuals)
  assert np.all(mesh.visual.vertex_colors == color)
  # Every triangle's vertex normals equal its face normal, even at box edges.
  assert np.allclose(mesh.vertex_normals[mesh.faces], mesh.face_normals[:, None, :])
