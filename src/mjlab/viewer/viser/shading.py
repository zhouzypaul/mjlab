"""Appearance adjustments for GLB meshes displayed by Viser."""

import io

import numpy as np
import trimesh
import viser
from trimesh.exchange.gltf import export_glb


def set_glb_flat_shading(server: viser.ViserServer, node_name: str) -> None:
  """Give a GLB node face normals while retaining its handle and transforms.

  mjviser uses GLB handles, which do not expose Viser's flat_shading option.
  Splitting shared vertices prevents normals from smoothing across box edges.
  """
  handle = server.scene.get_handle_by_name(node_name)
  if not isinstance(handle, (viser.GlbHandle, viser.BatchedGlbHandle)):
    raise ValueError(f"Expected a GLB mesh at {node_name}")
  scene = trimesh.load_scene(
    io.BytesIO(handle.glb_data), file_type="glb", process=False
  )
  for mesh in scene.geometry.values():
    if isinstance(mesh, trimesh.Trimesh):
      mesh.unmerge_vertices()
      mesh.vertex_normals = np.repeat(mesh.face_normals, 3, axis=0)
  handle.glb_data = export_glb(scene, include_normals=True)
