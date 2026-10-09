"""Experiment client for OpenPI's NumPy MessagePack/WebSocket protocol.

Wire format reference: openpi-client/src/openpi_client/msgpack_numpy.py in
https://github.com/Physical-Intelligence/openpi (Apache-2.0).
"""

from typing import Any

import msgpack
import numpy as np
from websockets.sync.client import connect


def _encode_array(value: Any) -> Any:
  if isinstance(value, (np.ndarray, np.generic)):
    if value.dtype.kind in ("V", "O", "c"):
      raise ValueError(f"Unsupported array dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
      return {
        b"__ndarray__": True,
        b"data": value.tobytes(),
        b"dtype": value.dtype.str,
        b"shape": value.shape,
      }
    return {
      b"__npgeneric__": True,
      b"data": value.item(),
      b"dtype": value.dtype.str,
    }
  raise TypeError(f"Cannot serialize {type(value).__name__}")


def _decode_array(value: dict) -> Any:
  if b"__ndarray__" in value:
    dtype = np.dtype(value[b"dtype"])
    if dtype.kind in ("V", "O", "c"):
      raise ValueError(f"Unsupported array dtype: {dtype}")
    return np.frombuffer(value[b"data"], dtype=dtype).reshape(value[b"shape"])
  if b"__npgeneric__" in value:
    return np.dtype(value[b"dtype"]).type(value[b"data"])
  return value


def pack_message(value: dict) -> bytes:
  packed = msgpack.packb(value, default=_encode_array)
  assert isinstance(packed, bytes)
  return packed


def unpack_message(value: bytes) -> dict[str, Any]:
  result = msgpack.unpackb(value, object_hook=_decode_array)
  if not isinstance(result, dict):
    raise ValueError("OpenPI response must be a dictionary")
  return result


class OpenpiClient:
  """Synchronous client for an upstream scripts/serve_policy.py server."""

  def __init__(self, uri: str, timeout: float = 300.0) -> None:
    self._timeout = timeout
    self._connection = connect(
      uri, compression=None, max_size=None, open_timeout=timeout, proxy=None
    )
    try:
      self.metadata = self._receive()
    except BaseException:
      self.close()
      raise

  def _receive(self) -> dict[str, Any]:
    message = self._connection.recv(timeout=self._timeout)
    if isinstance(message, str):
      raise RuntimeError(f"OpenPI server error:\n{message}")
    return unpack_message(message)

  def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
    self._connection.send(pack_message(observation))
    return self._receive()

  def close(self) -> None:
    self._connection.close()
