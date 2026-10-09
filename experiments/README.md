# Experiments

Keep experimental runners and their integration helpers together in a named
subdirectory here. Experiments consume `mjlab` robot assets and task scenes;
simulation modules must not depend on experiments. This directory is used from
the repository checkout and is not part of the installed `mjlab` package.

The DROID/OpenPI experiment lives in `droid_openpi/`:

- `run.py`: command-line options, rollout scheduling, and saved records.
- `policy_adapter.py`: model observations, camera rendering, and action conversion.
- `client.py`: OpenPI WebSocket and NumPy MessagePack transport.

Run from the repository root with an OpenPI server listening on port 8000:

```sh
uv run --extra cpu --group droid-openpi -m experiments.droid_openpi.run \
  --prompt "pick up the red cube"
```

See [DROID_sim.md](../DROID_sim.md) for installation and model-server instructions.
The `dev` dependency group includes this experiment's client dependencies.
Formatting and type checks include experiment code, and its tests live under
`tests/experiments/`:

```sh
make check
uv run --group dev pytest tests/experiments/test_droid_openpi.py
```
