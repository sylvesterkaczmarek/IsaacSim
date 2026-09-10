# Python Package Installation

Use Python 3.12 for the current Isaac Sim Python package workflow on supported Linux and Windows platforms.

Baseline command:

```bash
python3.12 -m venv ~/env_isaacsim
source ~/env_isaacsim/bin/activate
pip install --upgrade pip
pip --isolated install 'isaacsim[all,extscache]==6.0.1' --extra-index-url https://pypi.nvidia.com
```

Important details:

- Python 3.12 must provide both `venv` and `ensurepip`. Run `scripts/preflight.py` before installing. On Debian/Ubuntu, install the explicit prerequisite with `sudo apt install python3.12-venv`, then rerun preflight.
- The operator must accept the NVIDIA Omniverse End User License Agreement (<https://docs.omniverse.nvidia.com/eula>) before installation. The installer records `--accept-eula` in the approved plan but does not import or launch Isaac Sim.
- `--extra-index-url https://pypi.nvidia.com` adds the NVIDIA PyPI index alongside the public PyPI. `pip` resolves packages from both indexes, which introduces dependency-confusion risk if a matching package name appears on public PyPI. The bundled installer therefore requires an exact `--version`, invokes pip with `--isolated`, disables inherited pip configuration, and requires every installed `isaacsim*` artifact in pip's resolution report to use HTTPS on an NVIDIA domain before reporting success or executing an Isaac Sim command. Additional mitigations:
  - Prefer `--index-url https://pypi.nvidia.com` for Isaac Sim installs when possible so only the NVIDIA index is consulted.
  - When you must use `--extra-index-url`, pin exact versions (`isaacsim[all,extscache]==<version>`) and prefer `pip install --require-hashes` with a lockfile so unexpected substitutions are caught.
  - Treat `https://pypi.nvidia.com` as the authoritative source for the `isaacsim` package. Cross-check the resolved wheel URL against `pypi.nvidia.com` before installing.
- `No matching distribution found` usually means wrong Python version, missing NVIDIA PyPI index, unsupported platform, or unavailable exact package version.
- Some examples/extensions may require additional Python dependencies.

Script, after explicit approval:

```bash
python3 scripts/install_python.py --env-dir ~/env_isaacsim --version 6.0.1 --accept-eula --execute
```

Script examples:

```bash
# Dry-run the Python 3.12 venv and package install plan.
python3 scripts/install_python.py --env-dir ~/env_isaacsim --accept-eula

# Execute the Python package install after explicit approval.
python3 scripts/install_python.py --env-dir ~/env_isaacsim --version 6.0.1 --accept-eula --execute

# Pin an Isaac Sim package version when required, after explicit approval.
python3 scripts/install_python.py --env-dir ~/env_isaacsim --version 6.0.1 --accept-eula --execute

# Use a different extras set after explicit approval.
python3 scripts/install_python.py --env-dir ~/env_isaacsim --extras all --version 6.0.1 --accept-eula --execute
```

After installation, report the virtual-environment path, Python executable, and installed package request. Do not import `isaacsim` or create a `SimulationApp`.

On Windows only, install the full selected package first and then run the Compatibility Checker from the same environment:

```bash
%USERPROFILE%\env_isaacsim\Scripts\isaacsim.exe isaacsim.exp.compatibility_check --no-window
```

`install_python.py` includes this exact Windows-only checker command after the full pip installation, without `--/app/quitAfter=10`. It stops the checker after the semantic result, requires the explicit `System checking result: PASSED` marker, and then reports the environment and Python executable paths. For Isaac Sim 6.0.1, it also adds `packaging` because the Windows checker imports `packaging.version` but the published packages do not install it transitively.

On Linux, retain the separate pre-install checker workflow:

```bash
python3 scripts/validate_install.py python-compatibility \
  --isaacsim ~/env_isaacsim_checker/bin/isaacsim
```
