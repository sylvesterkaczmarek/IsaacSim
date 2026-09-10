# Troubleshooting

Use log-first diagnosis.

## Triage Order

1. Identify install type and exact command.
2. Find the first real failure line.
3. Separate install, launch, environment, and user-script failures.
4. Check only relevant system facts: Python version, GPU/driver, Docker runtime, EULA, display, or network.
5. Give the smallest repair path and a validation command.

## Log Locations

Use the relevant path with `scripts/diagnose_logs.py`.

Exit codes are automation-safe: `0` means no known issue was found, `1` means
one or more known problem patterns were found, and `2` means a log could not be
scanned (for example, because it is missing).

- Standalone binary: check the terminal output first, then inspect `~/.nvidia-omniverse/logs/Kit/Isaac-Sim/` and `<install-dir>/kit/logs/` if present.
- Docker/container: inspect `docker logs <container-name>` for active or named containers, and the mounted logs directory from `install_container.py` (default `~/docker/isaac-sim/logs`).
- Python package: check the terminal output first, then inspect the active working directory, `~/.nvidia-omniverse/logs/Kit/Isaac-Sim/`, and any log path printed by the failing Python process.
- Validation script: use the command output from `scripts/validate_install.py`; if it launched Isaac Sim, follow the matching binary, Python, or container log path above.

## Common Patterns

- `No matching distribution found`: wrong Python version, missing NVIDIA PyPI index, unsupported platform, or unavailable version.
- EULA prompt or runtime block: set `OMNI_KIT_ACCEPT_EULA=YES` before import/launch, or `ACCEPT_EULA=Y` for containers.
- Docker sees no GPU: install/configure NVIDIA Container Toolkit and restart Docker.
- WebRTC connects but no video: confirm the container was launched with published `-p` mappings for `49100/tcp` and `47998/udp` under bridge networking, set `ISAACSIM_HOST` to a reachable interface, and ensure both ports are reachable from the client.
- GUI/display failures: verify `DISPLAY`, Xauthority, X11/Wayland state, and driver stack.
- Slow first launch: shader/cache warmup may be normal; inspect logs for progress.
- Standalone warmup assertion or abort: inspect `<install-dir>/install-warmup.log`. The installer must leave `.isaac-sim-warmup-complete` absent and report warmup incomplete. Resolve the first fatal condition, then use `install_binary.py --install-dir <path> --platform <platform> --warmup-only --execute` only after explicit retry approval.
