# Standalone Binary Installation

Use this reference to download, verify, extract, run the required warmup, and report the standalone install directory.

Public workstation docs show:

- Download the latest Isaac Sim release for the platform from the public Download Isaac Sim page.
- Create an `isaacsim` install folder.
- Unzip the standalone package into that folder.
- Run `post_install`.
- Run `warmup.sh` on Linux or `warmup.bat` on Windows after `post_install`.
- Stream warmup output to `<install-dir>/install-warmup.log`. Treat nonzero exit, `Assertion Failed`, abort/core-dump, or failed child-process execution as incomplete warmup even when the packaged script later returns zero.
- Write `<install-dir>/.isaac-sim-warmup-complete` only after a clean warmup.
- Stop after warmup; do not launch Isaac Sim.

Linux x86_64 example from public docs:

```bash
mkdir ~/isaacsim
cd ~/Downloads
unzip "isaac-sim-standalone-6.0.1-linux-x86_64.zip" -d ~/isaacsim
cd ~/isaacsim
./post_install.sh
./warmup.sh
echo "Isaac Sim installed at $HOME/isaacsim"
```

Windows example from public docs:

```cmd
mkdir C:\isaacsim
cd %USERPROFILE%/Downloads
tar -xvzf "isaac-sim-standalone-6.0.1-windows-x86_64.zip" -C C:\isaacsim
cd C:\isaacsim
post_install.bat
warmup.bat
echo Isaac Sim installed at C:\isaacsim
```

Runtime release resolution:

- By default, `install_binary.py` fetches the public Isaac Sim download page and uses the current "Latest Release" standalone binary link plus MD5 for the selected platform.
- If the docs page is unreachable, the script falls back to pinned metadata so users still get a safe, reproducible plan.
- Use `--release-source fallback` to force pinned metadata for offline/reproducible runs.
- Use `--download-url` and `--expected-md5` only when the user explicitly supplies a custom public download source.

Pinned fallback metadata:

| Platform | Version | Filename | MD5 |
|---|---:|---|---|
| Linux x86_64 | 6.0.1 | `isaac-sim-standalone-6.0.1-linux-x86_64.zip` | `65e2c2e83e2461ce0f33b0732d0ee4a3` |
| Linux aarch64 | 6.0.1 | `isaac-sim-standalone-6.0.1-linux-aarch64.zip` | `1b18ff16e1746d2800df59a7e4ba04b4` |
| Windows x86_64 | 6.0.1 | `isaac-sim-standalone-6.0.1-windows-x86_64.zip` | `c7fa3a830b251f10305cd7883039df9b` |

Script, after explicit approval:

```bash
python3 scripts/install_binary.py --zip ~/Downloads/isaac-sim-standalone-6.0.1-linux-x86_64.zip --install-dir ~/isaacsim --execute
```

Download and install from latest public-doc defaults, after explicit approval:

```bash
python3 scripts/install_binary.py --platform linux-x86_64 --download --verify-md5 --install-dir ~/isaacsim --execute
```

Script examples:

```bash
# Dry-run latest public-docs binary download and install plan.
python3 scripts/install_binary.py --platform linux-x86_64 --download --verify-md5 --install-dir ~/isaacsim

# Execute latest public-docs binary download and install only after explicit approval.
python3 scripts/install_binary.py --platform linux-x86_64 --download --verify-md5 --install-dir ~/isaacsim --execute

# Retry only a failed warmup after a separate explicit retry approval.
python3 scripts/install_binary.py --platform linux-x86_64 --install-dir ~/isaacsim --warmup-only
python3 scripts/install_binary.py --platform linux-x86_64 --install-dir ~/isaacsim --warmup-only --execute

# Install from an already downloaded standalone zip only after explicit approval.
python3 scripts/install_binary.py --zip ~/Downloads/isaac-sim-standalone-6.0.1-linux-x86_64.zip --install-dir ~/isaacsim --execute

# Force pinned fallback metadata instead of live public-docs lookup.
python3 scripts/install_binary.py --release-source fallback --platform linux-x86_64 --download --verify-md5 --install-dir ~/isaacsim

# Install an older build only when the user provides the URL and MD5, and explicitly approves execution.
python3 scripts/install_binary.py --download --download-url "https://downloads.isaacsim.nvidia.com/<older-build>.zip" --expected-md5 "<md5>" --verify-md5 --install-dir ~/isaacsim --execute
```

Public docs:

- Download Isaac Sim: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html
- Workstation installation: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html

The public docs may describe launch steps. This installation skill runs the required warmup and then stops; it never invokes `isaac-sim.sh` or `isaac-sim.bat`.

## Compatibility Checker

The Compatibility Checker itself supports both Windows and Linux. On Windows, `install_binary.py` automatically runs it after extraction and before `post_install.bat` and `warmup.bat`, and requires the explicit `System checking result: PASSED` marker.

For an already extracted workstation build, plan it with:

```bash
# Windows
python3 scripts/validate_install.py workstation-compatibility --install-dir C:\isaacsim

# Linux
python3 scripts/validate_install.py workstation-compatibility --install-dir ~/isaacsim
```

After explicit approval, add `--execute`. This invokes `isaac-sim.compatibility_check.bat --no-window` on Windows so an SSH-driven check does not wait for GUI backbuffers, stops the checker after it prints its semantic result, and requires `System checking result: PASSED`. On Linux it invokes `isaac-sim.compatibility_check.sh` without changing the existing Linux behavior. It does not invoke the normal Isaac Sim launcher.
