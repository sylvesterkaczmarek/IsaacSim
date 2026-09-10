#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plan or execute Isaac Sim standalone binary installation."""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

DOCS_DOWNLOAD_PAGE = "https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html"
WARMUP_LOG_NAME = "install-warmup.log"
WARMUP_MARKER_NAME = ".isaac-sim-warmup-complete"

FATAL_WARMUP_PATTERNS = (
    re.compile(r"Assertion Failed!", re.IGNORECASE),
    re.compile(r"Assertion\s+\(.+?\)\s+failed", re.IGNORECASE),
    re.compile(r"Aborted(?:\s+\(core dumped\))?", re.IGNORECASE),
    re.compile(r"failed to exec the new image", re.IGNORECASE),
    re.compile(r"failed to launch the new process", re.IGNORECASE),
)

# Only variables required by the packaged installation helpers and graphics
# warmup are forwarded. Credentials and Python injection variables are
# deliberately excluded.
_SAFE_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TEMP",
        "TMP",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "ALLUSERSPROFILE",
        "COMMONPROGRAMFILES",
        "COMMONPROGRAMFILES(X86)",
        "COMMONPROGRAMW6432",
        "COMPUTERNAME",
        "DRIVERDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "NUMBER_OF_PROCESSORS",
        "OS",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "PROCESSOR_LEVEL",
        "PROCESSOR_REVISION",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "PROGRAMW6432",
        "PUBLIC",
        "USERDOMAIN",
        "USERNAME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "XAUTHORITY",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "VK_ICD_FILENAMES",
        "__GLX_VENDOR_LIBRARY_NAME",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)


def _safe_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for key in _SAFE_ENV_KEYS:
        value = os.getenv(key)
        if value is not None:
            filtered[key] = value
    if extra:
        filtered.update(extra)
    return filtered


FALLBACK_DOWNLOADS = {
    "linux-x86_64": {
        "version": "6.0.1",
        "filename": "isaac-sim-standalone-6.0.1-linux-x86_64.zip",
        "url": "https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-6.0.1-linux-x86_64.zip",
        "md5": "65e2c2e83e2461ce0f33b0732d0ee4a3",
    },
    "linux-aarch64": {
        "version": "6.0.1",
        "filename": "isaac-sim-standalone-6.0.1-linux-aarch64.zip",
        "url": "https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-6.0.1-linux-aarch64.zip",
        "md5": "1b18ff16e1746d2800df59a7e4ba04b4",
    },
    "windows-x86_64": {
        "version": "6.0.1",
        "filename": "isaac-sim-standalone-6.0.1-windows-x86_64.zip",
        "url": "https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-6.0.1-windows-x86_64.zip",
        "md5": "c7fa3a830b251f10305cd7883039df9b",
    },
}


def default_platform() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system.startswith("win"):
        if machine not in {"amd64", "x86_64"}:
            raise RuntimeError(
                f"Unsupported Isaac Sim Windows architecture: {machine}. " "Only Windows x86_64 is supported."
            )
        return "windows-x86_64"
    if system == "linux":
        if machine in {"aarch64", "arm64"}:
            return "linux-aarch64"
        if machine in {"amd64", "x86_64"}:
            return "linux-x86_64"
        raise RuntimeError(
            f"Unsupported Isaac Sim Linux architecture: {machine}. "
            "Only Linux x86_64 and Linux aarch64 are supported."
        )
    raise RuntimeError(
        f"Unsupported Isaac Sim standalone host OS: {platform.system()}. "
        "Use Linux x86_64, Linux aarch64 on a supported DGX Spark system, or Windows x86_64."
    )


def version_from_filename(filename: str) -> str:
    match = re.search(r"isaac-sim-standalone-([0-9.]+)-", filename)
    return match.group(1) if match else "unknown"


_LABELED_MD5 = re.compile(
    r"\b(?:MD5|md5sum|md5\s*checksum|md5\s*hash)\b\s*[:=]?\s*([a-fA-F0-9]{32})",
    re.IGNORECASE,
)
_ROW_BOUNDARY = re.compile(r"</tr\s*>", re.IGNORECASE)


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", html.unescape(text))


def _find_md5_near(content: str, start: int, end: int, window: int = 2000) -> str | None:
    """Search for an MD5 checksum near a link, respecting HTML row boundaries."""
    before_raw = content[max(0, start - window) : start]
    after_raw = content[end : end + window]
    # Confine the before-window to text after the last </tr> so we don't reach
    # into the previous row and grab that row's checksum.
    last_tr_before = None
    for m in _ROW_BOUNDARY.finditer(before_raw):
        last_tr_before = m
    if last_tr_before is not None:
        before_raw = before_raw[last_tr_before.end() :]
    # Confine the after-window to text before the next </tr> for the same reason.
    first_tr_after = _ROW_BOUNDARY.search(after_raw)
    if first_tr_after is not None:
        after_raw = after_raw[: first_tr_after.start()]
    before = _strip_tags(before_raw)
    after = _strip_tags(after_raw)
    # Prefer a match in the after-window; else use the LAST match in the
    # before-window (closest to the URL).
    after_match = _LABELED_MD5.search(after)
    if after_match:
        return after_match.group(1).lower()
    before_matches = _LABELED_MD5.findall(before)
    if before_matches:
        return before_matches[-1].lower()
    # Fall back to a nearby bare 32-hex string only if it is clearly on its own.
    bare = re.search(r"(?<![a-fA-F0-9])([a-fA-F0-9]{32})(?![a-fA-F0-9])", after)
    if bare:
        return bare.group(1).lower()
    return None


def parse_downloads(content: str) -> dict[str, dict[str, str]]:
    downloads: dict[str, dict[str, str]] = {}
    # Match the standalone zip filename anywhere in href/src or plain text,
    # regardless of host (docs page has migrated hosts before) or protocol.
    filename_pattern = re.compile(
        r"(?P<url>https?://[^\s\"'<>]+?/"
        r"(?P<filename>isaac-sim-standalone-(?P<version>[0-9.]+)-"
        r"(?P<platform>linux-x86_64|linux-aarch64|windows-x86_64)\.zip))",
        re.IGNORECASE,
    )
    for match in filename_pattern.finditer(content):
        platform_name = match.group("platform").lower()
        url = html.unescape(match.group("url"))
        # Keep the highest version seen per platform in case older releases are also linked.
        current = downloads.get(platform_name)
        if current and _version_key(current["version"]) >= _version_key(match.group("version")):
            continue
        md5 = _find_md5_near(content, match.start(), match.end())
        downloads[platform_name] = {
            "version": match.group("version"),
            "filename": match.group("filename"),
            "url": url,
            "md5": md5 or "",
        }
    return downloads


def _version_key(version: str) -> tuple[int, ...]:
    parts = []
    for chunk in version.split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(0)
    return tuple(parts)


_ALLOWED_URL_SCHEMES = frozenset({"https"})


def _require_https(url: str) -> None:
    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        raise RuntimeError(f"Refusing to fetch URL with disallowed scheme '{scheme}'. Only HTTPS is permitted.")


def read_url(url: str, timeout: int) -> str:
    _require_https(url)
    try:
        # nosec B310 - scheme allowlist enforced above; only HTTPS is permitted.
        with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec B310
            return response.read().decode("utf-8", errors="ignore")
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Could not fetch {url}: {exc}") from exc


def fetch_latest_downloads(docs_url: str = DOCS_DOWNLOAD_PAGE, timeout: int = 20) -> dict[str, dict[str, str]]:
    downloads = parse_downloads(read_url(docs_url, timeout))
    if not downloads:
        raise RuntimeError("Could not find Isaac Sim standalone binary links in download docs.")
    return downloads


def resolve_downloads(source: str) -> tuple[dict[str, dict[str, str]], str, str | None]:
    if source == "fallback":
        return FALLBACK_DOWNLOADS, "fallback", None
    try:
        return fetch_latest_downloads(), "public-docs", None
    except RuntimeError as exc:
        return FALLBACK_DOWNLOADS, "fallback", str(exc)


def print_command(command: list[str]) -> None:
    print(format_command(command))


def format_command(command: list[str]) -> str:
    if platform.system().lower().startswith("win"):
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def format_cwd_command(cwd: Path, command: list[str]) -> str:
    if platform.system().lower().startswith("win"):
        return f"(cd /d {subprocess.list2cmdline([str(cwd)])} && {format_command(command)})"
    return f"(cd {shlex.quote(str(cwd))} && {format_command(command)})"


def run(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print_command(command)
    subprocess.run(command, cwd=cwd, env=_safe_env(env), check=True)


def run_warmup(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    marker_path: Path,
) -> None:
    """Stream warmup output, persist it, and fail on hidden fatal signatures."""
    marker_path.unlink(missing_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print_command(command)
    print(f"Warmup log: {log_path}")

    first_fatal: str | None = None
    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            cwd=cwd,
            env=_safe_env({"OMNI_KIT_ACCEPT_EULA": "YES"}),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        ) as process:
            assert process.stdout is not None
            with process.stdout:
                for line in process.stdout:
                    print(line, end="")
                    log.write(line)
                    if first_fatal is None and any(pattern.search(line) for pattern in FATAL_WARMUP_PATTERNS):
                        first_fatal = line.strip()
            returncode = process.wait()

    if returncode != 0 or first_fatal is not None:
        detail = first_fatal or f"warmup process exited with status {returncode}"
        raise RuntimeError(
            "Isaac Sim files were installed, but required warmup is incomplete. "
            f"First fatal condition: {detail}. Full log: {log_path}. "
            "Do not launch Isaac Sim or retry automatically. After resolving the cause, "
            "request an explicit retry with --warmup-only --execute."
        )

    marker_path.write_text("completed\n", encoding="utf-8")
    print(f"Warmup completion marker: {marker_path}")


def download_file(url: str, destination: Path) -> None:
    _require_https(url)
    print(f"Downloading {url} -> {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # nosec B310 - scheme allowlist enforced above; only HTTPS is permitted.
    with urllib.request.urlopen(url) as response, destination.open("wb") as file:  # nosec B310
        shutil.copyfileobj(response, file)


def extract_zip(zip_path: Path, install_dir: Path) -> None:
    print(f"Extracting {zip_path} -> {install_dir}")
    install_dir.mkdir(parents=True, exist_ok=True)
    install_root = install_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_name = member.filename
            member_path = Path(member_name)
            member_mode = (member.external_attr >> 16) & 0o170000
            if (
                member_path.is_absolute()
                or re.match(r"^[A-Za-z]:", member_name)
                or ".." in member_path.parts
                or member_mode == stat.S_IFLNK
            ):
                raise RuntimeError(f"Unsafe zip member path: {member_name}")
            target_path = (install_root / member_path).resolve()
            if not target_path.is_relative_to(install_root):
                raise RuntimeError(f"Zip member escapes install directory: {member_name}")
        archive.extractall(install_dir)


LAUNCHER_SCRIPTS = (
    "isaac-sim.sh",
    "isaac-sim.selector.sh",
    "isaac-sim.streaming.sh",
    "isaac-sim.headless.native.sh",
    "isaac-sim.headless.webrtc.sh",
    "isaac-sim.compatibility_check.sh",
    "post_install.sh",
    "warmup.sh",
    "runheadless.sh",
    "python.sh",
    "kit/kit",
    "kit/kit-gcov",
)

PACKAGED_EXECUTABLE_GLOBS = ("extscache/omni.kit.telemetry-*/omni.telemetry.transmitter/omni.telemetry.transmitter",)


def make_shell_scripts_executable(install_dir: Path) -> None:
    print(f"Making packaged launchers runnable under {install_dir}")
    for relative in LAUNCHER_SCRIPTS:
        script = install_dir / relative
        if script.is_file():
            script.chmod(script.stat().st_mode | 0o111)
    for pattern in PACKAGED_EXECUTABLE_GLOBS:
        for executable in install_dir.glob(pattern):
            if executable.is_file():
                executable.chmod(executable.stat().st_mode | 0o111)
    python_bin = install_dir / "kit" / "python" / "bin"
    if python_bin.is_dir():
        for executable in python_bin.iterdir():
            if executable.is_file():
                executable.chmod(executable.stat().st_mode | 0o111)


def md5sum(path: Path) -> str:
    # MD5 is used solely to match the publisher-supplied checksum on the
    # Isaac Sim download page. It is not used for authentication or any
    # security decision, so usedforsecurity=False is correct here.
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def post_install_script(install_dir: Path, platform_name: str) -> Path:
    if platform_name.startswith("windows"):
        return install_dir / "post_install.bat"
    return install_dir / "post_install.sh"


def warmup_script(install_dir: Path, platform_name: str) -> Path:
    if platform_name.startswith("windows"):
        return install_dir / "warmup.bat"
    return install_dir / "warmup.sh"


def script_command(script: Path, platform_name: str, *, strict: bool = False) -> list[str]:
    if platform_name.startswith("windows"):
        return ["cmd", "/c", str(script)]
    return ["bash", "-e", str(script)] if strict else ["bash", str(script)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Isaac Sim from a standalone zip package.")
    parser.add_argument("--zip", help="Path to the downloaded Isaac Sim standalone zip.")
    parser.add_argument("--install-dir", default="~/isaacsim", help="Directory to extract Isaac Sim into.")
    parser.add_argument(
        "--platform",
        choices=sorted(FALLBACK_DOWNLOADS),
        default=None,
        help="Public-docs download platform to infer when --zip is omitted.",
    )
    parser.add_argument(
        "--release-source",
        choices=["docs", "fallback"],
        default="docs",
        help="Use latest public docs at runtime or pinned fallback metadata.",
    )
    parser.add_argument("--download-dir", default="~/Downloads", help="Directory for inferred or downloaded zip.")
    parser.add_argument("--download-url", help="Override Isaac Sim standalone zip URL.")
    parser.add_argument("--expected-md5", help="Override expected MD5 for --download-url or custom zip validation.")
    parser.add_argument("--download", action="store_true", help="Download the standalone zip before installing.")
    parser.add_argument("--verify-md5", action="store_true", help="Verify the standalone zip MD5 before installing.")
    parser.add_argument("--skip-post-install", action="store_true", help="Skip post_install script.")
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help=(
            "Retry only the required warmup for an existing extracted install. "
            "Use only after explicit retry approval."
        ),
    )
    parser.add_argument("--execute", action="store_true", help="Actually run commands. Default is dry-run.")
    args = parser.parse_args()

    if args.platform is None:
        args.platform = default_platform()

    if args.platform.startswith("windows") and args.skip_post_install:
        parser.error(
            "--skip-post-install is not allowed on Windows; the required order is "
            "Compatibility Checker, post_install.bat, then warmup.bat."
        )
    if args.download_url and (not args.expected_md5 or not args.verify_md5):
        parser.error("--download-url requires --expected-md5 and --verify-md5.")
    if args.warmup_only and any(
        (
            args.zip,
            args.download,
            args.verify_md5,
            args.download_url,
            args.expected_md5,
            args.skip_post_install,
        )
    ):
        parser.error("--warmup-only cannot be combined with download, archive, checksum, or post-install flags.")

    install_dir = Path(args.install_dir).expanduser()
    warmup = warmup_script(install_dir, args.platform)
    warmup_command = script_command(warmup, args.platform, strict=True)
    warmup_log = install_dir / WARMUP_LOG_NAME
    warmup_marker = install_dir / WARMUP_MARKER_NAME

    if args.warmup_only:
        print("Isaac Sim standalone warmup-only retry plan:")
        print("# Retry requires explicit user approval; do not download, extract, post-install, or launch.")
        print(format_cwd_command(install_dir, warmup_command))
        print(f"# Stream output and save it to: {warmup_log}")
        print(f"# Write completion marker only after a clean run: {warmup_marker}")
        if not args.execute:
            print("\nDry-run only. Re-run with --execute only after explicit warmup retry approval.")
            return 0
        if not install_dir.is_dir():
            raise FileNotFoundError(f"Isaac Sim install directory not found: {install_dir}")
        if not warmup.exists():
            raise FileNotFoundError(f"Required standalone warmup script not found: {warmup}")
        if not args.platform.startswith("windows"):
            make_shell_scripts_executable(install_dir)
        run_warmup(
            warmup_command,
            cwd=install_dir,
            log_path=warmup_log,
            marker_path=warmup_marker,
        )
        print(f"Standalone warmup completed: {warmup.resolve()}")
        print(f"Install location: {install_dir.resolve()}")
        print("No Isaac Sim application launcher was run.")
        return 0

    downloads, release_source, release_warning = resolve_downloads(args.release_source)
    download_info = downloads.get(args.platform, FALLBACK_DOWNLOADS[args.platform])
    download_url = args.download_url or download_info["url"]
    filename = Path(download_url).name if args.download_url and not args.zip else download_info["filename"]
    zip_path = Path(args.zip).expanduser() if args.zip else Path(args.download_dir).expanduser() / filename
    expected_md5 = args.expected_md5 or download_info.get("md5")
    version = download_info.get("version") or version_from_filename(filename)
    post_install = post_install_script(install_dir, args.platform)
    post_install_command = script_command(post_install, args.platform)
    is_windows = args.platform.startswith("windows")
    compatibility_checker = install_dir / "isaac-sim.compatibility_check.bat"
    compatibility_command = [
        "cmd",
        "/c",
        str(compatibility_checker),
        "--no-window",
    ]
    compatibility_validator_command = [
        sys.executable,
        str(Path(__file__).with_name("validate_install.py")),
        "workstation-compatibility",
        "--install-dir",
        str(install_dir),
        "--execute",
    ]

    commands: list[tuple[str, list[str], Path | None]] = []
    if args.download:
        commands.extend(
            [
                ("native", ["create directory", str(zip_path.parent)], None),
                ("native", ["download", download_url, str(zip_path)], None),
            ]
        )
    commands.extend(
        [
            ("native", ["create directory", str(install_dir)], None),
            ("native", ["extract zip", str(zip_path), str(install_dir)], None),
        ]
    )
    if not args.platform.startswith("windows"):
        commands.append(("native", ["make packaged executables runnable", str(install_dir)], None))
    if is_windows:
        commands.append(("compatibility", compatibility_command, install_dir))
    if not args.skip_post_install:
        commands.append(("subprocess", post_install_command, install_dir))
    commands.append(("warmup", warmup_command, install_dir))

    print("Isaac Sim standalone binary install plan:")
    print("# Installation-only boundary: run required warmup, but do not launch Isaac Sim.")
    if is_windows:
        print("# Windows-only order: download, verify, extract, Compatibility Checker, " "post_install, warmup.")
    print(f"# Release source: {release_source}")
    if release_warning:
        print(f"# WARNING: {release_warning}")
    print(f"# Isaac Sim release: {version} for {args.platform}")
    print(f"# Download URL: {download_url}")
    print(f"# Expected MD5: {expected_md5 or 'not provided'}")
    if not zip_path.exists():
        print(f"# WARNING: zip file does not exist yet: {zip_path}")
    for kind, command, cwd in commands:
        if kind == "native":
            print(f"# Python: {' '.join(command)}")
        elif kind == "compatibility":
            assert cwd is not None
            print(format_cwd_command(cwd, command))
            print("# Require the explicit result: System checking result: PASSED")
        elif kind == "warmup":
            assert cwd is not None
            print(format_cwd_command(cwd, command))
            print(f"# Stream output and save it to: {warmup_log}")
            print(f"# Write completion marker only after a clean run: {warmup_marker}")
        elif cwd:
            print(format_cwd_command(cwd, command))
        else:
            print_command(command)

    if not args.execute:
        print("\nDry-run only. Re-run with --execute after confirming the downloaded zip and install directory.")
        return 0

    if args.verify_md5 and not expected_md5:
        raise RuntimeError("Cannot verify MD5 because no expected checksum is available. Pass --expected-md5.")
    if args.download:
        download_file(download_url, zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Isaac Sim zip not found: {zip_path}")
    if args.verify_md5:
        actual_md5 = md5sum(zip_path)
        if actual_md5 != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {zip_path}: expected {expected_md5}, got {actual_md5}")
        print(f"MD5 verified: {actual_md5}")
    extract_zip(zip_path, install_dir)
    if not args.platform.startswith("windows"):
        make_shell_scripts_executable(install_dir)
    if is_windows:
        if not compatibility_checker.is_file():
            raise FileNotFoundError(f"Required Windows Compatibility Checker not found: {compatibility_checker}")
        compatibility_result = subprocess.run(
            compatibility_validator_command,
            check=False,
            env=_safe_env({"OMNI_KIT_ACCEPT_EULA": "YES"}),
        )
        if compatibility_result.returncode != 0:
            raise RuntimeError(
                "Windows Compatibility Checker did not report PASSED. " "post_install.bat and warmup.bat were not run."
            )
    if not args.skip_post_install:
        run(
            post_install_command,
            cwd=install_dir,
            env={"OMNI_KIT_ACCEPT_EULA": "YES"},
        )
    if not warmup.exists():
        raise FileNotFoundError(f"Required standalone warmup script not found: {warmup}")
    run_warmup(
        warmup_command,
        cwd=install_dir,
        log_path=warmup_log,
        marker_path=warmup_marker,
    )
    print("Isaac Sim standalone binary installed successfully.")
    print(f"Install location: {install_dir.resolve()}")
    print(f"Downloaded archive: {zip_path.resolve()}")
    print(f"Warmup completed: {warmup.resolve()}")
    print(f"Warmup log: {warmup_log.resolve()}")
    print(f"Warmup marker: {warmup_marker.resolve()}")
    if is_windows:
        print(f"Compatibility Checker: PASSED ({compatibility_checker.resolve()})")
    print("The Isaac Sim main application launcher was not run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
