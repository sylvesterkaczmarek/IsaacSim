"""Batch-run the AssetTransformerManager over a list of USD assets.

This standalone script launches a headless SimulationApp, enables the asset
transformer extensions, and processes every asset listed in a text file through
the configured rule profile.

Usage::

    python.sh source/extensions/isaacsim.asset.transformer/scripts/run_asset_transformer.py \\
        /path/to/asset_root \\
        /path/to/asset_list.txt \\
        /path/to/output_root
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_LOGGER = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run AssetTransformerManager on a batch of USD assets.",
    )
    parser.add_argument(
        "asset_root",
        help="Root directory containing the source USD assets.",
    )
    parser.add_argument(
        "asset_list",
        help="Path to a text file listing asset paths relative to asset_root (one per line).",
    )
    parser.add_argument(
        "output_root",
        help="Root directory for all output packages.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help=(
            "Path to a profile JSON file. Defaults to the bundled " "isaacsim_structure.json from the rules extension."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-process all assets even if they were previously completed.",
    )
    return parser.parse_args()


def _resolve_default_profile() -> Path:
    """Resolve the default profile JSON path relative to the script location.

    Returns:
        Path to the rules extension's ``isaacsim_structure.json``.
    """
    return (
        _SCRIPT_DIR.parent.parent
        / "extensions"
        / "isaacsim.asset.transformer.rules"
        / "data"
        / "isaacsim_structure.json"
    ).resolve()


def main() -> None:
    """Entry point for the batch asset transformer."""
    args = _parse_args()

    asset_root = Path(args.asset_root).resolve()
    asset_list_path = Path(args.asset_list).resolve()
    output_root = Path(args.output_root).resolve()

    # ------------------------------------------------------------------
    # 1. Read the asset list before starting the app (fast-fail on bad input)
    # ------------------------------------------------------------------
    if not asset_list_path.is_file():
        _LOGGER.error("Asset list file not found: %s", asset_list_path)
        sys.exit(1)

    # Late import: util.py is in the package, available after extensions load.
    # Read the list with stdlib here to fast-fail before SimulationApp starts.
    asset_paths: list[str] = []
    with open(asset_list_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            asset_paths.append(line)

    if not asset_paths:
        _LOGGER.error("No assets found in %s", asset_list_path)
        sys.exit(1)

    _LOGGER.info("Found %d asset(s) to process.", len(asset_paths))

    # ------------------------------------------------------------------
    # 2. Launch headless SimulationApp and enable extensions
    # ------------------------------------------------------------------
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})

    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.asset.transformer", True)
    ext_manager.set_extension_enabled_immediate("isaacsim.asset.transformer.rules", True)

    # ------------------------------------------------------------------
    # 3. Import transformer classes (extensions must be enabled first)
    # ------------------------------------------------------------------
    from isaacsim.asset.transformer import AssetTransformerManager, RuleProfile

    # ------------------------------------------------------------------
    # 4. Load the rule profile
    # ------------------------------------------------------------------
    profile_path = Path(args.profile).resolve() if args.profile else _resolve_default_profile()

    if not profile_path.is_file():
        _LOGGER.error("Profile JSON not found: %s", profile_path)
        simulation_app.close()
        sys.exit(1)

    with open(profile_path, "r", encoding="utf-8") as f:
        profile = RuleProfile.from_json(f.read())

    _LOGGER.info("Loaded profile '%s' from %s", profile.profile_name, profile_path)

    # ------------------------------------------------------------------
    # 5. Process each asset
    # ------------------------------------------------------------------
    manager = AssetTransformerManager()

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for idx, relative_path in enumerate(asset_paths, start=1):
        input_stage_path = asset_root / relative_path
        output_dir = output_root / Path(relative_path).parent / Path(relative_path).stem
        report_path = output_dir / "report.json"

        # Skip already-processed assets unless --force is set
        if not args.force and report_path.is_file():
            _LOGGER.info(
                "[%d/%d] Skipping (already processed): %s",
                idx,
                len(asset_paths),
                relative_path,
            )
            successes.append(relative_path)
            continue

        _LOGGER.info(
            "[%d/%d] Processing: %s -> %s",
            idx,
            len(asset_paths),
            relative_path,
            output_dir,
        )

        if not input_stage_path.is_file():
            msg = f"Input file not found: {input_stage_path}"
            _LOGGER.error(msg)
            failures.append((relative_path, msg))
            continue

        # Remove any stale report from a previously interrupted run
        if report_path.is_file():
            report_path.unlink()

        try:
            t0 = time.time()
            report = manager.run(
                input_stage=str(input_stage_path),
                profile=profile,
                package_root=str(output_dir),
            )
            elapsed = time.time() - t0

            # Write report.json atomically so a partial file is never left behind
            output_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=str(output_dir), suffix=".tmp", prefix="report_")
            try:
                with open(fd, "w", encoding="utf-8") as f:
                    f.write(json.dumps(json.loads(report.to_json()), indent=2))
                Path(tmp_path).replace(report_path)
            except BaseException:
                Path(tmp_path).unlink(missing_ok=True)
                raise

            _LOGGER.info(
                "[%d/%d] Done in %.1fs: %s",
                idx,
                len(asset_paths),
                elapsed,
                relative_path,
            )
            successes.append(relative_path)

        except Exception as exc:
            msg = str(exc)
            _LOGGER.exception("[%d/%d] FAILED: %s", idx, len(asset_paths), relative_path)
            failures.append((relative_path, msg))
            # Ensure no report.json is left for a failed asset
            if report_path.is_file():
                report_path.unlink()

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    _LOGGER.info("=" * 60)
    _LOGGER.info(
        "Batch complete: %d succeeded, %d failed out of %d",
        len(successes),
        len(failures),
        len(asset_paths),
    )
    if failures:
        _LOGGER.info("Failed assets:")
        for path, err in failures:
            _LOGGER.info("  %s: %s", path, err)
    _LOGGER.info("=" * 60)

    # ------------------------------------------------------------------
    # 7. Shutdown
    # ------------------------------------------------------------------
    simulation_app.close()


if __name__ == "__main__":
    main()
