# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Batch-validate a list of USD assets using omni.asset_validator.core.

This standalone script launches a headless SimulationApp, enables the asset
validation extensions, and validates every asset listed in a text file against
the configured rule categories.

Usage::

    python.sh source/extensions/isaacsim.asset.validation/scripts/run_asset_validation.py \\
        asset_list.txt output_dir --asset-root /path/to/assets

Examples::

    # Validate with the Isaac Sim robot defaults (SimReady physics, robot, and geometry categories)
    python.sh .../run_asset_validation.py robots.txt output/ --asset-root /data/assets

    # Specific rules (format: Category.RuleName)
    python.sh .../run_asset_validation.py robots.txt output/ -r RobotCore.CleanFolder

    # Use ALL registered rules
    python.sh .../run_asset_validation.py robots.txt output/ --init-rules

    # Dry-run: show what would be validated without launching SimulationApp
    python.sh .../run_asset_validation.py robots.txt output/ --pretend
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_LOGGER = logging.getLogger(__name__)

DEFAULT_CATEGORIES = [
    "BaseArticulation",
    "ColliderApproximations",
    "Geometry",
    "Hierarchy",
    "IsaacComposition",
    "IsaacSim.SensorRules",
    "NamingPaths",
    "PhysicsDrivenJoints",
    "PhysicsGraspable",
    "PhysicsGrippers",
    "PhysicsJoints",
    "PhysicsMaterials",
    "PhysicsRigidBodies",
    "RobotCore",
    "RobotMaterials",
    "Units",
]


class AssetInfo(NamedTuple):
    """Asset information for the validation pipeline."""

    rel_path: str
    full_path: str
    csv_name: str


class ValidationResult(NamedTuple):
    """Per-asset validation result counts."""

    num_errors: int
    num_warnings: int


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Validate USD assets using omni.asset_validator.core.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Validate with Isaac Sim defaults\n"
            "  run_asset_validation.py robots.txt output/ --asset-root /path/to/assets\n"
            "\n"
            "  # Specific rules (format: Category.RuleName)\n"
            "  run_asset_validation.py robots.txt output/ -r RobotCore.CleanFolder\n"
            "\n"
            "  # Use ALL registered rules\n"
            "  run_asset_validation.py robots.txt output/ --init-rules\n"
        ),
    )
    parser.add_argument(
        "asset_list",
        help="Text file listing USD assets (one per line, relative paths).",
    )
    parser.add_argument(
        "output_dir",
        help="Directory for CSV output files and summary.json.",
    )
    parser.add_argument(
        "--asset-root",
        default=None,
        help="Prefix path prepended to each asset in the list.",
    )
    parser.add_argument(
        "-c",
        "--category",
        action="append",
        dest="categories",
        metavar="CATEGORY",
        help="Enable a rule category (can be repeated). Replaces defaults when specified.",
    )
    parser.add_argument(
        "-r",
        "--rule",
        action="append",
        dest="rules",
        metavar="CATEGORY.RULE",
        help="Enable a specific rule as Category.RuleName (can be repeated).",
    )
    parser.add_argument(
        "--init-rules",
        action="store_true",
        help="Use ALL registered rules (ignores -c and -r).",
    )
    parser.add_argument(
        "--no-init-rules",
        action="store_true",
        help="Do not load default rules (this is the default behavior).",
    )
    parser.add_argument(
        "--pretend",
        action="store_true",
        help="Show what would be validated without launching SimulationApp.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-validate all assets even if output CSV already exists.",
    )
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Path to an alternate Isaac Sim repo checkout. Adds that repo's source/extensions/ to Kit's extension search paths so validation rules come from the alternate repo.",
    )
    return parser.parse_args()


def _get_asset_list(asset_list_file: str, asset_root: str | None) -> list[AssetInfo]:
    """Read the asset list file and build full paths and CSV file names.

    Args:
        asset_list_file: Path to a text file with one asset per line.
        asset_root: Optional prefix path prepended to each asset.

    Returns:
        List of ``AssetInfo`` entries.
    """
    assets: list[AssetInfo] = []
    with open(asset_list_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            rel_path = raw_line.strip()
            if not rel_path or rel_path.startswith("#"):
                continue

            full_path = rel_path
            if asset_root:
                full_path = f"{asset_root}/{rel_path}"

            csv_name = rel_path.replace(os.sep, "_").replace(" ", "_") + ".csv"
            assets.append(AssetInfo(rel_path, full_path, csv_name))

    return assets


def _validate_single_asset(
    asset_path: str,
    csv_path: str,
    init_rules: bool,
    categories: list[str] | None,
    rules: list[str] | None,
) -> ValidationResult:
    """Validate a single asset and write per-asset CSV results.

    This function performs a late import of ``omni.asset_validator.core`` modules
    because they are only available after ``SimulationApp`` has been initialized.

    Args:
        asset_path: Full path to the USD asset.
        csv_path: Output path for the per-asset CSV report.
        init_rules: If ``True``, use all registered validation rules.
        categories: Rule categories to enable (when *init_rules* is ``False``).
        rules: Specific rules to enable as ``Category.RuleName`` strings.

    Returns:
        A ``ValidationResult`` with error and warning counts.
    """
    from omni.asset_validator.core import IssueSeverity, ValidationEngine, ValidationRulesRegistry

    if init_rules:
        engine = ValidationEngine(initRules=True)
    else:
        engine = ValidationEngine(initRules=False)
        registry = ValidationRulesRegistry()

        if categories:
            for category in categories:
                for rule_class in registry.rules(category=category):
                    engine.enable_rule(rule_class)

        if rules:
            for rule_spec in rules:
                if "." not in rule_spec:
                    _LOGGER.warning("Invalid rule format '%s', expected 'Category.RuleName'", rule_spec)
                    continue
                category, rule_name = rule_spec.rsplit(".", 1)
                category_rules = {r.__name__: r for r in registry.rules(category=category)}
                if rule_name in category_rules:
                    engine.enable_rule(category_rules[rule_name])
                else:
                    _LOGGER.warning("Rule '%s' not found in category '%s'", rule_name, category)

    try:
        issues = engine.validate(asset_path)
    except Exception:
        _LOGGER.exception("Error validating %s", asset_path)
        issues = []

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Asset", "Rule", "Message", "Severity", "Suggestion", "Location"])

        if not issues:
            writer.writerow([asset_path, "None", "No issues found", "Info", "", ""])
        else:
            for issue in issues:
                rule_name = issue.rule.__name__ if issue.rule else "Unknown"
                writer.writerow(
                    [
                        asset_path,
                        rule_name,
                        issue.message,
                        issue.severity.name,
                        issue.suggestion or "",
                        str(issue.at) if issue.at else "",
                    ]
                )

    num_errors = sum(1 for i in issues if i.severity == IssueSeverity.ERROR)
    num_warnings = sum(1 for i in issues if i.severity == IssueSeverity.WARNING)

    return ValidationResult(num_errors, num_warnings)


def _write_summary_json(summary_path: Path, summary_data: dict) -> None:
    """Write summary.json atomically using a temporary file and os.replace.

    Args:
        summary_path: Target path for the summary JSON file.
        summary_data: Dictionary to serialize as JSON.
    """
    fd, tmp_path = tempfile.mkstemp(dir=str(summary_path.parent), suffix=".tmp", prefix="summary_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        os.replace(tmp_path, str(summary_path))
    except BaseException:
        os.unlink(tmp_path)
        raise


def main() -> None:
    """Entry point for the batch asset validation script."""

    # ------------------------------------------------------------------
    # 1. Parse args and read asset list (fast-fail on bad input)
    # ------------------------------------------------------------------
    args = _parse_args()

    if not os.path.isfile(args.asset_list):
        _LOGGER.error("Asset list file not found: %s", args.asset_list)
        sys.exit(1)

    assets = _get_asset_list(args.asset_list, args.asset_root)
    if not assets:
        _LOGGER.error("No assets found in %s", args.asset_list)
        sys.exit(1)

    # Determine rule configuration
    init_rules = args.init_rules
    if args.no_init_rules:
        init_rules = False
    categories = args.categories
    rules = args.rules

    if not init_rules and not categories and not rules:
        categories = list(DEFAULT_CATEGORIES)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    _LOGGER.info("Found %d asset(s) to process.", len(assets))

    # ------------------------------------------------------------------
    # 2. Handle --pretend mode (early exit, no SimulationApp)
    # ------------------------------------------------------------------
    if args.pretend:
        _LOGGER.info("PRETEND MODE - not running validations")
        _LOGGER.info("Would validate %d assets", len(assets))
        if init_rules:
            _LOGGER.info("Using: --init-rules (all registered rules)")
        else:
            _LOGGER.info("Categories: %s", categories or "none")
            _LOGGER.info("Rules: %s", rules or "none")
        for asset in assets:
            csv_path = Path(output_dir) / asset.csv_name
            _LOGGER.info("  %s -> %s", asset.full_path, csv_path)
        return

    # ------------------------------------------------------------------
    # 3. Launch headless SimulationApp and enable extensions
    # ------------------------------------------------------------------
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})

    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()

    # When --repo-path is set, prepend that repo's extension directories so Kit
    # loads validation/transformer rules from the alternate checkout.
    if args.repo_path is not None:
        from pathlib import Path as _Path

        repo = _Path(args.repo_path).resolve()
        for ext_dir in [
            repo / "source" / "extensions",
        ]:
            if ext_dir.is_dir():
                _LOGGER.info("Adding extension search path: %s", ext_dir)
                ext_manager.add_path(str(ext_dir))

    ext_manager.set_extension_enabled_immediate("omni.asset_validator.core", True)
    ext_manager.set_extension_enabled_immediate("isaacsim.asset.validation", True)

    # ------------------------------------------------------------------
    # 4. Process each asset (skip/resume, validate, write CSV)
    # ------------------------------------------------------------------
    summary_assets: list[dict] = []
    total_errors = 0
    total_warnings = 0

    for idx, asset in enumerate(assets, start=1):
        csv_path = Path(output_dir) / asset.csv_name

        # Skip check: existing non-empty CSV unless --force
        if not args.force and csv_path.is_file() and csv_path.stat().st_size > 0:
            _LOGGER.info("[%d/%d] Skipping (already validated): %s", idx, len(assets), asset.rel_path)
            summary_assets.append(
                {
                    "rel_path": asset.rel_path,
                    "status": "skipped",
                    "num_errors": 0,
                    "num_warnings": 0,
                }
            )
            continue

        # File-not-found check
        if not Path(asset.full_path).is_file():
            msg = f"File not found: {asset.full_path}"
            _LOGGER.error("[%d/%d] %s", idx, len(assets), msg)
            summary_assets.append(
                {
                    "rel_path": asset.rel_path,
                    "status": "failed",
                    "error": msg,
                    "num_errors": 0,
                    "num_warnings": 0,
                }
            )
            continue

        _LOGGER.info("[%d/%d] Validating: %s", idx, len(assets), asset.full_path)

        try:
            result = _validate_single_asset(
                asset.full_path,
                str(csv_path),
                init_rules,
                categories,
                rules,
            )
        except Exception:
            _LOGGER.exception("[%d/%d] FAILED: %s", idx, len(assets), asset.rel_path)
            summary_assets.append(
                {
                    "rel_path": asset.rel_path,
                    "status": "failed",
                    "error": "Unhandled exception during validation",
                    "num_errors": 0,
                    "num_warnings": 0,
                }
            )
            continue

        _LOGGER.info(
            "[%d/%d] Done: %s (%d errors, %d warnings)",
            idx,
            len(assets),
            asset.rel_path,
            result.num_errors,
            result.num_warnings,
        )
        total_errors += result.num_errors
        total_warnings += result.num_warnings
        summary_assets.append(
            {
                "rel_path": asset.rel_path,
                "status": "validated",
                "num_errors": result.num_errors,
                "num_warnings": result.num_warnings,
                "csv_file": asset.csv_name,
            }
        )

        simulation_app.update()

    # ------------------------------------------------------------------
    # 5. Write summary.json atomically
    # ------------------------------------------------------------------
    processed_count = sum(1 for a in summary_assets if a["status"] == "validated")
    skipped_count = sum(1 for a in summary_assets if a["status"] == "skipped")
    failed_count = sum(1 for a in summary_assets if a["status"] == "failed")

    summary_data = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_assets": len(assets),
        "processed": processed_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "assets": summary_assets,
    }

    summary_path = Path(output_dir) / "summary.json"
    _write_summary_json(summary_path, summary_data)
    _LOGGER.info("Summary written to %s", summary_path)

    # ------------------------------------------------------------------
    # 6. Print summary
    # ------------------------------------------------------------------
    _LOGGER.info("=" * 60)
    _LOGGER.info(
        "Validation complete: %d processed, %d skipped, %d failed out of %d",
        processed_count,
        skipped_count,
        failed_count,
        len(assets),
    )
    for entry in summary_assets:
        if entry["status"] == "validated":
            _LOGGER.info(
                "  %s: %d errors, %d warnings",
                entry["rel_path"],
                entry["num_errors"],
                entry["num_warnings"],
            )
        elif entry["status"] == "failed":
            _LOGGER.info("  %s: FAILED - %s", entry["rel_path"], entry.get("error", "unknown"))
        else:
            _LOGGER.info("  %s: skipped", entry["rel_path"])
    _LOGGER.info("Totals: %d errors, %d warnings", total_errors, total_warnings)
    _LOGGER.info("=" * 60)

    # ------------------------------------------------------------------
    # 7. Shutdown
    # ------------------------------------------------------------------
    simulation_app.close()


if __name__ == "__main__":
    main()
