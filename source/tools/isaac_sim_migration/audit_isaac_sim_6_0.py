# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static inventory helper for Isaac Sim 6.0 migration planning.

Run this script against a project checkout before editing code. It reports
imports, extension dependencies, saved graph tokens, ROS workspace files, and
benchmark patterns that usually need manual migration review.

This script is not a complete migration analyzer. It can miss project wrappers,
generated files, binary USD layers, dynamic imports, custom OmniGraph nodes,
extension settings, and behavior that only appears at runtime.

By default, findings are reported as migration inventory and the script exits
successfully. Use ``--fail-on required`` or ``--fail-on any`` only when a CI
job intentionally gates on unresolved migration findings.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".ruff_cache",
    ".svn",
    ".venv",
    "__pycache__",
    "_build",
    "build",
    "dist",
    "install",
    "node_modules",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".kit",
    ".md",
    ".ogn",
    ".py",
    ".rst",
    ".toml",
    ".usda",
    ".usd",
    ".xml",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class Rule:
    category: str
    pattern: str
    message: str
    guide: str
    severity: str = "review"
    file_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    category: str
    severity: str
    match: str
    message: str
    guide: str


RULES = [
    Rule(
        "Extension namespace",
        r"\bomni\.isaac\.[A-Za-z0-9_.]*",
        "Migrate old omni.isaac extension names and imports to isaacsim names before applying the 6.0 guides.",
        "isaac_sim_4_5/extensions_renaming",
        "required",
    ),
    Rule(
        "Python runtime",
        r"\b(python_requires|requires-python|python3\.11|Python 3\.11|cp311|py311|3\.11\.13)\b",
        "Review Python runtime pins and rebuild project environments, compiled wheels, ROS Python packages, and ML dependencies for Isaac Sim 6.0's Python 3.12 runtime.",
        "isaac_sim_6_0/migration_strategy",
    ),
    Rule(
        "Core API",
        r"\bisaacsim\.core\.(api|prims|utils)\b",
        "Move Core API, prim wrappers, and utilities to isaacsim.core.experimental.* or explicit managers.",
        "isaac_sim_6_0/core_api_to_core_experimental",
        "required",
    ),
    Rule(
        "Sensor API",
        r"\bisaacsim\.sensors\.(camera|rtx|physics|physx)\b",
        "Review sensor package migration, including RTX sensor authoring, physics sensors, and OmniGraph node namespaces.",
        "isaac_sim_6_0/sensors_camera_to_experimental_rtx",
        "required",
    ),
    Rule(
        "Camera prim RTX lidar",
        r"\b(RtxLidar|IsaacRtxLidarSensorAPI|lidar_config|json_config|config_file_name|Camera[^\n]{0,80}RTX|RTX[^\n]{0,80}Camera|OmniSensorGenericLidarCoreAPI)\b",
        "If this file configures RTX lidar assets, verify Camera-prim JSON sensors were converted to OmniSensor/OmniLidar assets.",
        "isaac_sim_6_0/sensor_timing_and_rtx_lidar_asset_migration",
    ),
    Rule(
        "RTX lidar tool path",
        r"\bsource/tools/isaacsim\.sensors\.rtx\b|convert_lidar_json_to_usda\.py|set_lidar_tick_rates\.py",
        "Use the packaged 6.0 lidar conversion tools under tools/isaacsim.sensors.rtx and validate converted OmniSensor/OmniLidar assets.",
        "isaac_sim_6_0/sensor_timing_and_rtx_lidar_asset_migration",
    ),
    Rule(
        "PhysX raycast sensor",
        r"\b(_range_sensor|get_linear_depth_data|get_point_cloud_data|get_prim_data|send_next_batch|set_next_batch_rays|sensor_pattern|origin_offsets|rotationRate|curtainLength|curtainAxis|numRays|forwardAxis)\b",
        "Migrate deprecated PhysX generic, lidar, or lightbeam raycast sensors to isaacsim.sensors.experimental.physics.RaycastSensor.",
        "isaac_sim_6_0/sensors_physx_lidar_to_physics_raycast",
        "required",
    ),
    Rule(
        "ROS 2 publisher graph",
        r"\b(ROS2PublishTransformTree|ROS2PublishJointState|ROS2PublishOdometry|IsaacComputeOdometry)\b",
        "Audit saved Action Graphs for deprecated prim-resolving ROS 2 publisher inputs.",
        "isaac_sim_6_0/ros2_omnigraph_migration",
        "required",
    ),
    Rule(
        "ROS 2 publisher graph prim inputs",
        r"\b(targetPrim|targetPrims)\b",
        "Review saved USD Action Graphs for deprecated prim-resolving ROS 2 publisher inputs.",
        "isaac_sim_6_0/ros2_omnigraph_migration",
        "review",
        ("*.usd", "*.usda"),
    ),
    Rule(
        "ROS 2 sensor graph",
        r"\b(frameSkipCount|fullScan|outputNormal|outputIntensity|outputTimestamp|outputObjectId)\b",
        "Review ROS 2 sensor helper, point-cloud, and metadata-output migration.",
        "isaac_sim_6_0/ros2_sensor_graph_migration",
        "required",
    ),
    Rule(
        "Multitick timing",
        r"\bomni:sensor:tickRate\b",
        "Review multitick and external simulation time ownership; sensor tick timing changed in 6.0.",
        "isaac_sim_6_0/sensor_timing_and_rtx_lidar_asset_migration",
        "review",
    ),
    Rule(
        "ROS 2 workspace",
        r"\b(isaacsim_bringup|cmdvel_to_ackermann|h1_fullbody_controller|isaac_moveit|ament_cmake|ament_python|run_isaacsim\.launch\.py|dds_type|use_internal_libs)\b",
        "Review ROS workspace package rename, ament_python layout, launcher parameters, and distro-specific validation.",
        "isaac_sim_6_0/ros2_workspace_package_migration",
    ),
    Rule(
        "Old ROS package name",
        r"<name>\s*isaacsim\s*</name>|\bget_package_share_directory\([\"']isaacsim[\"']\)|\bros2\s+launch\s+isaacsim\b",
        "Rename downstream references from the old isaacsim ROS package to isaacsim_bringup.",
        "isaac_sim_6_0/ros2_workspace_package_migration",
        "required",
    ),
    Rule(
        "Robot asset path",
        r"/Isaac/Robots/",
        "Review robot asset references for available /Isaac/Robots_Multiphysics catalog entries and validation status.",
        "isaac_sim_6_0/robot_asset_path_migration",
    ),
    Rule(
        "URDF importer API",
        r"\b(URDFImportFromROS2Node|URDFCreateImportConfig|URDFParseText|URDFParseFile|URDFParseAndImportFile|URDFImportRobot|acquire_urdf_interface|fix_base|merge_mesh)\b|_urdf\.ImportConfig|_mjcf\.ImportConfig",
        "Review URDF importer API changes, ROS 2 robot-description import, and tri-state base behavior.",
        "isaac_sim_6_0/urdf_mjcf_importer_exporter_pipeline",
        "required",
    ),
    Rule(
        "URDF mimic joint schema",
        r"\b(PhysxMimicJointAPI|PhysxSchema\.PhysxMimicJointAPI|physxMimicJoint|PhysxMimic|physx:mimic)\b",
        "Update mimic-joint validators and exporters for NewtonMimicAPI and newton:mimicJoint/newton:mimicCoef0/newton:mimicCoef1 attributes.",
        "isaac_sim_6_0/urdf_mjcf_importer_exporter_pipeline",
        "required",
    ),
    Rule(
        "Robot setup",
        r"\b(isaacsim\.robot_setup\.wizard|isaacsim\.util\.merge_mesh|MergeMeshRule|omni\.scene\.optimizer\.core)\b",
        "Split Robot Wizard and mesh-merge workflows across supported robot setup and Scene Optimizer tools.",
        "isaac_sim_6_0/robot_setup_wizard_and_mesh_merge",
        "required",
    ),
    Rule(
        "Robot control",
        r"\bisaacsim\.robot\.(manipulators|wheeled_robots)\b",
        "Migrate robot control packages to the experimental manipulator or wheeled robot packages.",
        "isaac_sim_6_0/robot_control_extensions_to_experimental",
        "required",
    ),
    Rule(
        "Robot motion",
        r"\bisaacsim\.robot_motion\.(lula|motion_generation|lula_test_widget)\b",
        "Migrate motion-generation code to experimental motion generation, cuMotion, or PINK.",
        "isaac_sim_6_0/robot_motion_to_experimental_motion_generation",
        "required",
    ),
    Rule(
        "Replicator",
        r"\bisaacsim\.replicator\.(domain_randomization|mobility_gen|scene_blox|agent)\b|state/common/.*\.npy",
        "Review Replicator domain randomization, MobilityGen recordings, Scene Blox removal, and Agent 1.x migration.",
        "isaac_sim_6_0/replicator_domain_randomization_to_experimental",
        "required",
    ),
    Rule(
        "Domain randomization registration",
        r"\bregister_(rigid_prim|articulation)_view\s*\(",
        'In Isaac Sim 6.0, register_rigid_prim_view and register_articulation_view require an explicit name argument, e.g. register_rigid_prim_view(view, "object_view").',
        "isaac_sim_6_0/replicator_domain_randomization_to_experimental",
        "required",
    ),
    Rule(
        "Replicator Agent config",
        r"\b(scene\.asset_path|nova_carter_num|iw_hub_num|camera_num|camera_list|command_file)\b",
        "Review Isaac Replicator Agent 0.x configs for the 1.x schema, including environment.base_stage_asset_path, named actor groups, and inline routines/triggers.",
        "isaac_sim_6_0/ext_isaacsim_replicator_agent_migration_guide",
    ),
    Rule(
        "Replicator Agent config",
        r"^\s{0,4}(response|incident):\s*(?:#.*)?$",
        "Review Isaac Replicator Agent 0.x response or incident config sections for the 1.x routine and trigger schema.",
        "isaac_sim_6_0/ext_isaacsim_replicator_agent_migration_guide",
        file_patterns=("*.yaml", "*.yml"),
    ),
    Rule(
        "Benchmark services",
        r"\b(gpu_frametime\s*=|IsaacUpdateFrametimeCollector)\b",
        "Remove removed gpu_frametime constructor arguments and legacy frame-time collectors.",
        "isaac_sim_6_0/benchmark_services_migration",
        "required",
    ),
    Rule(
        "Benchmark registry import",
        r"from\s+isaacsim\.benchmark\.services\s+import[^\n]*MeasurementDataRecorderRegistry",
        "Import MeasurementDataRecorderRegistry from isaacsim.benchmark.services.datarecorders.",
        "isaac_sim_6_0/benchmark_services_migration",
        "required",
    ),
    Rule(
        "Benchmark services inventory",
        r"\b(DEFAULT_RECORDERS|MeasurementDataRecorderRegistry|BaseIsaacBenchmark)\b|\brecorders\s*=",
        "Review benchmark recorder selection and custom recorder registration during migration.",
        "isaac_sim_6_0/benchmark_services_migration",
    ),
    Rule(
        "Removed tools",
        r"\b(isaacsim\.app\.selector|isaacsim\.benchmark\.examples|isaacsim\.replicator\.scene_blox|isaacsim\.asset\.browser|omni\.isaac\.ml_archive)\b",
        "Remove no-replacement tools or migrate to the documented replacement dependency.",
        "isaac_sim_6_0/removed_tools_and_replacements",
        "required",
    ),
    Rule(
        "Extension templates",
        r"\bisaacsim\.examples\.extension\b|Extension Template Generator",
        "Generate new extensions with ./repo.sh template new instead of the deprecated extension template UI.",
        "isaac_sim_6_0/extension_template_generator_to_cli_templates",
        "required",
    ),
]


def is_binary_usd_crate(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(8) == b"PXR-USDC"
    except OSError:
        return False


def should_scan(path: Path, relative_parts: tuple[str, ...]) -> bool:
    if any(part in SKIP_DIRS for part in relative_parts):
        return False
    if path.suffix == ".usd" and is_binary_usd_crate(path):
        return False
    return path.suffix in TEXT_SUFFIXES


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and should_scan(path, path.relative_to(root).parts):
            yield path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def scan_file(root: Path, path: Path, rules: list[tuple[Rule, re.Pattern[str]]]) -> list[Finding]:
    relative = path.relative_to(root).as_posix()
    try:
        text = read_text(path)
    except OSError:
        return []

    findings: list[Finding] = []
    applicable_rules = [
        (rule, regex)
        for rule, regex in rules
        if not rule.file_patterns or any(path.match(pattern) for pattern in rule.file_patterns)
    ]
    for line_number, line in enumerate(text.splitlines(), start=1):
        for rule, regex in applicable_rules:
            match = regex.search(line)
            if match:
                findings.append(
                    Finding(
                        file=relative,
                        line=line_number,
                        category=rule.category,
                        severity=rule.severity,
                        match=match.group(0),
                        message=rule.message,
                        guide=rule.guide,
                    )
                )
    return findings


def scan(root: Path) -> list[Finding]:
    compiled = [(rule, re.compile(rule.pattern)) for rule in RULES]
    findings: list[Finding] = []
    for path in iter_files(root):
        findings.extend(scan_file(root, path, compiled))
    return findings


def print_text(findings: list[Finding], max_results: int) -> None:
    shown = findings if max_results == 0 else findings[:max_results]
    if not findings:
        print("No Isaac Sim 6.0 migration patterns were found.")
        return

    for finding in shown:
        print(f"{finding.severity.upper()}: {finding.file}:{finding.line}: {finding.category}")
        print(f"  match: {finding.match}")
        print(f"  action: {finding.message}")
        print(f"  guide: {finding.guide}")

    hidden = len(findings) - len(shown)
    if hidden > 0:
        print(f"\n{hidden} additional findings hidden. Re-run with --max-results 0 to show all findings.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Project root to scan.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument("--max-results", type=int, default=200, help="Maximum findings to print; use 0 for all.")
    parser.add_argument(
        "--fail-on",
        choices=("none", "required", "any"),
        default="none",
        help=(
            "Exit nonzero for matching findings. The default, 'none', treats "
            "the report as migration inventory rather than a CI gate."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Path is not a directory: {root}")

    findings = scan(root)
    findings.sort(key=lambda item: (item.severity != "required", item.file, item.line, item.category))
    if args.json:
        print(json.dumps([asdict(finding) for finding in findings], indent=2))
    else:
        print_text(findings, args.max_results)
    if args.fail_on == "any" and findings:
        return 1
    if args.fail_on == "required" and any(finding.severity == "required" for finding in findings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
