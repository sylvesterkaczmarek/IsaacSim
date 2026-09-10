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

"""Migration-only shims for policy classes removed in extension version 7.0.0."""

_MIGRATION_GUIDE = "isaacsim.robot.policy.examples/docs/Overview.md, 'Migrating from extension 6.x'"
_IO_DESCRIPTOR_GUIDE = (
    "https://isaac-sim.github.io/IsaacLab/main/source/policy_deployment/01_io_descriptors/io_descriptors_101.html"
)

_DESCRIPTOR_GUIDANCE = f"""
descriptor_path is the path or omniverse:// URL of IO_descriptors.yaml exported from the same
manager-based Isaac Lab task used to train the model; it is not env.yaml. Export an existing task with:

./isaaclab.sh -p scripts/environments/export_io_descriptors.py --task <task_name> --output_dir <output_dir>

Or add --export_io_descriptors to the Isaac Lab training command. The generated file is
<output_dir>/IO_descriptors.yaml. Isaac Lab guide: {_IO_DESCRIPTOR_GUIDE}
""".strip()


def _runner_migration(
    class_name: str,
    spec_factory: str,
    imports: str,
    construction: str,
    step_call: str,
) -> str:
    return f"""
{class_name} was removed in isaacsim.robot.policy.examples 7.0.0 and can no longer run a policy.

Use RobotPolicyRunner with {spec_factory}() instead.

Replacement code:

{imports}

{construction}
runner.spawn()

Required API changes:
1. Create the runner and call runner.spawn() while the timeline is stopped.
2. After physics starts, replace policy.initialize() with runner.initialize().
3. Replace policy.forward(...) with {step_call}, called once per physics tick.
4. Replace policy.robot with runner.articulation.
5. After resetting or teleporting runner.articulation, call runner.initialize() to reset policy state.
6. Call runner.close() during teardown.

Constructor argument changes:
- Pass prim_path, position, and orientation to RobotPolicyRunner.
- Remove root_path; RobotPolicyRunner resolves the articulation from prim_path.
- Move a custom usd_path to {spec_factory}(usd_path=usd_path).
- A custom policy_path or env_config_path now requires a PolicyArtifact and IO_descriptors.yaml.

{_DESCRIPTOR_GUIDANCE}

Full migration guide: {_MIGRATION_GUIDE}
""".strip()


class _RemovedPolicyClass:
    """Base for legacy names that provide an actionable migration failure.

    Args:
        *_args: Positional arguments accepted and ignored.
        **_kwargs: Keyword arguments accepted and ignored.
    """

    _migration: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise RuntimeError(self._migration)


class PolicyController(_RemovedPolicyClass):
    """Migration shim for the removed articulation-owning policy controller."""

    _migration = f"""
PolicyController was removed in isaacsim.robot.policy.examples 7.0.0 and can no longer be used as a policy base class.

For end-to-end policy deployment, use RobotPolicyRunner with PolicySpec and PolicyArtifact instead:

from isaacsim.robot.policy.examples import PolicyArtifact, PolicySpec, RobotPolicyRunner

training_engine = "physx"  # Or "newton".
artifact = PolicyArtifact.from_files(
    policy_path,
    env_config_path,
    descriptor_path,
    model_sha256=trusted_release_manifest["policy.pt"],
)
spec = PolicySpec(name="my_policy", usd_path=usd_path, engines={{training_engine: artifact}})
runner = RobotPolicyRunner(spec, prim_path=prim_path, training_engine=training_engine)
runner.spawn()

Required API changes:
1. Call runner.spawn() while the timeline is stopped and runner.initialize() after physics starts.
2. Replace the custom forward loop with runner.step(dt, command), called once per physics tick.
3. Replace policy.robot with runner.articulation.
4. Reset or teleport the articulation before calling runner.initialize() for a replay.
5. Call runner.close() during teardown.

PolicyArtifact.from_files() requires the model, env config, and exported IO_descriptors.yaml.
TorchScript also requires its full SHA-256 from a trusted release manifest. Supply an explicit
PolicySpec binding only when the IO descriptor cannot represent the policy interface.

{_DESCRIPTOR_GUIDANCE}

Use IsaacLabPolicyController instead only if your application already owns the articulation
lifecycle and directly exchanges isaacsim.robot_motion.experimental.motion_generation.RobotState
values. It is not a drop-in replacement for PolicyController.

Full migration guide: {_MIGRATION_GUIDE}
""".strip()


class AnymalFlatTerrainPolicy(PolicyController):
    """Migration shim for the removed ANYmal policy class."""

    _migration = _runner_migration(
        "AnymalFlatTerrainPolicy",
        "get_anymal_spec",
        "from isaacsim.robot.policy.examples import RobotPolicyRunner, get_anymal_spec",
        """runner = RobotPolicyRunner(
    get_anymal_spec(), prim_path=prim_path, position=position, orientation=orientation
)""",
        "runner.step(dt, command)",
    )


class CartpolePolicy(PolicyController):
    """Migration shim for the removed Cartpole policy class."""

    _migration = _runner_migration(
        "CartpolePolicy",
        "get_cartpole_spec",
        "from isaacsim.robot.policy.examples import RobotPolicyRunner, get_cartpole_spec",
        """runner = RobotPolicyRunner(
    get_cartpole_spec(), prim_path=prim_path, position=position, orientation=orientation
)""",
        "runner.step(dt) without a command",
    )


class FrankaOpenDrawerPolicy(PolicyController):
    """Migration shim for the removed Franka drawer policy class."""

    _migration = _runner_migration(
        "FrankaOpenDrawerPolicy",
        "get_franka_spec",
        """from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.policy.examples import (
    PolicyEnvConfig,
    RobotPolicyRunner,
    get_franka_spec,
    make_franka_task_state_provider,
)""",
        """spec = get_franka_spec()
engine = (SimulationManager.get_active_physics_engine() or "").lower()
env_config = PolicyEnvConfig.from_file(spec.engines[engine].env_config_path)
runner = RobotPolicyRunner(
    spec,
    prim_path=prim_path,
    position=position,
    orientation=orientation,
    task_state_provider=make_franka_task_state_provider(cabinet, env_config),
)""",
        "runner.step(dt) without a command",
    )


class Go2FlatTerrainPolicy(PolicyController):
    """Migration shim for the removed Go2 policy class."""

    _migration = _runner_migration(
        "Go2FlatTerrainPolicy",
        "get_go2_spec",
        "from isaacsim.robot.policy.examples import RobotPolicyRunner, get_go2_spec",
        """runner = RobotPolicyRunner(
    get_go2_spec(), prim_path=prim_path, position=position, orientation=orientation
)""",
        "runner.step(dt, command)",
    )


class H1FlatTerrainPolicy(PolicyController):
    """Migration shim for the removed H1 policy class."""

    _migration = _runner_migration(
        "H1FlatTerrainPolicy",
        "get_h1_spec",
        "from isaacsim.robot.policy.examples import RobotPolicyRunner, get_h1_spec",
        """runner = RobotPolicyRunner(
    get_h1_spec(), prim_path=prim_path, position=position, orientation=orientation
)""",
        "runner.step(dt, command)",
    )


class SpotFlatTerrainPolicy(PolicyController):
    """Migration shim for the removed Spot policy class."""

    _migration = _runner_migration(
        "SpotFlatTerrainPolicy",
        "get_spot_spec",
        "from isaacsim.robot.policy.examples import RobotPolicyRunner, get_spot_spec",
        """runner = RobotPolicyRunner(
    get_spot_spec(), prim_path=prim_path, position=position, orientation=orientation
)""",
        "runner.step(dt, command)",
    )
