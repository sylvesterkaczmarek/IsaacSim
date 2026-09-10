// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "isaacsim/common/array/nanobind/ArrayCaster.hpp"
#include "isaacsim/physics/entities/ArticulationEntity.hpp"
#include "isaacsim/physics/entities/PhysicsEntity.hpp"
#include "isaacsim/physics/entities/RigidBodyEntity.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>

namespace nb = nanobind;
using namespace isaacsim::physics::entities;

NB_MODULE(_bindings, m)
{
    // PhysicsEntity.hpp
    nb::class_<PhysicsEntity>(m, "PhysicsEntity")
        .def(
            "__init__",
            [](PhysicsEntity* self, const std::string& engine, const std::string& entity,
               const std::variant<std::string, std::vector<std::string>>& paths)
            { new (self) PhysicsEntity(engine, entity, paths); },
            nb::arg("engine"), nb::arg("entity"), nb::arg("paths"))
        .def_prop_ro("num_prims", &PhysicsEntity::numPrims)
        .def("get_data", &PhysicsEntity::getData, nb::arg("name"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_data", &PhysicsEntity::setData, nb::arg("name"), nb::arg("data"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_multi_data", &PhysicsEntity::getMultiData, nb::arg("name"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_multi_data", &PhysicsEntity::setMultiData, nb::arg("name"), nb::arg("data"), nb::kw_only(),
             nb::arg("indices") = nb::none());

    // RigidBodyEntity.hpp
    nb::class_<RigidBodyEntity, PhysicsEntity>(m, "RigidBodyEntity")
        .def(
            "__init__",
            [](RigidBodyEntity* self, const std::string& engine,
               const std::variant<std::string, std::vector<std::string>>& paths)
            { new (self) RigidBodyEntity(engine, paths); },
            nb::arg("engine"), nb::arg("paths"))
        .def_prop_ro("num_shapes", &RigidBodyEntity::numShapes)
        .def("get_world_poses", &RigidBodyEntity::getWorldPoses, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_world_poses", &RigidBodyEntity::setWorldPoses, nb::arg("positions") = nb::none(),
             nb::arg("orientations") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_velocities", &RigidBodyEntity::getVelocities, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_velocities", &RigidBodyEntity::setVelocities, nb::arg("linear_velocities") = nb::none(),
             nb::arg("angular_velocities") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("apply_forces", &RigidBodyEntity::applyForces, nb::arg("forces"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("apply_forces_and_torques_at_positions", &RigidBodyEntity::applyForcesAndTorquesAtPositions,
             nb::arg("forces") = nb::none(), nb::arg("torques") = nb::none(), nb::arg("positions") = nb::none(),
             nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_masses", &RigidBodyEntity::getMasses, nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("inverse") = false)
        .def("set_masses", &RigidBodyEntity::setMasses, nb::arg("masses"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_inertias", &RigidBodyEntity::getInertias, nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("inverse") = false)
        .def("set_inertias", &RigidBodyEntity::setInertias, nb::arg("inertias"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_coms", &RigidBodyEntity::getComs, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_coms", &RigidBodyEntity::setComs, nb::arg("positions") = nb::none(),
             nb::arg("orientations") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_enabled_rigid_bodies", &RigidBodyEntity::getEnabledRigidBodies, nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_enabled_rigid_bodies", &RigidBodyEntity::setEnabledRigidBodies, nb::arg("enabled"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_enabled_gravities", &RigidBodyEntity::getEnabledGravities, nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_enabled_gravities", &RigidBodyEntity::setEnabledGravities, nb::arg("enabled"), nb::kw_only(),
             nb::arg("indices") = nb::none());

    // ArticulationEntity.hpp
    nb::class_<ArticulationEntity, PhysicsEntity>(m, "ArticulationEntity")
        .def(
            "__init__",
            [](ArticulationEntity* self, const std::string& engine,
               const std::variant<std::string, std::vector<std::string>>& paths)
            { new (self) ArticulationEntity(engine, paths); },
            nb::arg("engine"), nb::arg("paths"))
        .def_prop_ro("num_dofs", &ArticulationEntity::numDofs)
        .def_prop_ro("dof_names", &ArticulationEntity::dofNames)
        .def_prop_ro("dof_paths", &ArticulationEntity::dofPaths)
        .def_prop_ro("dof_types", &ArticulationEntity::dofTypes)
        .def_prop_ro("num_joints", &ArticulationEntity::numJoints)
        .def_prop_ro("joint_names", &ArticulationEntity::jointNames)
        .def_prop_ro("joint_paths", &ArticulationEntity::jointPaths)
        .def_prop_ro("joint_types", &ArticulationEntity::jointTypes)
        .def_prop_ro("num_links", &ArticulationEntity::numLinks)
        .def_prop_ro("link_names", &ArticulationEntity::linkNames)
        .def_prop_ro("link_paths", &ArticulationEntity::linkPaths)
        .def_prop_ro("num_shapes", &ArticulationEntity::numShapes)
        .def_prop_ro("num_fixed_tendons", &ArticulationEntity::numFixedTendons)
        .def_prop_ro("jacobian_matrix_shape", &ArticulationEntity::jacobianMatrixShape)
        .def_prop_ro("mass_matrix_shape", &ArticulationEntity::massMatrixShape)
        .def("get_link_indices", &ArticulationEntity::getLinkIndices, nb::arg("names"))
        .def("get_joint_indices", &ArticulationEntity::getJointIndices, nb::arg("names"))
        .def("get_dof_indices", &ArticulationEntity::getDofIndices, nb::arg("names"))
        .def("get_dof_limits", &ArticulationEntity::getDofLimits, nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("dof_indices") = nb::none())
        .def("set_dof_limits", &ArticulationEntity::setDofLimits, nb::arg("lower") = nb::none(),
             nb::arg("upper") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("dof_indices") = nb::none())
        .def("get_dof_friction_properties", &ArticulationEntity::getDofFrictionProperties, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("set_dof_friction_properties", &ArticulationEntity::setDofFrictionProperties,
             nb::arg("static_frictions") = nb::none(), nb::arg("dynamic_frictions") = nb::none(),
             nb::arg("viscous_frictions") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("dof_indices") = nb::none())
        .def("get_dof_drive_model_properties", &ArticulationEntity::getDofDriveModelProperties, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("set_dof_drive_model_properties", &ArticulationEntity::setDofDriveModelProperties,
             nb::arg("speed_effort_gradients") = nb::none(), nb::arg("maximum_actuator_velocities") = nb::none(),
             nb::arg("velocity_dependent_resistances") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("dof_indices") = nb::none())
        .def("get_dof_armatures", &ArticulationEntity::getDofArmatures, nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("dof_indices") = nb::none())
        .def("set_dof_armatures", &ArticulationEntity::setDofArmatures, nb::arg("armatures"), nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_gains", &ArticulationEntity::getDofGains, nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("dof_indices") = nb::none())
        .def("set_dof_gains", &ArticulationEntity::setDofGains, nb::arg("stiffnesses") = nb::none(),
             nb::arg("dampings") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("dof_indices") = nb::none(), nb::arg("update_default_gains") = true)
        .def("switch_dof_control_mode", &ArticulationEntity::switchDofControlMode, nb::arg("mode"), nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_max_efforts", &ArticulationEntity::getDofMaxEfforts, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("set_dof_max_efforts", &ArticulationEntity::setDofMaxEfforts, nb::arg("max_efforts"), nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_max_velocities", &ArticulationEntity::getDofMaxVelocities, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("set_dof_max_velocities", &ArticulationEntity::setDofMaxVelocities, nb::arg("max_velocities"),
             nb::kw_only(), nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_drive_types", &ArticulationEntity::getDofDriveTypes, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_position_targets", &ArticulationEntity::getDofPositionTargets, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("set_dof_position_targets", &ArticulationEntity::setDofPositionTargets, nb::arg("positions"),
             nb::kw_only(), nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_positions", &ArticulationEntity::getDofPositions, nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("dof_indices") = nb::none())
        .def("set_dof_positions", &ArticulationEntity::setDofPositions, nb::arg("positions"), nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_velocity_targets", &ArticulationEntity::getDofVelocityTargets, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("set_dof_velocity_targets", &ArticulationEntity::setDofVelocityTargets, nb::arg("velocities"),
             nb::kw_only(), nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_velocities", &ArticulationEntity::getDofVelocities, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("set_dof_velocities", &ArticulationEntity::setDofVelocities, nb::arg("velocities"), nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_efforts", &ArticulationEntity::getDofEfforts, nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("dof_indices") = nb::none())
        .def("set_dof_efforts", &ArticulationEntity::setDofEfforts, nb::arg("efforts"), nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_projected_joint_forces", &ArticulationEntity::getDofProjectedJointForces, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_world_poses", &ArticulationEntity::getWorldPoses, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_world_poses", &ArticulationEntity::setWorldPoses, nb::arg("positions") = nb::none(),
             nb::arg("orientations") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_velocities", &ArticulationEntity::getVelocities, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_velocities", &ArticulationEntity::setVelocities, nb::arg("linear_velocities") = nb::none(),
             nb::arg("angular_velocities") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_link_incoming_joint_force", &ArticulationEntity::getLinkIncomingJointForce, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("link_indices") = nb::none())
        .def("get_jacobian_matrices", &ArticulationEntity::getJacobianMatrices, nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_mass_matrices", &ArticulationEntity::getMassMatrices, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_dof_coriolis_and_centrifugal_compensation_forces",
             &ArticulationEntity::getDofCoriolisAndCentrifugalCompensationForces, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_gravity_compensation_forces", &ArticulationEntity::getDofGravityCompensationForces, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_link_masses", &ArticulationEntity::getLinkMasses, nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("link_indices") = nb::none(), nb::arg("inverse") = false)
        .def("set_link_masses", &ArticulationEntity::setLinkMasses, nb::arg("masses"), nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("link_indices") = nb::none())
        .def("get_link_coms", &ArticulationEntity::getLinkComs, nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("link_indices") = nb::none())
        .def("set_link_coms", &ArticulationEntity::setLinkComs, nb::arg("positions") = nb::none(),
             nb::arg("orientations") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("link_indices") = nb::none())
        .def("get_link_inertias", &ArticulationEntity::getLinkInertias, nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("link_indices") = nb::none(), nb::arg("inverse") = false)
        .def("set_link_inertias", &ArticulationEntity::setLinkInertias, nb::arg("inertias"), nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("link_indices") = nb::none())
        .def("get_link_enabled_gravities", &ArticulationEntity::getLinkEnabledGravities, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("link_indices") = nb::none())
        .def("set_link_enabled_gravities", &ArticulationEntity::setLinkEnabledGravities, nb::arg("enabled"),
             nb::kw_only(), nb::arg("indices") = nb::none(), nb::arg("link_indices") = nb::none())
        .def("get_fixed_tendon_stiffnesses", &ArticulationEntity::getFixedTendonStiffnesses, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("tendon_indices") = nb::none())
        .def("get_fixed_tendon_dampings", &ArticulationEntity::getFixedTendonDampings, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("tendon_indices") = nb::none())
        .def("get_fixed_tendon_limit_stiffnesses", &ArticulationEntity::getFixedTendonLimitStiffnesses, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("tendon_indices") = nb::none())
        .def("get_fixed_tendon_limits", &ArticulationEntity::getFixedTendonLimits, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("tendon_indices") = nb::none())
        .def("get_fixed_tendon_rest_lengths", &ArticulationEntity::getFixedTendonRestLengths, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("tendon_indices") = nb::none())
        .def("get_fixed_tendon_offsets", &ArticulationEntity::getFixedTendonOffsets, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("tendon_indices") = nb::none())
        .def("set_fixed_tendon_properties", &ArticulationEntity::setFixedTendonProperties, nb::kw_only(),
             nb::arg("stiffnesses") = nb::none(), nb::arg("dampings") = nb::none(),
             nb::arg("limit_stiffnesses") = nb::none(), nb::arg("lower_limits") = nb::none(),
             nb::arg("upper_limits") = nb::none(), nb::arg("rest_lengths") = nb::none(),
             nb::arg("offsets") = nb::none(), nb::arg("indices") = nb::none(), nb::arg("tendon_indices") = nb::none());
}
