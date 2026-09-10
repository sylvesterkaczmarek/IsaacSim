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
#include "isaacsim/foundation/objects/Xform.hpp"
#include "isaacsim/foundation/prims/physics/Articulation.hpp"
#include "isaacsim/foundation/prims/physics/ColliderBody.hpp"
#include "isaacsim/foundation/prims/physics/RigidBody.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <variant>

namespace nb = nanobind;
using namespace isaacsim::foundation::objects;
using namespace isaacsim::foundation::prims;

NB_MODULE(_bindings, m)
{
    // physics/Articulation.hpp
    nb::class_<physics::Articulation, Xform>(m, "Articulation")
        .def(
            "__init__",
            [](physics::Articulation* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<array::Array>& positions, const std::optional<array::Array>& translations,
               const std::optional<array::Array>& orientations, const std::optional<array::Array>& scales,
               bool resetXformOpProperties) {
                new (self)
                    physics::Articulation(paths, positions, translations, orientations, scales, resetXformOpProperties);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("positions") = nb::none(), nb::arg("translations") = nb::none(),
            nb::arg("orientations") = nb::none(), nb::arg("scales") = nb::none(),
            nb::arg("reset_xform_op_properties") = true)
        .def_prop_ro("root_paths", &physics::Articulation::rootPaths)
        .def_prop_ro("num_dofs", &physics::Articulation::numDofs)
        .def_prop_ro("dof_names", &physics::Articulation::dofNames)
        .def_prop_ro("dof_paths", &physics::Articulation::dofPaths)
        .def_prop_ro("dof_types", &physics::Articulation::dofTypes)
        .def_prop_ro("num_joints", &physics::Articulation::numJoints)
        .def_prop_ro("joint_names", &physics::Articulation::jointNames)
        .def_prop_ro("joint_paths", &physics::Articulation::jointPaths)
        .def_prop_ro("joint_types", &physics::Articulation::jointTypes)
        .def_prop_ro("num_links", &physics::Articulation::numLinks)
        .def_prop_ro("link_names", &physics::Articulation::linkNames)
        .def_prop_ro("link_paths", &physics::Articulation::linkPaths)
        .def("get_dof_indices", &physics::Articulation::getDofIndices, nb::arg("names"))
        .def("get_joint_indices", &physics::Articulation::getJointIndices, nb::arg("names"))
        .def("get_link_indices", &physics::Articulation::getLinkIndices, nb::arg("names"))
        .def("get_dof_limits", &physics::Articulation::getDofLimits, nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("dof_indices") = nb::none())
        .def("set_dof_limits", &physics::Articulation::setDofLimits, nb::arg("lower") = nb::none(),
             nb::arg("upper") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("dof_indices") = nb::none())
        .def("get_dof_friction_properties", &physics::Articulation::getDofFrictionProperties, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("set_dof_friction_properties", &physics::Articulation::setDofFrictionProperties,
             nb::arg("static_frictions") = nb::none(), nb::arg("dynamic_frictions") = nb::none(),
             nb::arg("viscous_frictions") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("dof_indices") = nb::none())
        .def("get_dof_drive_model_properties", &physics::Articulation::getDofDriveModelProperties, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("set_dof_drive_model_properties", &physics::Articulation::setDofDriveModelProperties,
             nb::arg("speed_effort_gradients") = nb::none(), nb::arg("maximum_actuator_velocities") = nb::none(),
             nb::arg("velocity_dependent_resistances") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("dof_indices") = nb::none())
        .def("get_dof_armatures", &physics::Articulation::getDofArmatures, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("set_dof_armatures", &physics::Articulation::setDofArmatures, nb::arg("armatures"), nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_drive_types", &physics::Articulation::getDofDriveTypes, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("set_dof_drive_types", &physics::Articulation::setDofDriveTypes, nb::arg("types"), nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_max_velocities", &physics::Articulation::getDofMaxVelocities, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("set_dof_max_velocities", &physics::Articulation::setDofMaxVelocities, nb::arg("max_velocities"),
             nb::kw_only(), nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_max_efforts", &physics::Articulation::getDofMaxEfforts, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("set_dof_max_efforts", &physics::Articulation::setDofMaxEfforts, nb::arg("max_efforts"), nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_gains", &physics::Articulation::getDofGains, nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("dof_indices") = nb::none())
        .def("set_dof_gains", &physics::Articulation::setDofGains, nb::arg("stiffnesses") = nb::none(),
             nb::arg("dampings") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("dof_indices") = nb::none(), nb::arg("update_default_gains") = true)
        .def("switch_dof_control_mode", &physics::Articulation::switchDofControlMode, nb::arg("mode"), nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_position_targets", &physics::Articulation::getDofPositionTargets, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("set_dof_position_targets", &physics::Articulation::setDofPositionTargets, nb::arg("positions"),
             nb::kw_only(), nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_dof_velocity_targets", &physics::Articulation::getDofVelocityTargets, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("set_dof_velocity_targets", &physics::Articulation::setDofVelocityTargets, nb::arg("velocities"),
             nb::kw_only(), nb::arg("indices") = nb::none(), nb::arg("dof_indices") = nb::none())
        .def("get_link_masses", &physics::Articulation::getLinkMasses, nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("link_indices") = nb::none(), nb::arg("inverse") = false)
        .def("set_link_masses", &physics::Articulation::setLinkMasses, nb::arg("masses"), nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("link_indices") = nb::none())
        .def("get_link_enabled_gravities", &physics::Articulation::getLinkEnabledGravities, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("link_indices") = nb::none())
        .def("set_link_enabled_gravities", &physics::Articulation::setLinkEnabledGravities, nb::arg("enabled"),
             nb::kw_only(), nb::arg("indices") = nb::none(), nb::arg("link_indices") = nb::none())
        .def("get_solver_iteration_counts", &physics::Articulation::getSolverIterationCounts, nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_solver_iteration_counts", &physics::Articulation::setSolverIterationCounts,
             nb::arg("position_counts") = nb::none(), nb::arg("velocity_counts") = nb::none(), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_stabilization_thresholds", &physics::Articulation::getStabilizationThresholds, nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_stabilization_thresholds", &physics::Articulation::setStabilizationThresholds, nb::arg("thresholds"),
             nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_enabled_self_collisions", &physics::Articulation::getEnabledSelfCollisions, nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_enabled_self_collisions", &physics::Articulation::setEnabledSelfCollisions, nb::arg("enabled"),
             nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_sleep_thresholds", &physics::Articulation::getSleepThresholds, nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_sleep_thresholds", &physics::Articulation::setSleepThresholds, nb::arg("thresholds"), nb::kw_only(),
             nb::arg("indices") = nb::none());

    // physics/ColliderBody.hpp
    nb::class_<physics::ColliderBody, Xform>(m, "ColliderBody")
        .def(
            "__init__",
            [](physics::ColliderBody* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<std::variant<std::string, std::vector<std::string>>>& approximations,
               bool applyCollisionApis, const std::optional<array::Array>& positions,
               const std::optional<array::Array>& translations, const std::optional<array::Array>& orientations,
               const std::optional<array::Array>& scales, bool resetXformOpProperties)
            {
                new (self) physics::ColliderBody(paths, approximations, applyCollisionApis, positions, translations,
                                                 orientations, scales, resetXformOpProperties);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("approximations") = nb::none(),
            nb::arg("apply_collision_apis") = true, nb::arg("positions") = nb::none(),
            nb::arg("translations") = nb::none(), nb::arg("orientations") = nb::none(), nb::arg("scales") = nb::none(),
            nb::arg("reset_xform_op_properties") = true)
        .def("apply_collision_apis", &physics::ColliderBody::applyCollisionApis, nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("remove_collision_apis", &physics::ColliderBody::removeCollisionApis, nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_offsets", &physics::ColliderBody::setOffsets, nb::arg("contact_offsets") = nb::none(),
             nb::arg("rest_offsets") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_offsets", &physics::ColliderBody::getOffsets, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_torsional_patch_radii", &physics::ColliderBody::setTorsionalPatchRadii, nb::arg("radii"),
             nb::kw_only(), nb::arg("indices") = nb::none(), nb::arg("minimum") = false)
        .def("get_torsional_patch_radii", &physics::ColliderBody::getTorsionalPatchRadii, nb::kw_only(),
             nb::arg("indices") = nb::none(), nb::arg("minimum") = false)
        .def("set_collision_approximations", &physics::ColliderBody::setCollisionApproximations,
             nb::arg("approximations"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_collision_approximations", &physics::ColliderBody::getCollisionApproximations, nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_enabled_collisions", &physics::ColliderBody::setEnabledCollisions, nb::arg("enabled"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_enabled_collisions", &physics::ColliderBody::getEnabledCollisions, nb::kw_only(),
             nb::arg("indices") = nb::none());

    // physics/RigidBody.hpp
    nb::class_<physics::RigidBody, Xform>(m, "RigidBody")
        .def(
            "__init__",
            [](physics::RigidBody* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<array::Array>& masses, const std::optional<array::Array>& densities,
               bool applyPhysicsApis, const std::optional<array::Array>& positions,
               const std::optional<array::Array>& translations, const std::optional<array::Array>& orientations,
               const std::optional<array::Array>& scales, bool resetXformOpProperties)
            {
                new (self) physics::RigidBody(paths, masses, densities, applyPhysicsApis, positions, translations,
                                              orientations, scales, resetXformOpProperties);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("masses") = nb::none(), nb::arg("densities") = nb::none(),
            nb::arg("apply_physics_apis") = true, nb::arg("positions") = nb::none(),
            nb::arg("translations") = nb::none(), nb::arg("orientations") = nb::none(), nb::arg("scales") = nb::none(),
            nb::arg("reset_xform_op_properties") = true)
        .def("apply_physics_apis", &physics::RigidBody::applyPhysicsApis, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("remove_physics_apis", &physics::RigidBody::removePhysicsApis, nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_velocities", &physics::RigidBody::setVelocities, nb::arg("linear_velocities") = nb::none(),
             nb::arg("angular_velocities") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_velocities", &physics::RigidBody::getVelocities, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_masses", &physics::RigidBody::getMasses, nb::kw_only(), nb::arg("indices") = nb::none(),
             nb::arg("inverse") = false)
        .def("set_masses", &physics::RigidBody::setMasses, nb::arg("masses"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_densities", &physics::RigidBody::getDensities, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_densities", &physics::RigidBody::setDensities, nb::arg("densities"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_sleep_thresholds", &physics::RigidBody::getSleepThresholds, nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_sleep_thresholds", &physics::RigidBody::setSleepThresholds, nb::arg("thresholds"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_enabled_rigid_bodies", &physics::RigidBody::setEnabledRigidBodies, nb::arg("enabled"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_enabled_rigid_bodies", &physics::RigidBody::getEnabledRigidBodies, nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_enabled_gravities", &physics::RigidBody::setEnabledGravities, nb::arg("enabled"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_enabled_gravities", &physics::RigidBody::getEnabledGravities, nb::kw_only(),
             nb::arg("indices") = nb::none());
}
