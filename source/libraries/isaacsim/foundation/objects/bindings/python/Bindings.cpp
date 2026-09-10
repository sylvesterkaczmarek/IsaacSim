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
#include "isaacsim/foundation/objects/Camera.hpp"
#include "isaacsim/foundation/objects/Mesh.hpp"
#include "isaacsim/foundation/objects/Prim.hpp"
#include "isaacsim/foundation/objects/Stage.hpp"
#include "isaacsim/foundation/objects/Xform.hpp"
#include "isaacsim/foundation/objects/lights/CylinderLight.hpp"
#include "isaacsim/foundation/objects/lights/DiskLight.hpp"
#include "isaacsim/foundation/objects/lights/DistantLight.hpp"
#include "isaacsim/foundation/objects/lights/DomeLight.hpp"
#include "isaacsim/foundation/objects/lights/Light.hpp"
#include "isaacsim/foundation/objects/lights/RectLight.hpp"
#include "isaacsim/foundation/objects/lights/SphereLight.hpp"
#include "isaacsim/foundation/objects/shapes/Capsule.hpp"
#include "isaacsim/foundation/objects/shapes/Cone.hpp"
#include "isaacsim/foundation/objects/shapes/Cube.hpp"
#include "isaacsim/foundation/objects/shapes/Cylinder.hpp"
#include "isaacsim/foundation/objects/shapes/Plane.hpp"
#include "isaacsim/foundation/objects/shapes/Shape.hpp"
#include "isaacsim/foundation/objects/shapes/Sphere.hpp"

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

NB_MODULE(_bindings, m)
{
    // Prim.h
    nb::class_<Prim>(m, "Prim")
        .def(
            "__init__",
            [](Prim* self, const std::variant<std::string, std::vector<std::string>>& paths, bool resolvePaths)
            { new (self) Prim(paths, resolvePaths); },
            nb::arg("paths"), nb::kw_only(), nb::arg("resolve_paths") = true)
        .def_prop_ro("paths", &Prim::paths)
        .def("get_stage", &Prim::getStage, nb::rv_policy::reference_internal)
        .def_static(
            "resolve_paths",
            [](const std::variant<std::string, std::vector<std::string>>& paths, bool raiseOnMixedPaths)
            { return Prim({}, false).resolvePaths(paths, raiseOnMixedPaths); },
            nb::arg("paths"), nb::kw_only(), nb::arg("raise_on_mixed_paths") = true)
        .def("get_name", &Prim::getName, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_type_name", &Prim::getTypeName, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_parent", &Prim::getParent, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_children", &Prim::getChildren, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_variant_sets", &Prim::getVariantSets, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_variant_selection", &Prim::getVariantSelection, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_variant_selection", &Prim::setVariantSelection, nb::arg("variants"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("is_a", &Prim::isA, nb::arg("schema_type"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("has_api", &Prim::hasApi, nb::arg("schema_type"), nb::kw_only(), nb::arg("instance_name") = nb::none(),
             nb::arg("indices") = nb::none())
        .def("apply_api", &Prim::applyApi, nb::arg("schema_type"), nb::kw_only(), nb::arg("instance_name") = nb::none(),
             nb::arg("indices") = nb::none())
        .def("remove_api", &Prim::removeApi, nb::arg("schema_type"), nb::kw_only(),
             nb::arg("instance_name") = nb::none(), nb::arg("indices") = nb::none())
        .def("get_applied_schemas", &Prim::getAppliedSchemas, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_attribute_values", &Prim::getAttributeValues, nb::arg("attribute_name"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_attribute_values", &Prim::setAttributeValues, nb::arg("attribute_name"), nb::arg("values"),
             nb::kw_only(), nb::arg("indices") = nb::none());

    // Stage.h
    nb::class_<Stage>(m, "Stage")
        .def(nb::init<std::string, std::optional<int64_t>>(), nb::arg("backend"), nb::arg("stage_id") = nb::none())
        .def("get_backend", &Stage::getBackend)
        .def("get_stage_id", &Stage::getStageId)
        .def("get_stage_ptr", [](const Stage& self) { return reinterpret_cast<uintptr_t>(self.getStagePtr()); })
        .def("is_valid", &Stage::isValid)
        .def(
            "open_stage",
            [](nb::object self, const std::string& usdPath, bool makeDefault) -> nb::object
            {
                nb::cast<Stage&>(self).openStage(usdPath, makeDefault);
                return self;
            },
            nb::arg("usd_path"), nb::kw_only(), nb::arg("make_default") = true)
        .def(
            "create_stage",
            [](nb::object self, std::optional<std::string> templateName, bool makeDefault) -> nb::object
            {
                nb::cast<Stage&>(self).createStage(templateName, makeDefault);
                return self;
            },
            nb::kw_only(), nb::arg("template") = nb::none(), nb::arg("make_default") = true)
        .def("save_stage", &Stage::saveStage, nb::arg("usd_path"))
        .def("export_stage_to_string", &Stage::exportStageToString)
        .def(
            "import_stage_from_string",
            [](nb::object self, const std::string& usdString, bool makeDefault) -> nb::object
            {
                nb::cast<Stage&>(self).importStageFromString(usdString, makeDefault);
                return self;
            },
            nb::arg("usd_string"), nb::kw_only(), nb::arg("make_default") = true)
        .def("close_stage", &Stage::closeStage)
        .def("add_reference", &Stage::addReference, nb::arg("usd_path"), nb::arg("path"), nb::kw_only(),
             nb::arg("prim_type") = "Xform", nb::arg("variants") = nb::none())
        .def("define_prim", &Stage::definePrim, nb::arg("path"), nb::arg("type_name") = "Xform")
        .def("move_prim", &Stage::movePrim, nb::arg("target"), nb::arg("destination"))
        .def("remove_prim", &Stage::removePrim, nb::arg("path"))
        .def("get_up_axis", &Stage::getUpAxis)
        .def("set_up_axis", &Stage::setUpAxis, nb::arg("up_axis"))
        .def("get_units", &Stage::getUnits)
        .def("set_units", &Stage::setUnits, nb::kw_only(), nb::arg("meters_per_unit") = nb::none(),
             nb::arg("kilograms_per_unit") = nb::none())
        .def("get_time_code", &Stage::getTimeCode)
        .def("set_time_code", &Stage::setTimeCode, nb::kw_only(), nb::arg("start_time_code") = nb::none(),
             nb::arg("end_time_code") = nb::none(), nb::arg("time_codes_per_second") = nb::none())
        .def("generate_string_representation", &Stage::generateStringRepresentation, nb::kw_only(),
             nb::arg("mode") = "tree")
        .def("__str__", [](const Stage& self) { return self.generateStringRepresentation("tree"); });

    // Xform.h
    nb::class_<Xform, Prim>(m, "Xform")
        .def(
            "__init__",
            [](Xform* self, const std::variant<std::string, std::vector<std::string>>& paths, bool resolvePaths,
               const std::optional<array::Array>& positions, const std::optional<array::Array>& translations,
               const std::optional<array::Array>& orientations, const std::optional<array::Array>& scales,
               bool resetXformOpProperties) {
                new (self)
                    Xform(paths, positions, translations, orientations, scales, resetXformOpProperties, resolvePaths);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("resolve_paths") = true, nb::arg("positions") = nb::none(),
            nb::arg("translations") = nb::none(), nb::arg("orientations") = nb::none(), nb::arg("scales") = nb::none(),
            nb::arg("reset_xform_op_properties") = true)
        .def("set_visibilities", &Xform::setVisibilities, nb::arg("visibilities"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_visibilities", &Xform::getVisibilities, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_world_poses", &Xform::getWorldPoses, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_local_poses", &Xform::getLocalPoses, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_local_scales", &Xform::getLocalScales, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_local_scales", &Xform::setLocalScales, nb::arg("scales"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_local_poses", &Xform::setLocalPoses, nb::arg("translations") = nb::none(),
             nb::arg("orientations") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_world_poses", &Xform::setWorldPoses, nb::arg("positions") = nb::none(),
             nb::arg("orientations") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("reset_xform_op_properties", &Xform::resetXformOpProperties);

    // shapes/Shape.h
    nb::class_<shapes::Shape, Xform>(m, "Shape")
        .def("set_display_colors", &shapes::Shape::setDisplayColors, nb::arg("colors"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_display_colors", &shapes::Shape::getDisplayColors, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("update_extents", &shapes::Shape::updateExtents);

    // shapes/Cube.h
    nb::class_<shapes::Cube, shapes::Shape>(m, "Cube")
        .def(
            "__init__",
            [](shapes::Cube* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<array::Array>& sizes, const std::optional<shapes::ColorType>& colors,
               const std::optional<array::Array>& positions, const std::optional<array::Array>& translations,
               const std::optional<array::Array>& orientations, const std::optional<array::Array>& scales,
               bool resetXformOpProperties)
            {
                new (self) shapes::Cube(
                    paths, sizes, colors, positions, translations, orientations, scales, resetXformOpProperties);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("sizes") = nb::none(), nb::arg("colors") = nb::none(),
            nb::arg("positions") = nb::none(), nb::arg("translations") = nb::none(), nb::arg("orientations") = nb::none(),
            nb::arg("scales") = nb::none(), nb::arg("reset_xform_op_properties") = true)
        .def("set_sizes", &shapes::Cube::setSizes, nb::arg("sizes"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_sizes", &shapes::Cube::getSizes, nb::kw_only(), nb::arg("indices") = nb::none());

    // shapes/Sphere.h
    nb::class_<shapes::Sphere, shapes::Shape>(m, "Sphere")
        .def(
            "__init__",
            [](shapes::Sphere* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<array::Array>& radii, const std::optional<shapes::ColorType>& colors,
               const std::optional<array::Array>& positions, const std::optional<array::Array>& translations,
               const std::optional<array::Array>& orientations, const std::optional<array::Array>& scales,
               bool resetXformOpProperties)
            {
                new (self) shapes::Sphere(
                    paths, radii, colors, positions, translations, orientations, scales, resetXformOpProperties);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("radii") = nb::none(), nb::arg("colors") = nb::none(),
            nb::arg("positions") = nb::none(), nb::arg("translations") = nb::none(), nb::arg("orientations") = nb::none(),
            nb::arg("scales") = nb::none(), nb::arg("reset_xform_op_properties") = true)
        .def("set_radii", &shapes::Sphere::setRadii, nb::arg("radii"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_radii", &shapes::Sphere::getRadii, nb::kw_only(), nb::arg("indices") = nb::none());

    // shapes/Cylinder.h
    nb::class_<shapes::Cylinder, shapes::Shape>(m, "Cylinder")
        .def(
            "__init__",
            [](shapes::Cylinder* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<array::Array>& radii, const std::optional<array::Array>& heights,
               const std::optional<std::variant<std::string, std::vector<std::string>>>& axes,
               const std::optional<shapes::ColorType>& colors, const std::optional<array::Array>& positions,
               const std::optional<array::Array>& translations, const std::optional<array::Array>& orientations,
               const std::optional<array::Array>& scales, bool resetXformOpProperties)
            {
                new (self) shapes::Cylinder(paths, radii, heights, axes, colors, positions, translations, orientations,
                                            scales, resetXformOpProperties);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("radii") = nb::none(), nb::arg("heights") = nb::none(),
            nb::arg("axes") = nb::none(), nb::arg("colors") = nb::none(), nb::arg("positions") = nb::none(),
            nb::arg("translations") = nb::none(), nb::arg("orientations") = nb::none(), nb::arg("scales") = nb::none(),
            nb::arg("reset_xform_op_properties") = true)
        .def("set_radii", &shapes::Cylinder::setRadii, nb::arg("radii"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_radii", &shapes::Cylinder::getRadii, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_heights", &shapes::Cylinder::setHeights, nb::arg("heights"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_heights", &shapes::Cylinder::getHeights, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_axes", &shapes::Cylinder::setAxes, nb::arg("axes"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_axes", &shapes::Cylinder::getAxes, nb::kw_only(), nb::arg("indices") = nb::none());

    // shapes/Capsule.h
    nb::class_<shapes::Capsule, shapes::Shape>(m, "Capsule")
        .def(
            "__init__",
            [](shapes::Capsule* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<array::Array>& radii, const std::optional<array::Array>& heights,
               const std::optional<std::variant<std::string, std::vector<std::string>>>& axes,
               const std::optional<shapes::ColorType>& colors, const std::optional<array::Array>& positions,
               const std::optional<array::Array>& translations, const std::optional<array::Array>& orientations,
               const std::optional<array::Array>& scales, bool resetXformOpProperties)
            {
                new (self) shapes::Capsule(paths, radii, heights, axes, colors, positions, translations, orientations,
                                           scales, resetXformOpProperties);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("radii") = nb::none(), nb::arg("heights") = nb::none(),
            nb::arg("axes") = nb::none(), nb::arg("colors") = nb::none(), nb::arg("positions") = nb::none(),
            nb::arg("translations") = nb::none(), nb::arg("orientations") = nb::none(), nb::arg("scales") = nb::none(),
            nb::arg("reset_xform_op_properties") = true)
        .def("set_radii", &shapes::Capsule::setRadii, nb::arg("radii"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_radii", &shapes::Capsule::getRadii, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_heights", &shapes::Capsule::setHeights, nb::arg("heights"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_heights", &shapes::Capsule::getHeights, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_axes", &shapes::Capsule::setAxes, nb::arg("axes"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_axes", &shapes::Capsule::getAxes, nb::kw_only(), nb::arg("indices") = nb::none());

    // shapes/Cone.h
    nb::class_<shapes::Cone, shapes::Shape>(m, "Cone")
        .def(
            "__init__",
            [](shapes::Cone* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<array::Array>& radii, const std::optional<array::Array>& heights,
               const std::optional<std::variant<std::string, std::vector<std::string>>>& axes,
               const std::optional<shapes::ColorType>& colors, const std::optional<array::Array>& positions,
               const std::optional<array::Array>& translations, const std::optional<array::Array>& orientations,
               const std::optional<array::Array>& scales, bool resetXformOpProperties)
            {
                new (self) shapes::Cone(paths, radii, heights, axes, colors, positions, translations, orientations,
                                        scales, resetXformOpProperties);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("radii") = nb::none(), nb::arg("heights") = nb::none(),
            nb::arg("axes") = nb::none(), nb::arg("colors") = nb::none(), nb::arg("positions") = nb::none(),
            nb::arg("translations") = nb::none(), nb::arg("orientations") = nb::none(), nb::arg("scales") = nb::none(),
            nb::arg("reset_xform_op_properties") = true)
        .def("set_radii", &shapes::Cone::setRadii, nb::arg("radii"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_radii", &shapes::Cone::getRadii, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_heights", &shapes::Cone::setHeights, nb::arg("heights"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_heights", &shapes::Cone::getHeights, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_axes", &shapes::Cone::setAxes, nb::arg("axes"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_axes", &shapes::Cone::getAxes, nb::kw_only(), nb::arg("indices") = nb::none());

    // shapes/Plane.h
    nb::class_<shapes::Plane, shapes::Shape>(m, "Plane")
        .def(
            "__init__",
            [](shapes::Plane* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<array::Array>& widths, const std::optional<array::Array>& lengths,
               const std::optional<std::variant<std::string, std::vector<std::string>>>& axes,
               const std::optional<shapes::ColorType>& colors, const std::optional<array::Array>& positions,
               const std::optional<array::Array>& translations, const std::optional<array::Array>& orientations,
               const std::optional<array::Array>& scales, bool resetXformOpProperties)
            {
                new (self) shapes::Plane(paths, widths, lengths, axes, colors, positions, translations, orientations,
                                         scales, resetXformOpProperties);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("widths") = nb::none(), nb::arg("lengths") = nb::none(),
            nb::arg("axes") = nb::none(), nb::arg("colors") = nb::none(), nb::arg("positions") = nb::none(),
            nb::arg("translations") = nb::none(), nb::arg("orientations") = nb::none(), nb::arg("scales") = nb::none(),
            nb::arg("reset_xform_op_properties") = true)
        .def("set_widths", &shapes::Plane::setWidths, nb::arg("widths"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_widths", &shapes::Plane::getWidths, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_lengths", &shapes::Plane::setLengths, nb::arg("lengths"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_lengths", &shapes::Plane::getLengths, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_axes", &shapes::Plane::setAxes, nb::arg("axes"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_axes", &shapes::Plane::getAxes, nb::kw_only(), nb::arg("indices") = nb::none());

    // lights/Light.h
    nb::class_<lights::Light, Xform>(m, "Light")
        .def("set_intensities", &lights::Light::setIntensities, nb::arg("intensities"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_intensities", &lights::Light::getIntensities, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_exposures", &lights::Light::setExposures, nb::arg("exposures"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_exposures", &lights::Light::getExposures, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_multipliers", &lights::Light::setMultipliers, nb::arg("diffuse_multipliers") = nb::none(),
             nb::arg("specular_multipliers") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_multipliers", &lights::Light::getMultipliers, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_enabled_normalizations", &lights::Light::setEnabledNormalizations, nb::arg("enabled"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_enabled_normalizations", &lights::Light::getEnabledNormalizations, nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_enabled_color_temperatures", &lights::Light::setEnabledColorTemperatures, nb::arg("enabled"),
             nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_enabled_color_temperatures", &lights::Light::getEnabledColorTemperatures, nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_color_temperatures", &lights::Light::setColorTemperatures, nb::arg("color_temperatures"),
             nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_color_temperatures", &lights::Light::getColorTemperatures, nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("set_colors", &lights::Light::setColors, nb::arg("colors"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_colors", &lights::Light::getColors, nb::kw_only(), nb::arg("indices") = nb::none());

    // lights/Sphere.h
    nb::class_<lights::SphereLight, lights::Light>(m, "SphereLight")
        .def(
            "__init__",
            [](lights::SphereLight* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<array::Array>& radii, const std::optional<array::Array>& positions,
               const std::optional<array::Array>& translations, const std::optional<array::Array>& orientations,
               const std::optional<array::Array>& scales, bool resetXformOpProperties)
            {
                new (self) lights::SphereLight(
                    paths, radii, positions, translations, orientations, scales, resetXformOpProperties);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("radii") = nb::none(), nb::arg("positions") = nb::none(),
            nb::arg("translations") = nb::none(), nb::arg("orientations") = nb::none(), nb::arg("scales") = nb::none(),
            nb::arg("reset_xform_op_properties") = true)
        .def("set_radii", &lights::SphereLight::setRadii, nb::arg("radii"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_radii", &lights::SphereLight::getRadii, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_enabled_treat_as_points", &lights::SphereLight::setEnabledTreatAsPoints, nb::arg("enabled"),
             nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_enabled_treat_as_points", &lights::SphereLight::getEnabledTreatAsPoints, nb::kw_only(),
             nb::arg("indices") = nb::none());

    // lights/Distant.h
    nb::class_<lights::DistantLight, lights::Light>(m, "DistantLight")
        .def(
            "__init__",
            [](lights::DistantLight* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<array::Array>& angles, const std::optional<array::Array>& positions,
               const std::optional<array::Array>& translations, const std::optional<array::Array>& orientations,
               const std::optional<array::Array>& scales, bool resetXformOpProperties)
            {
                new (self) lights::DistantLight(
                    paths, angles, positions, translations, orientations, scales, resetXformOpProperties);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("angles") = nb::none(), nb::arg("positions") = nb::none(),
            nb::arg("translations") = nb::none(), nb::arg("orientations") = nb::none(), nb::arg("scales") = nb::none(),
            nb::arg("reset_xform_op_properties") = true)
        .def("set_angles", &lights::DistantLight::setAngles, nb::arg("angles"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_angles", &lights::DistantLight::getAngles, nb::kw_only(), nb::arg("indices") = nb::none());

    // lights/Cylinder.h
    nb::class_<lights::CylinderLight, lights::Light>(m, "CylinderLight")
        .def(
            "__init__",
            [](lights::CylinderLight* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<array::Array>& radii, const std::optional<array::Array>& lengths,
               const std::optional<array::Array>& positions, const std::optional<array::Array>& translations,
               const std::optional<array::Array>& orientations, const std::optional<array::Array>& scales,
               bool resetXformOpProperties)
            {
                new (self) lights::CylinderLight(
                    paths, radii, lengths, positions, translations, orientations, scales, resetXformOpProperties);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("radii") = nb::none(), nb::arg("lengths") = nb::none(),
            nb::arg("positions") = nb::none(), nb::arg("translations") = nb::none(), nb::arg("orientations") = nb::none(),
            nb::arg("scales") = nb::none(), nb::arg("reset_xform_op_properties") = true)
        .def("set_radii", &lights::CylinderLight::setRadii, nb::arg("radii"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_radii", &lights::CylinderLight::getRadii, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_lengths", &lights::CylinderLight::setLengths, nb::arg("lengths"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_lengths", &lights::CylinderLight::getLengths, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_enabled_treat_as_lines", &lights::CylinderLight::setEnabledTreatAsLines, nb::arg("enabled"),
             nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_enabled_treat_as_lines", &lights::CylinderLight::getEnabledTreatAsLines, nb::kw_only(),
             nb::arg("indices") = nb::none());

    // lights/Rect.h
    nb::class_<lights::RectLight, lights::Light>(m, "RectLight")
        .def(
            "__init__",
            [](lights::RectLight* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<array::Array>& widths, const std::optional<array::Array>& heights,
               const std::optional<std::variant<std::string, std::vector<std::string>>>& textureFiles,
               const std::optional<array::Array>& positions, const std::optional<array::Array>& translations,
               const std::optional<array::Array>& orientations, const std::optional<array::Array>& scales,
               bool resetXformOpProperties)
            {
                new (self) lights::RectLight(paths, widths, heights, textureFiles, positions, translations,
                                             orientations, scales, resetXformOpProperties);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("widths") = nb::none(), nb::arg("heights") = nb::none(),
            nb::arg("texture_files") = nb::none(), nb::arg("positions") = nb::none(),
            nb::arg("translations") = nb::none(), nb::arg("orientations") = nb::none(), nb::arg("scales") = nb::none(),
            nb::arg("reset_xform_op_properties") = true)
        .def("set_widths", &lights::RectLight::setWidths, nb::arg("widths"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_widths", &lights::RectLight::getWidths, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_heights", &lights::RectLight::setHeights, nb::arg("heights"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_heights", &lights::RectLight::getHeights, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_texture_files", &lights::RectLight::setTextureFiles, nb::arg("texture_files"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_texture_files", &lights::RectLight::getTextureFiles, nb::kw_only(), nb::arg("indices") = nb::none());

    // lights/Dome.h
    nb::class_<lights::DomeLight, lights::Light>(m, "DomeLight")
        .def(
            "__init__",
            [](lights::DomeLight* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<array::Array>& radii,
               const std::optional<std::variant<std::string, std::vector<std::string>>>& textureFiles,
               const std::optional<std::variant<std::string, std::vector<std::string>>>& textureFormats,
               const std::optional<array::Array>& positions, const std::optional<array::Array>& translations,
               const std::optional<array::Array>& orientations, const std::optional<array::Array>& scales,
               bool resetXformOpProperties)
            {
                new (self) lights::DomeLight(paths, radii, textureFiles, textureFormats, positions, translations,
                                             orientations, scales, resetXformOpProperties);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("radii") = nb::none(), nb::arg("texture_files") = nb::none(),
            nb::arg("texture_formats") = nb::none(), nb::arg("positions") = nb::none(),
            nb::arg("translations") = nb::none(), nb::arg("orientations") = nb::none(), nb::arg("scales") = nb::none(),
            nb::arg("reset_xform_op_properties") = true)
        .def("set_guide_radii", &lights::DomeLight::setGuideRadii, nb::arg("radii"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_guide_radii", &lights::DomeLight::getGuideRadii, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_texture_files", &lights::DomeLight::setTextureFiles, nb::arg("texture_files"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_texture_files", &lights::DomeLight::getTextureFiles, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_texture_formats", &lights::DomeLight::setTextureFormats, nb::arg("texture_formats"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_texture_formats", &lights::DomeLight::getTextureFormats, nb::kw_only(),
             nb::arg("indices") = nb::none());

    // lights/Disk.h
    nb::class_<lights::DiskLight, lights::Light>(m, "DiskLight")
        .def(
            "__init__",
            [](lights::DiskLight* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<array::Array>& radii, const std::optional<array::Array>& positions,
               const std::optional<array::Array>& translations, const std::optional<array::Array>& orientations,
               const std::optional<array::Array>& scales, bool resetXformOpProperties) {
                new (self) lights::DiskLight(
                    paths, radii, positions, translations, orientations, scales, resetXformOpProperties);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("radii") = nb::none(), nb::arg("positions") = nb::none(),
            nb::arg("translations") = nb::none(), nb::arg("orientations") = nb::none(), nb::arg("scales") = nb::none(),
            nb::arg("reset_xform_op_properties") = true)
        .def("set_radii", &lights::DiskLight::setRadii, nb::arg("radii"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_radii", &lights::DiskLight::getRadii, nb::kw_only(), nb::arg("indices") = nb::none());

    // Mesh.h
    nb::class_<Mesh, Xform>(m, "Mesh")
        .def(
            "__init__",
            [](Mesh* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<std::variant<std::string, std::vector<std::string>>>& primitives,
               const std::optional<ColorType>& colors, const std::optional<array::Array>& positions,
               const std::optional<array::Array>& translations, const std::optional<array::Array>& orientations,
               const std::optional<array::Array>& scales, bool resetXformOpProperties) {
                new (self) Mesh(
                    paths, primitives, colors, positions, translations, orientations, scales, resetXformOpProperties);
            },
            nb::arg("paths"), nb::kw_only(), nb::arg("primitives") = nb::none(), nb::arg("colors") = nb::none(),
            nb::arg("positions") = nb::none(), nb::arg("translations") = nb::none(), nb::arg("orientations") = nb::none(),
            nb::arg("scales") = nb::none(), nb::arg("reset_xform_op_properties") = true)
        .def_prop_ro("num_faces", &Mesh::numFaces)
        .def("set_points", &Mesh::setPoints, nb::arg("points"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_points", &Mesh::getPoints, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_normals", &Mesh::setNormals, nb::arg("normals"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_normals", &Mesh::getNormals, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_face_specs", &Mesh::setFaceSpecs, nb::arg("vertex_indices") = nb::none(),
             nb::arg("vertex_counts") = nb::none(), nb::arg("varying_linear_interpolations") = nb::none(),
             nb::arg("hole_indices") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_face_specs", &Mesh::getFaceSpecs, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_crease_specs", &Mesh::setCreaseSpecs, nb::arg("crease_indices"), nb::arg("crease_lengths"),
             nb::arg("crease_sharpnesses"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_crease_specs", &Mesh::getCreaseSpecs, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_corner_specs", &Mesh::setCornerSpecs, nb::arg("corner_indices"), nb::arg("corner_sharpnesses"),
             nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_corner_specs", &Mesh::getCornerSpecs, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_subdivision_specs", &Mesh::setSubdivisionSpecs, nb::arg("subdivision_schemes") = nb::none(),
             nb::arg("interpolate_boundaries") = nb::none(), nb::arg("triangle_subdivision_rules") = nb::none(),
             nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_subdivision_specs", &Mesh::getSubdivisionSpecs, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_display_colors", &Mesh::setDisplayColors, nb::arg("colors"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_display_colors", &Mesh::getDisplayColors, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("update_extents", &Mesh::updateExtents);

    // Camera.h
    nb::class_<Camera, Xform>(m, "Camera")
        .def(
            "__init__",
            [](Camera* self, const std::variant<std::string, std::vector<std::string>>& paths,
               const std::optional<array::Array>& positions, const std::optional<array::Array>& translations,
               const std::optional<array::Array>& orientations, const std::optional<array::Array>& scales,
               bool resetXformOpProperties)
            { new (self) Camera(paths, positions, translations, orientations, scales, resetXformOpProperties); },
            nb::arg("paths"), nb::kw_only(), nb::arg("positions") = nb::none(), nb::arg("translations") = nb::none(),
            nb::arg("orientations") = nb::none(), nb::arg("scales") = nb::none(),
            nb::arg("reset_xform_op_properties") = true)
        .def("set_focal_lengths", &Camera::setFocalLengths, nb::arg("focal_lengths"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_focal_lengths", &Camera::getFocalLengths, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_focus_distances", &Camera::setFocusDistances, nb::arg("focus_distances"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_focus_distances", &Camera::getFocusDistances, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_stereo_roles", &Camera::setStereoRoles, nb::arg("roles"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_stereo_roles", &Camera::getStereoRoles, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_fstops", &Camera::setFStops, nb::arg("fstops"), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_fstops", &Camera::getFStops, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_apertures", &Camera::setApertures, nb::arg("horizontal_apertures") = nb::none(),
             nb::arg("vertical_apertures") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_apertures", &Camera::getApertures, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_aperture_offsets", &Camera::setApertureOffsets, nb::arg("horizontal_offsets") = nb::none(),
             nb::arg("vertical_offsets") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_aperture_offsets", &Camera::getApertureOffsets, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_projections", &Camera::setProjections, nb::arg("projections"), nb::kw_only(),
             nb::arg("indices") = nb::none())
        .def("get_projections", &Camera::getProjections, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_clipping_ranges", &Camera::setClippingRanges, nb::arg("near_distances") = nb::none(),
             nb::arg("far_distances") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_clipping_ranges", &Camera::getClippingRanges, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("set_shutter_times", &Camera::setShutterTimes, nb::arg("open_times") = nb::none(),
             nb::arg("close_times") = nb::none(), nb::kw_only(), nb::arg("indices") = nb::none())
        .def("get_shutter_times", &Camera::getShutterTimes, nb::kw_only(), nb::arg("indices") = nb::none())
        .def("enforce_square_pixels", &Camera::enforceSquarePixels, nb::arg("resolutions"), nb::kw_only(),
             nb::arg("modes") = "horizontal", nb::arg("indices") = nb::none());
}
