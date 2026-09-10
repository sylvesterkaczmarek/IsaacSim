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

#include "isaacsim/foundation/utils/Backend.hpp"
#include "isaacsim/foundation/utils/Prim.hpp"
#include "isaacsim/foundation/utils/Semantics.hpp"
#include "isaacsim/foundation/utils/Stage.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <memory>

namespace nb = nanobind;
namespace objects = isaacsim::foundation::objects;
using namespace isaacsim::foundation::utils;

namespace
{

// Heap-allocated wrapper so the RAII guard lifetime is controlled explicitly via close().
struct BackendGuardWrapper
{
    std::unique_ptr<BackendGuard> guard;

    BackendGuardWrapper(const std::string& backend, bool raiseOnUnsupported, bool raiseOnFallback)
        : guard(std::make_unique<BackendGuard>(backend, raiseOnUnsupported, raiseOnFallback))
    {
    }

    void close()
    {
        guard.reset(); // Reset on null unique_ptr is a no-op.
    }
};

// Heap-allocated wrapper so the RAII guard lifetime is controlled explicitly via close().
struct StageGuardWrapper
{
    std::unique_ptr<StageGuard> guard;

    StageGuardWrapper(objects::Stage stage) : guard(std::make_unique<StageGuard>(std::move(stage)))
    {
    }

    void close()
    {
        guard.reset(); // Reset on null unique_ptr is a no-op.
    }
};

} // namespace

NB_MODULE(_bindings, m)
{
    // Backend.h
    nb::class_<BackendGuardWrapper>(m, "_BackendGuard")
        .def(nb::init<std::string, bool, bool>(), nb::arg("backend"), nb::arg("raise_on_unsupported") = false,
             nb::arg("raise_on_fallback") = false)
        .def("close", &BackendGuardWrapper::close);

    m.def("is_backend_set", &isBackendSet);
    m.def("should_raise_on_unsupported", &shouldRaiseOnUnsupported);
    m.def("should_raise_on_fallback", &shouldRaiseOnFallback);
    m.def(
        "get_current_backend",
        [](const std::vector<std::string>& supported, std::optional<bool> raiseOnUnsupported)
        { return getCurrentBackend(supported, raiseOnUnsupported); },
        nb::arg("supported_backends"), nb::arg("raise_on_unsupported") = nb::none());
    // Stage.h
    nb::class_<StageGuardWrapper>(m, "_StageGuard")
        .def(nb::init<objects::Stage>(), nb::arg("stage"))
        .def("close", &StageGuardWrapper::close);

    m.def("set_default_stage", &setDefaultStage, nb::arg("stage"));
    m.def("get_default_stage", &getDefaultStage);
    m.def("get_active_stage", &getActiveStage);
    // Prim.h
    m.def("find_matching_prim_paths", &findMatchingPrimPaths, nb::arg("path"), nb::kw_only(),
          nb::arg("traverse") = false);
    m.def("get_all_matching_child_prims", &getAllMatchingChildPrims, nb::arg("path"), nb::kw_only(),
          nb::arg("predicate"), nb::arg("include_self") = false, nb::arg("max_depth") = nb::none());
    m.def("get_first_matching_child_prim", &getFirstMatchingChildPrim, nb::arg("path"), nb::kw_only(),
          nb::arg("predicate"), nb::arg("include_self") = false);
    m.def("get_first_matching_parent_prim", &getFirstMatchingParentPrim, nb::arg("path"), nb::kw_only(),
          nb::arg("predicate"), nb::arg("include_self") = false);
    // Semantics.h
    m.def("add_labels", &addLabels, nb::arg("path"), nb::kw_only(), nb::arg("labels"), nb::arg("taxonomy") = "class");
    m.def("get_labels", &getLabels, nb::arg("path"), nb::kw_only(), nb::arg("include_descendants") = false);
    m.def("remove_labels", &removeLabels, nb::arg("path"), nb::kw_only(), nb::arg("labels"),
          nb::arg("taxonomy") = nb::none(), nb::arg("include_descendants") = false);
    m.def("remove_all_labels", &removeAllLabels, nb::arg("path"), nb::kw_only(), nb::arg("remove_taxonomies") = false,
          nb::arg("include_descendants") = false);
}
