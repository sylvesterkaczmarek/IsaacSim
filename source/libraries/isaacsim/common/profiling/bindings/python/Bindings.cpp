// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaacsim/common/profiling/Profiling.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <string>

namespace nb = nanobind;
using namespace isaacsim::common::profiling;

namespace
{

struct PythonLocation
{
    std::string file;
    std::string function;
    std::uint32_t line{ 0 };

    SourceLocation view() const noexcept
    {
        return makeSourceLocation(
            file.empty() ? nullptr : file.c_str(), function.empty() ? nullptr : function.c_str(), line);
    }
};

std::string getUnicodeAttribute(PyObject* object, const char* name)
{
    PyObject* attribute = PyObject_GetAttrString(object, name);
    if (attribute == nullptr)
    {
        PyErr_Clear();
        return {};
    }
    Py_ssize_t size = 0;
    const char* text = PyUnicode_AsUTF8AndSize(attribute, &size);
    std::string result;
    if (text != nullptr)
    {
        result.assign(text, static_cast<std::size_t>(size));
    }
    else
    {
        PyErr_Clear();
    }
    Py_DECREF(attribute);
    return result;
}

PythonLocation captureLocation()
{
    PythonLocation result;
    PyFrameObject* frame = PyEval_GetFrame();
    if (frame == nullptr)
    {
        return result;
    }
    const int line = PyFrame_GetLineNumber(frame);
    result.line = line > 0 ? static_cast<std::uint32_t>(line) : 0;
    PyCodeObject* code = PyFrame_GetCode(frame);
    if (code != nullptr)
    {
        PyObject* object = reinterpret_cast<PyObject*>(code);
        result.file = getUnicodeAttribute(object, "co_filename");
        result.function = getUnicodeAttribute(object, "co_name");
        Py_DECREF(object);
    }
    else
    {
        PyErr_Clear();
    }
    return result;
}

} // namespace

NB_MODULE(_bindings, module)
{
    module.doc() = "Low-overhead profiling through an application-owned Carbonite profiler.";

    nb::enum_<InstantType>(module, "InstantType", "Timeline occupied by an instant event.")
        .value("THREAD", InstantType::eThread, "Emit on the calling thread timeline.")
        .value("PROCESS", InstantType::eProcess, "Emit on the process timeline.");

    nb::class_<Zone>(module, "Zone", "Move-only ownership of one active profiling zone.")
        .def("close", &Zone::close, "End the profiling zone if it is active.")
        .def_prop_ro(
            "active", [](const Zone& zone) { return static_cast<bool>(zone); }, "Whether the zone is active.")
        .def(
            "__enter__", [](Zone& zone) -> Zone& { return zone; }, nb::rv_policy::reference)
        .def("__exit__", [](Zone& zone, nb::args) { zone.close(); });

    module.def("is_enabled", &isEnabled, nb::arg("mask") = 0,
               "Return whether the attached Carbonite profiler admits the selected mask.");

    module.def(
        "begin",
        [](const std::string& name, std::uint64_t mask, bool captureSource, const std::string& sourceFile,
           const std::string& sourceFunction, std::uint32_t sourceLine)
        {
            if (!isEnabled(mask))
            {
                return Zone{};
            }
            PythonLocation location;
            if (captureSource)
            {
                if (!sourceFile.empty() || !sourceFunction.empty() || sourceLine != 0)
                {
                    location = PythonLocation{ sourceFile, sourceFunction, sourceLine };
                }
                else
                {
                    location = captureLocation();
                }
            }
            Zone result;
            {
                nb::gil_scoped_release release;
                result = Zone(name, mask, location.view());
            }
            return result;
        },
        nb::arg("name"), nb::arg("mask") = 0, nb::arg("capture_source") = true, nb::arg("source_file") = "",
        nb::arg("source_function") = "", nb::arg("source_line") = 0, "Begin a profiling zone.");

    module.def(
        "frame",
        [](const std::string& name, std::uint64_t mask)
        {
            nb::gil_scoped_release release;
            frame(name, mask);
        },
        nb::arg("name"), nb::arg("mask") = 0, "Emit a frame marker for the calling thread.");

    module.def(
        "value",
        [](const std::string& name, nb::handle selected, std::uint64_t mask)
        {
            if (nb::isinstance<nb::int_>(selected) && !nb::isinstance<nb::bool_>(selected))
            {
                const std::int32_t integer = nb::cast<std::int32_t>(selected);
                nb::gil_scoped_release release;
                value(name, integer, mask);
                return;
            }
            if (nb::isinstance<nb::float_>(selected))
            {
                const float floatingPoint = nb::cast<float>(selected);
                nb::gil_scoped_release release;
                value(name, floatingPoint, mask);
                return;
            }
            throw nb::type_error("Profiling values must be int or float");
        },
        nb::arg("name"), nb::arg("selected"), nb::arg("mask") = 0, "Emit a numeric value.");

    module.def(
        "instant",
        [](const std::string& name, InstantType type, std::uint64_t mask, bool captureSource)
        {
            if (!isEnabled(mask))
            {
                return;
            }
            const PythonLocation location = captureSource ? captureLocation() : PythonLocation{};
            nb::gil_scoped_release release;
            instant(name, type, mask, location.view());
        },
        nb::arg("name"), nb::arg("type") = InstantType::eThread, nb::arg("mask") = 0, nb::arg("capture_source") = true,
        "Emit an instant event.");

    module.def(
        "flow_begin",
        [](std::uint64_t identifier, const std::string& name, std::uint64_t mask, bool captureSource)
        {
            if (!isEnabled(mask))
            {
                return;
            }
            const PythonLocation location = captureSource ? captureLocation() : PythonLocation{};
            nb::gil_scoped_release release;
            flow(true, identifier, name, mask, location.view());
        },
        nb::arg("identifier"), nb::arg("name"), nb::arg("mask") = 0, nb::arg("capture_source") = true,
        "Emit the beginning of a cross-thread flow.");

    module.def(
        "flow_end",
        [](std::uint64_t identifier, std::uint64_t mask, bool captureSource)
        {
            if (!isEnabled(mask))
            {
                return;
            }
            const PythonLocation location = captureSource ? captureLocation() : PythonLocation{};
            nb::gil_scoped_release release;
            flow(false, identifier, {}, mask, location.view());
        },
        nb::arg("identifier"), nb::arg("mask") = 0, nb::arg("capture_source") = true,
        "Emit the end of a cross-thread flow.");

    module.def(
        "set_thread_name",
        [](const std::string& name)
        {
            nb::gil_scoped_release release;
            setThreadName(name);
        },
        nb::arg("name"), "Assign a profiler-visible name to the calling thread.");
}
