// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Python.h must precede headers that can include the C or C++ standard library.
// clang-format off
#include <Python.h>
// clang-format on

#include <isaacsim/common/logging/Logging.hpp>

#include <exception>

namespace
{

PyObject* sayHello(PyObject*, PyObject*)
{
    try
    {
        isaacsim::common::logging::Logger logger("isaacsim.examples.hello_world.cpp_python");
        logger.report("Hello World from C++ with Python.");
        Py_RETURN_NONE;
    }
    catch (const std::exception& error)
    {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
    catch (...)
    {
        PyErr_SetString(PyExc_RuntimeError, "The Isaac Sim C++ API reported an unknown error.");
        return nullptr;
    }
}

PyMethodDef g_methods[] = {
    { "say_hello", sayHello, METH_NOARGS, "Print the Hello World result through the Isaac Sim C++ logging API." },
    { nullptr, nullptr, 0, nullptr },
};

PyModuleDef g_module = {
    PyModuleDef_HEAD_INIT, "_hello_world_cpp", "C++ with Python Isaac Sim Hello World example.", -1, g_methods,
};

} // namespace

PyMODINIT_FUNC PyInit__hello_world_cpp()
{
    return PyModule_Create(&g_module);
}
