# Physics Manager Test Resources

This directory contains USD scenes shared by the physics manager and physics-engine
integration tests.

Tests locate this directory through `ISAACSIM_TEST_RESOURCE_ROOT`. Native tests use it
as their working directory through the module CMake `RESOURCE_DIR` contract.
