# OV SIM C++ interfaces

This directory contains the canonical C++ function-pointer type aliases for the OV SIM API. It is shared interface
source, not a separately registered module.

`isaacsim.foundation.ovsim` is the current implementation. Its development component installs these canonical headers
with the implementation headers because the public foundation API includes them. Any additional OV SIM implementation
must provide concrete free functions whose signatures match the aliases it implements.

## Layout

```
ovsim/interfaces/
├── control/
│   ├── authoring/
│   │   └── Authoring.hpp   # ovsim::interfaces::control::authoring
│   └── simulation/
│       └── Simulation.hpp  # ovsim::interfaces::control::simulation
├── data/
│   └── Data.hpp            # ovsim::interfaces::data
└── details/
    └── Exception.hpp       # Shared implementation exceptions
```

## Verification

`source/libraries/isaacsim/foundation/ovsim/src/VerifyInterfaces.cpp` enforces the current contract at compile time. It
includes each canonical interface header alongside the concrete foundation header, then assigns the concrete function
addresses to variables of the alias types:

```cpp
#include <isaacsim/foundation/ovsim/control/authoring/Authoring.hpp>
#include <isaacsim/foundation/ovsim/control/simulation/Simulation.hpp>
#include <isaacsim/foundation/ovsim/data/Data.hpp>
#include <ovsim/interfaces/control/authoring/Authoring.hpp>
#include <ovsim/interfaces/control/simulation/Simulation.hpp>
#include <ovsim/interfaces/data/Data.hpp>

namespace iface_authoring = ovsim::interfaces::control::authoring;
namespace iface_simulation = ovsim::interfaces::control::simulation;
namespace iface_data = ovsim::interfaces::data;
namespace ns_control = isaacsim::foundation::ovsim::control;
namespace ns_data = isaacsim::foundation::ovsim::data;

[[maybe_unused]] iface_data::ReadFn _read = &ns_data::read;
[[maybe_unused]] iface_authoring::CreateStageFn _createStage = &ns_control::authoring::createStage;
[[maybe_unused]] iface_simulation::PlayFn _play = &ns_control::simulation::play;
```

A mismatching return type, parameter type, or parameter count causes a compile error on that assignment.

## Notes

- Default argument values are not part of a function pointer type. Packages may
  differ in which parameters carry defaults without triggering a compile error
  here; document canonical defaults in the package header.
- Callers invoking through an alias must pass every parameter explicitly, even
  those that have defaults in the concrete declaration.
