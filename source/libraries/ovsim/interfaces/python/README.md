# OV SIM Python interfaces

This directory contains the canonical Python callable-type aliases for the OV SIM API. It is currently source-only:
no `source/libraries` distribution registers these files for staging or installation, and no Python provider is checked
against them. The packaged `isaacsim.foundation.ovsim` implementation currently exposes only the C++ contract.

When a Python OV SIM provider is added, it must expose concrete free functions whose signatures match the aliases it
implements and add provider-side static verification.

## Layout

```
ovsim/
├── __init__.py
└── interfaces/
    ├── __init__.py
    ├── data/
    │   └── __init__.py        # ovsim.interfaces.data
    └── control/
        ├── __init__.py
        ├── authoring/
        │   └── __init__.py    # ovsim.interfaces.control.authoring
        └── simulation/
            └── __init__.py    # ovsim.interfaces.control.simulation
```

## Verification

To enforce compliance at static-analysis time, assign each concrete provider function to a variable annotated with
the corresponding alias in a provider-side `verify_interfaces.py`. In this illustrative pattern, replace `provider`
with the implementation module:

```python
import provider

import ovsim.interfaces.control.authoring as iface_authoring
import ovsim.interfaces.control.simulation as iface_simulation
import ovsim.interfaces.data as iface_data

_create_stage: iface_authoring.CreateStageFn = provider.control.authoring.create_stage
_open_stage: iface_authoring.OpenStageFn = provider.control.authoring.open_stage
_define_prim: iface_authoring.DefinePrimFn = provider.control.authoring.define_prim
_play: iface_simulation.PlayFn = provider.control.simulation.play
_read: iface_data.ReadFn = provider.data.read
_write: iface_data.WriteFn = provider.data.write
```

A static type checker such as mypy or pyright reports a mismatching signature on the assignment. These aliases do not
perform runtime validation.

## Notes

- Default argument values are not captured by `Callable` aliases — packages may
  differ in which parameters carry defaults without triggering a static type checker
  error here; document canonical defaults in the package module.
- Callers invoking through an alias must pass every parameter explicitly, even
  those that have defaults in the underlying implementation.
