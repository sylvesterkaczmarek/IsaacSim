# Overview

The `isaacsim.common` extension packages the common modules for Kit and connects logging and profiling to the
application's Carbonite services. Callers use the same `isaacsim.common.logging` and `isaacsim.common.profiling` APIs
inside and outside Kit.

## Functionality

The native adapter makes `isaacsim.common.logging` borrow Kit's existing Carbonite logging interface. It does not
acquire, release, or configure Kit's Carbonite framework, so the Kit application continues to own process-wide
logging settings.

Logger construction only caches channel state. It does not initialize the standalone backend, so feature modules may
construct loggers before this extension starts. The adapter registers those existing channels when it attaches Kit's
logging interface.

Extensions that depend on `isaacsim.common` can use the logging API:

```python
from isaacsim.common.logging import Logger

logger = Logger("isaacsim.example.feature")
logger.info("The feature started")
logger.warning("The fallback path was selected", capture_source=True)
```

Python source capture is disabled by default. Pass `capture_source=True` to an individual logging call when its
filename, function, and line should be forwarded to the application logging system.

Runtime libraries can add profiling zones without depending on Kit or a specific profiler. The extension attaches
`isaacsim.common.profiling` to Kit's Carbonite profiler for its lifetime. Profiling calls are no-ops when the
application profiler does not admit them.
