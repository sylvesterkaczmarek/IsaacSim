# Public API for module isaacsim.code_editor.python_server:

## Classes

- class BoundEndpoint
  - host: str
  - port: int

- class ServerState(Enum)
  - STARTING: str
  - RUNNING: str
  - STOPPING: str
  - STOPPED: str
  - RESTARTING: str
  - ERROR: str

- class ServerStatus
  - state: ServerState
  - configured_host: str
  - configured_port: int
  - bound_endpoints: tuple[BoundEndpoint, Ellipsis]
  - active_connections: int
  - authentication_required: bool
  - last_error: str | None

## Functions

- def get_server_endpoint() -> tuple[str, int] | None
- def get_server_status() -> ServerStatus | None
- async def restart_server(host: str, port: int) -> bool
- async def start_server() -> bool
- async def stop_server() -> bool
- def subscribe_server_status(callback: StatusCallback) -> Callable[[], None]
