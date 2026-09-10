# Public API for module isaacsim.zmq.core:

## Classes

- class ZmqPublishSocket
  - def __init__(self, ip: str, port: int)
  - def send_multipart(self, topic: str, payload: bytes) -> bool
  - [property] def ip(self) -> str
  - [property] def port(self) -> int

- class ZmqSubscribeSocket
  - def __init__(self, ip: str, port: int, topic: str)
  - def try_recv(self) -> object
  - [property] def ip(self) -> str
  - [property] def port(self) -> int
  - [property] def topic(self) -> str
