# Public API for module isaacsim.replicator.behavior:

## Classes

- class BaseBehavior(BehaviorScript)
  - BEHAVIOR_NS: str
  - VARIABLES_TO_EXPOSE: List
  - def on_init(self)
  - def on_destroy(self)
  - def on_play(self)
  - def on_stop(self)
  - def on_update(self, current_time: float, delta_time: float)

- class ExampleBaseBehavior(BaseBehavior)
  - BEHAVIOR_NS: str
  - VARIABLES_TO_EXPOSE: Unknown
  - def on_init(self)
  - def on_destroy(self)
  - def on_play(self)
  - def on_stop(self)

- class ExampleBehavior(BehaviorScript)
  - BEHAVIOR_NS: str
  - VARIABLES_TO_EXPOSE: List
  - def on_init(self)
  - def on_destroy(self)
  - def on_play(self)
  - def on_stop(self)
  - def on_update(self, current_time: float, delta_time: float)

- class LightRandomizer(BehaviorScript)
  - BEHAVIOR_NS: str
  - VARIABLES_TO_EXPOSE: List
  - def on_init(self)
  - def on_destroy(self)
  - def on_play(self)
  - def on_stop(self)
  - def on_update(self, current_time: float, delta_time: float)
  - def set_rng(self, rng: np.random.Generator | None = None)

- class LocationRandomizer(BehaviorScript)
  - BEHAVIOR_NS: str
  - VARIABLES_TO_EXPOSE: List
  - def on_init(self)
  - def on_destroy(self)
  - def on_play(self)
  - def on_stop(self)
  - def on_update(self, current_time: float, delta_time: float)
  - def set_rng(self, rng: np.random.Generator | None = None)

- class LookAtBehavior(BehaviorScript)
  - BEHAVIOR_NS: str
  - VARIABLES_TO_EXPOSE: List
  - def on_init(self)
  - def on_destroy(self)
  - def on_play(self)
  - def on_stop(self)
  - def on_update(self, current_time: float, delta_time: float)

- class RotationRandomizer(BehaviorScript)
  - BEHAVIOR_NS: str
  - VARIABLES_TO_EXPOSE: List
  - def on_init(self)
  - def on_destroy(self)
  - def on_play(self)
  - def on_stop(self)
  - def on_update(self, current_time: float, delta_time: float)
  - def set_rng(self, rng: np.random.Generator | None = None)

- class TextureRandomizer(BehaviorScript)
  - BEHAVIOR_NS: str
  - VARIABLES_TO_EXPOSE: List
  - def on_init(self)
  - def on_destroy(self)
  - def on_play(self)
  - def on_stop(self)
  - def on_update(self, current_time: float, delta_time: float)
  - def set_rng(self, rng: np.random.Generator | None = None)

- class VolumeStackRandomizer(BehaviorScript)
  - BEHAVIOR_NS: str
  - EVENT_NAME_IN: Unknown
  - EVENT_NAME_OUT: Unknown
  - ACTION_FUNCTION_MAP: Dict
  - VARIABLES_TO_EXPOSE: List
  - def on_init(self)
  - def on_destroy(self)
  - def set_rng(self, rng: np.random.Generator | None = None)

## Functions

- def add_behavior_script(prim: Usd.Prim, script_path: str, allow_duplicates: bool = False)
- async def add_behavior_script_with_parameters_async(prim: Usd.Prim, script_path: str, exposed_variables: dict | None = None, allow_duplicates: bool = False)

## Variables

- EXPOSED_ATTR_NS: str
- EXPOSED_VARS_CHANGED_EVENT: Unknown
