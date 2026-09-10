# Overview

The isaacsim.test.action_event_sdg extension hosts downstream integration tests for the Action and Event Data Generation extensions (Isaac Sim Replicator Agent, `isaacsim.replicator.agent.core`, and its companions such as `omni.metropolis.utils`). These extensions are authored in the metrosim repository, published to the extension registry, and consumed by Isaac Sim as exact-version pins in the `isaacsim.exp.action_and_event_data_generation.*` app kit files. This extension runs tests against whatever versions those kit files pin, so version bumps are validated by this repository's test pipeline before they merge.

## Functionality

**End-to-End Data Generation Validation** - `TestDataGen` opens the test stage shipped with the pinned `isaacsim.replicator.agent.core` package, loads its data-generation config, runs the full asynchronous Replicator pipeline, and verifies that every camera sensor produces the expected number of RGB, object-detection, and camera-parameter frames.

**Writer Unit Tests** - `TestWriterUtils` and `TestIRABasicWriter` validate the writer utility functions and the `IRABasicWriter` used to serialize generated data.

**Simulation Manager Tests** - `TestSimulationManager` validates the simulation lifecycle management (configuration loading, data-generation orchestration) using mocked pipeline components.

## Integration

The test modules are copies of the upstream test suites from the metrosim repository (`isaacsim.replicator.agent.core` tests). They intentionally resolve test data through the `${isaacsim.replicator.agent.core}` extension token, so both the code under test and its data come from the installed registry package rather than from this repository. When the pinned versions change, rerunning this extension's tests exercises the new packages directly.
