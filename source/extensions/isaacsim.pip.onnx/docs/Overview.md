# ONNX Pip Archive

# Overview

The `isaacsim.pip.onnx` extension provides Python packages for ONNX model inference in Isaac Sim.

## Functionality

- Currently installs `onnxruntime-gpu` on x86_64 and the CPU `onnxruntime` package on Linux AArch64.
- Uses the CUDA runtime libraries supplied by `isaacsim.pip.nv` for Linux GPU inference.
- Supports loading and running ONNX models for robot policy inference within Isaac Sim workflows.
