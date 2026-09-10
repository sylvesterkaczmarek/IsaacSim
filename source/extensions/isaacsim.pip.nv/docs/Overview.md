# Overview

The isaacsim.pip.nv extension bundles the NVIDIA CUDA pip packages (the `nvidia-*-cu12` runtime wheels and `cuda-bindings`) required by PyTorch on Linux. It is loaded early in the extension startup order so that these CUDA libraries are available before any extension that depends on PyTorch.

This extension is managed automatically and does not require manual configuration.
