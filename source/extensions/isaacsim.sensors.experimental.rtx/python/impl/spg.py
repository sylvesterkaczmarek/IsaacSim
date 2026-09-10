# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Authoring helpers for RTX Sensor Processing Graphs (SPG).

This module provides :class:`SPGNode`, a lightweight description of a single SPG
shader node, and :func:`_author_spg`, which authors the USD prim structure for a
graph of such nodes onto a sensor's render product.

The API deliberately performs only *structural* validation (file co-location and
prim wiring). All CUDA / Lua / SPG semantic validation (kernel signatures, dtypes,
launch configuration, graph semantics) is owned by the ``omni.rtx.spg`` extension
at render time. See the SPG documentation:
https://docs.omniverse.nvidia.com/kit/docs/omni.rtx.spg/0.2.0/Overview.html
"""

from __future__ import annotations

import pathlib
import shutil
from dataclasses import dataclass, field
from typing import Any

import carb
import isaacsim.core.experimental.utils.stage as stage_utils
from pxr import Gf, Sdf, Usd, UsdRender, UsdShade

# Render context (source type) used for SPG source assets, producing the
# ``info:spg:sourceAsset`` / ``info:spg:sourceAsset:subIdentifier`` attributes.
_SPG_SOURCE_TYPE = "spg"
# Opaque AOV attribute authored on a ``RenderVar`` prim.
_AOV_ATTR = "omni:rtx:aov"


@dataclass
class SPGNode:
    """Description of a single SPG shader node.

    An SPG shader node is authored as a ``UsdShade.Shader`` prim that references a
    CUDA source (``.cu``) file via ``info:spg:sourceAsset`` and names an entry-point
    function via ``info:spg:sourceAsset:subIdentifier``. The ``subIdentifier`` must
    match both the ``extern "C"`` CUDA function name and the Lua function name in the
    co-located launch script.

    On construction, only *structural* validation is performed: the ``.cu`` file must
    exist and the co-located Lua launch script (``<cuda_kernel>.lua``, e.g.
    ``GrayscaleKernel.cu`` -> ``GrayscaleKernel.cu.lua``) must exist next to it. No
    CUDA or Lua semantic validation is performed; that is left to ``omni.rtx.spg``.

    Multiple :class:`SPGNode` instances may reference the same ``.cu`` file with
    different ``sub_identifier`` values (a single ``.cu`` may host several kernels).

    Args:
        name: Unique name for the ``Shader`` prim within the render product.
        cuda_kernel: Path to the ``.cu`` CUDA source file.
        sub_identifier: Entry-point function name (``extern "C"`` kernel and Lua
            function). Defaults to ``name`` when ``None``.
        inputs: Opaque input attribute names (authored as ``opaque inputs:<name>``).
        outputs: Opaque output attribute names (authored as ``opaque outputs:<name>``).
        params: Typed parameter attribute names mapped to values (authored as typed
            ``inputs:<name>``). Supported value types: ``bool``, ``int``, ``float``.

    Raises:
        ValueError: If ``cuda_kernel`` does not have a ``.cu`` suffix, or if input/param
            names collide or output names are not unique.
        FileNotFoundError: If the ``.cu`` file or its co-located ``.cu.lua`` launch
            script does not exist.
        TypeError: If a ``params`` value is not a ``bool``, ``int``, or ``float``.

    Example:

    .. code-block:: python

        >>> from isaacsim.sensors.experimental.rtx import SPGNode
        >>>
        >>> node = SPGNode(
        ...     "GrayscaleKernel",
        ...     "kernels/GrayscaleKernel.cu",
        ...     inputs=["LdrColor"],
        ...     outputs=["LdrGrayscale"],
        ... )  # doctest: +SKIP
    """

    name: str
    cuda_kernel: str
    sub_identifier: str | None = None
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict[str, bool | int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cu = pathlib.Path(self.cuda_kernel)
        if cu.suffix != ".cu":
            raise ValueError(f"SPGNode '{self.name}': cuda_kernel must be a '.cu' file, got '{self.cuda_kernel}'")
        if not cu.is_file():
            raise FileNotFoundError(f"SPGNode '{self.name}': CUDA kernel file not found: '{self.cuda_kernel}'")
        # SPG derives the launch script by appending '.lua' to the '.cu' path.
        lua = cu.with_name(cu.name + ".lua")
        if not lua.is_file():
            raise FileNotFoundError(
                f"SPGNode '{self.name}': co-located Lua launch script not found: '{lua}'. "
                f"SPG requires a launch script named '{cu.name}.lua' next to '{cu.name}'."
            )
        # Opaque inputs and typed params share the ``inputs:`` attribute namespace,
        # so their names (and the names within each list) must be unique.
        input_names = list(self.inputs) + list(self.params.keys())
        if len(input_names) != len(set(input_names)):
            raise ValueError(
                f"SPGNode '{self.name}': input and param names must be unique within the "
                f"'inputs:' namespace, got inputs={self.inputs} params={list(self.params.keys())}"
            )
        if len(self.outputs) != len(set(self.outputs)):
            raise ValueError(f"SPGNode '{self.name}': output names must be unique, got {self.outputs}")
        # Params are authored as typed shader inputs; only scalar types are supported.
        # Validate here so misuse fails at construction rather than late in authoring.
        for param_name, param_value in self.params.items():
            if not isinstance(param_value, (bool, int, float)):
                raise TypeError(
                    f"SPGNode '{self.name}': unsupported param value type "
                    f"'{type(param_value).__name__}' for param '{param_name}' (supported: bool, int, float)"
                )

        self._resolved_cu = str(cu.resolve())
        self._resolved_lua = str(lua.resolve())
        if self.sub_identifier is None:
            self.sub_identifier = self.name


def _sdf_type_for_value(value: Any) -> Sdf.ValueTypeName:
    """Return the ``Sdf`` value type for a scalar parameter value.

    Args:
        value: Scalar parameter value (``bool``, ``int``, or ``float``).

    Returns:
        The matching ``Sdf.ValueTypeName``.

    Raises:
        TypeError: If the value type is not supported.
    """
    # bool must be checked before int (bool is a subclass of int).
    if isinstance(value, bool):
        return Sdf.ValueTypeNames.Bool
    if isinstance(value, int):
        return Sdf.ValueTypeNames.Int
    if isinstance(value, float):
        return Sdf.ValueTypeNames.Float
    raise TypeError(f"Unsupported SPG param value type '{type(value).__name__}' (supported: bool, int, float)")


def _endpoint_is_port(endpoint: str) -> bool:
    """Return whether a connection endpoint refers to a shader port.

    A shader-port endpoint has the form ``"<NodeName>.inputs:<X>"`` or
    ``"<NodeName>.outputs:<Y>"``. Any other endpoint is treated as a bare AOV name.

    Args:
        endpoint: Connection endpoint string.

    Returns:
        True if the endpoint refers to a shader port, False for a bare AOV name.
    """
    if "." not in endpoint:
        return False
    _, attr = endpoint.split(".", 1)
    return attr.startswith(("inputs:", "outputs:"))


def _port_attr_path(rp_path: str, endpoint: str) -> Sdf.Path:
    """Return the attribute path for a shader-port endpoint.

    Args:
        rp_path: Render product prim path.
        endpoint: Shader-port endpoint (e.g. ``"Grayscale.outputs:LdrGrayscale"``).

    Returns:
        Absolute attribute path of the shader port.
    """
    node, attr = endpoint.split(".", 1)
    return Sdf.Path(f"{rp_path}/{node}.{attr}")


def _find_render_product_for_sensor(
    stage: Usd.Stage, sensor_prim_path: str, asset_root_path: str | None
) -> Usd.Prim | None:
    """Find a render product whose ``camera`` relationship targets the sensor prim.

    When ``asset_root_path`` is set, the search is scoped to that subtree; otherwise
    the whole stage is searched. Render products are matched wherever they are
    authored (including under a ``/Render`` scope, as the canonical ``omni.rtx.spg``
    examples author them), so an existing product tied to the sensor is reused rather
    than duplicated.

    Args:
        stage: The USD stage.
        sensor_prim_path: Path of the sensor prim the render product must target.
        asset_root_path: Optional subtree root to scope the search to.

    Returns:
        The matching ``RenderProduct`` prim, or ``None`` when none is found.
    """
    if asset_root_path is not None:
        root_prim = stage.GetPrimAtPath(asset_root_path)
        if not root_prim.IsValid():
            return None
        search = Usd.PrimRange(root_prim)
    else:
        search = stage.Traverse()
    for prim in search:
        if prim.GetTypeName() == "RenderProduct" and prim.HasRelationship("camera"):
            targets = prim.GetRelationship("camera").GetTargets()
            if len(targets) == 1 and str(targets[0]) == sensor_prim_path:
                return prim
    return None


def _init_render_product(rp: UsdRender.Product, sensor_prim_path: str, resolution: tuple[int, int]) -> None:
    """Initialize a newly created render product with a camera target and resolution.

    Args:
        rp: The render product to initialize.
        sensor_prim_path: Sensor prim path to target via the ``camera`` relationship.
        resolution: Render product resolution as ``(width, height)``.
    """
    rp.CreateCameraRel().SetTargets([Sdf.Path(sensor_prim_path)])
    rp.CreateResolutionAttr(Gf.Vec2i(int(resolution[0]), int(resolution[1])))


def _find_or_create_render_product(
    stage: Usd.Stage,
    sensor_prim_path: str,
    render_product: str | None,
    resolution: tuple[int, int],
    asset_root_path: str | None,
) -> Usd.Prim:
    """Resolve the render product to author the graph onto, creating one if needed.

    Args:
        stage: The USD stage.
        sensor_prim_path: Sensor prim path.
        render_product: Explicit render product path, or ``None`` to find-or-create.
        resolution: Resolution as ``(width, height)`` for newly created products.
        asset_root_path: Optional subtree root to scope the find to.

    Returns:
        The resolved (existing or newly created) ``RenderProduct`` prim.
    """
    if render_product is not None:
        prim = stage.GetPrimAtPath(render_product)
        if prim.IsValid():
            if prim.GetTypeName() != "RenderProduct":
                raise ValueError(
                    f"Prim at render_product path '{render_product}' is a "
                    f"'{prim.GetTypeName()}', expected a 'RenderProduct'."
                )
            return prim
        rp = UsdRender.Product.Define(stage, render_product)
        _init_render_product(rp, sensor_prim_path, resolution)
        return rp.GetPrim()

    found = _find_render_product_for_sensor(stage, sensor_prim_path, asset_root_path)
    if found is not None:
        return found

    parent, _, name = sensor_prim_path.rpartition("/")
    rp_path = f"{parent}/{name}_RenderProduct"
    rp = UsdRender.Product.Define(stage, rp_path)
    _init_render_product(rp, sensor_prim_path, resolution)
    return rp.GetPrim()


def _ensure_render_var(
    stage: Usd.Stage,
    rp_prim: Usd.Prim,
    rp_path: str,
    aov_name: str,
    ordered: list[Sdf.Path],
    *,
    is_source: bool,
) -> str:
    """Ensure a ``RenderVar`` for ``aov_name`` exists on the render product.

    Reuses an existing ``RenderVar`` child whose ``sourceName`` matches ``aov_name``.
    When missing and ``is_source`` is True (a source AOV that a shader consumes), a
    warning is logged before the ``RenderVar`` is created. When missing and
    ``is_source`` is False (a shader output AOV), it is created without a warning.

    Args:
        stage: The USD stage.
        rp_prim: The render product prim.
        rp_path: The render product prim path.
        aov_name: The AOV ``sourceName`` to find or create.
        ordered: Mutable list of ``orderedVars`` targets; appended to when needed.
        is_source: Whether the AOV is consumed as a shader input (source AOV).

    Returns:
        Path of the matching or newly created ``RenderVar`` prim.
    """
    for child in rp_prim.GetChildren():
        if child.GetTypeName() == "RenderVar" and child.HasAttribute("sourceName"):
            if child.GetAttribute("sourceName").Get() == aov_name:
                rv_path = str(child.GetPath())
                rv_sdf = Sdf.Path(rv_path)
                if rv_sdf not in ordered:
                    ordered.append(rv_sdf)
                return rv_path

    if is_source:
        carb.log_warn(
            f"SPG: source AOV '{aov_name}' has no RenderVar on render product '{rp_path}'. "
            f"Creating one. Verify the RTX renderer produces this AOV."
        )
    rv_path = f"{rp_path}/{aov_name}"
    rv_prim = stage.DefinePrim(rv_path, "RenderVar")
    rv_prim.CreateAttribute("sourceName", Sdf.ValueTypeNames.String, custom=False).Set(aov_name)
    rv_prim.CreateAttribute(_AOV_ATTR, Sdf.ValueTypeNames.Opaque, custom=False)
    rv_sdf = Sdf.Path(rv_path)
    if rv_sdf not in ordered:
        ordered.append(rv_sdf)
    return rv_path


def _copy_sources(nodes: list[SPGNode], copy_to: str) -> dict[str, str]:
    """Copy each distinct ``.cu`` and its co-located ``.cu.lua`` into ``copy_to``.

    Each distinct ``.cu`` (and its launch script) is copied exactly once, even when
    multiple nodes reference the same file.

    Args:
        nodes: Nodes whose CUDA sources are copied.
        copy_to: Destination directory (created if it does not exist).

    Returns:
        Mapping from each node's resolved ``.cu`` path to the copied file path.
    """
    dst_dir = pathlib.Path(copy_to)
    dst_dir.mkdir(parents=True, exist_ok=True)
    cu_ref_map: dict[str, str] = {}
    for node in nodes:
        if node._resolved_cu in cu_ref_map:
            continue
        cu_src = pathlib.Path(node._resolved_cu)
        lua_src = pathlib.Path(node._resolved_lua)
        cu_dst = dst_dir / cu_src.name
        lua_dst = dst_dir / lua_src.name
        shutil.copy2(cu_src, cu_dst)
        shutil.copy2(lua_src, lua_dst)
        cu_ref_map[node._resolved_cu] = str(cu_dst)
    return cu_ref_map


def _author_spg(
    sensor_prim_path: str,
    nodes: list[SPGNode],
    connections: list[tuple[str, str]] | None,
    *,
    render_product: str | None,
    resolution: tuple[int, int],
    copy_to: str | None,
    asset_root_path: str | None,
) -> str:
    """Author the USD prim structure for an SPG graph onto a sensor's render product.

    Args:
        sensor_prim_path: Path of the sensor prim the graph is authored for.
        nodes: SPG shader nodes to author.
        connections: ``(src, dst)`` endpoint pairs. A bare endpoint names an AOV; a
            ``"Node.inputs:X"`` / ``"Node.outputs:Y"`` endpoint names a shader port.
            The connection is authored on the consumer (``dst``) attribute.
        render_product: Explicit render product path, or ``None`` to find-or-create.
        resolution: Resolution as ``(width, height)`` for newly created products.
        copy_to: Optional directory to copy the ``.cu`` / ``.cu.lua`` sources into.
        asset_root_path: Optional subtree root used when finding an existing product.

    Returns:
        The render product prim path.
    """
    stage = stage_utils.get_current_stage(backend="usd")

    # Dry-run pre-pass: compute the ports every node *will* declare and validate all
    # connection endpoints before authoring anything. This ensures a structurally
    # invalid graph fails cleanly, without leaving partially-authored shader prims on
    # the stage (a bad connection used to raise mid-authoring, after prims existed).
    declared_ports: set[str] = set()
    for node in nodes:
        for input_name in node.inputs:
            declared_ports.add(f"{node.name}.inputs:{input_name}")
        for output_name in node.outputs:
            declared_ports.add(f"{node.name}.outputs:{output_name}")
        for param_name in node.params:
            declared_ports.add(f"{node.name}.inputs:{param_name}")

    for src, dst in connections or []:
        for endpoint in (src, dst):
            if _endpoint_is_port(endpoint) and endpoint not in declared_ports:
                raise ValueError(
                    f"SPG connection endpoint '{endpoint}' does not match any declared shader port. "
                    f"Declared ports: {sorted(declared_ports)}"
                )

    rp_prim = _find_or_create_render_product(stage, sensor_prim_path, render_product, resolution, asset_root_path)
    rp_path = str(rp_prim.GetPath())

    cu_ref_map = _copy_sources(nodes, copy_to) if copy_to is not None else {}

    # Author the shader prims (directly under the render product). Ports were already
    # validated against ``declared_ports`` in the pre-pass above.
    for node in nodes:
        shader = UsdShade.Shader.Define(stage, f"{rp_path}/{node.name}")
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        cu_ref = cu_ref_map.get(node._resolved_cu, node.cuda_kernel)
        shader.SetSourceAsset(Sdf.AssetPath(cu_ref), _SPG_SOURCE_TYPE)
        shader.SetSourceAssetSubIdentifier(node.sub_identifier, _SPG_SOURCE_TYPE)
        for input_name in node.inputs:
            shader.CreateInput(input_name, Sdf.ValueTypeNames.Opaque)
        for output_name in node.outputs:
            shader.CreateOutput(output_name, Sdf.ValueTypeNames.Opaque)
        for param_name, param_value in node.params.items():
            shader.CreateInput(param_name, _sdf_type_for_value(param_value)).Set(param_value)

    # Wire connections. The consumer is always the destination endpoint.
    ordered: list[Sdf.Path] = list(rp_prim.GetRelationship("orderedVars").GetTargets())
    for src, dst in connections or []:
        if _endpoint_is_port(dst):
            dst_attr = stage.GetAttributeAtPath(_port_attr_path(rp_path, dst))
        else:
            rv_path = _ensure_render_var(stage, rp_prim, rp_path, dst, ordered, is_source=False)
            dst_attr = stage.GetPrimAtPath(rv_path).GetAttribute(_AOV_ATTR)

        if _endpoint_is_port(src):
            src_path = _port_attr_path(rp_path, src)
        else:
            rv_path = _ensure_render_var(stage, rp_prim, rp_path, src, ordered, is_source=True)
            src_path = Sdf.Path(f"{rv_path}.{_AOV_ATTR}")

        if src_path not in dst_attr.GetConnections():
            dst_attr.AddConnection(src_path)

    rp_prim.CreateRelationship("orderedVars").SetTargets(ordered)
    return rp_path
