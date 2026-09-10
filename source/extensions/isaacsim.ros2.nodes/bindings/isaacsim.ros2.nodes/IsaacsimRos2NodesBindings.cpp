// SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <carb/BindingsPythonUtils.h>
#include <carb/logging/Log.h>

#include <isaacsim/ros2/nodes/FillPointCloudBufferHost.hpp>
#include <isaacsim/ros2/nodes/IRos2Nodes.hpp>
#include <isaacsim/ros2/nodes/SrtxPublisherFactory.hpp>
#include <pybind11/stl.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

CARB_BINDINGS("isaacsim.ros2.nodes.python")

namespace
{

namespace py = pybind11;

// Returns the total byte size of a C-contiguous buffer, throwing on violations.
size_t getContiguousBufferSize(const py::buffer_info& info, const char* name)
{
    // Zero-size buffers hold no bytes, so contiguity is irrelevant (and numpy's stride
    // conventions for zero-length dimensions would trip the walk below).
    if (info.size == 0)
    {
        return 0;
    }
    ptrdiff_t expectedStride = info.itemsize;
    for (ptrdiff_t i = info.ndim - 1; i >= 0; --i)
    {
        if (info.shape[i] > 1 && info.strides[i] != expectedStride)
        {
            throw std::invalid_argument(std::string(name) + " buffer must be C-contiguous");
        }
        expectedStride *= info.shape[i];
    }
    return static_cast<size_t>(info.size) * static_cast<size_t>(info.itemsize);
}

// True when [ptr, ptr + bytes) overlaps [destination, destination + destinationBytes).
bool overlapsDestination(const void* destination, size_t destinationBytes, const void* ptr, size_t bytes)
{
    const uintptr_t dstBegin = reinterpret_cast<uintptr_t>(destination);
    const uintptr_t srcBegin = reinterpret_cast<uintptr_t>(ptr);
    return bytes != 0 && destinationBytes != 0 && srcBegin < dstBegin + destinationBytes && dstBegin < srcBegin + bytes;
}

py::capsule wrapDescriptorAsCapsule(isaacsim::ros2::nodes::SrtxFrameCallbackDescriptor* desc)
{
    return py::capsule(desc, "SrtxFrameCallbackDescriptor",
                       [](PyObject* cap)
                       {
                           auto* d = static_cast<isaacsim::ros2::nodes::SrtxFrameCallbackDescriptor*>(
                               PyCapsule_GetPointer(cap, "SrtxFrameCallbackDescriptor"));
                           if (d)
                           {
                               delete d;
                           }
                       });
}

PYBIND11_MODULE(_ros2_nodes, m)
{
    // clang-format off
    using namespace carb;
    using namespace isaacsim::ros2::nodes;

    m.doc() = R"pbdoc(
        Internal interface that is automatically called when the extension is loaded so that Omnigraph nodes are registered.

        Example:

            # import  isaacsim.ros2.nodes.bindings._ros2_nodes as _ros2_nodes

            # Acquire the interface
            interface = _ros2_nodes.acquire_interface()

            # Use the interface
            # ...

            # Release the interface
            _ros2_nodes.release_interface(interface)
    )pbdoc";

    defineInterfaceClass<IRos2Nodes>(
        m,
        "IRos2Nodes",
        "acquire_interface",
        "release_interface"
    );

    m.def("create_image_publisher_capsule",
          [](const std::string& topicName,
             const std::string& frameId,
             const std::string& nodeNamespace,
             uint64_t queueSize,
             const std::string& qosProfile) -> py::capsule
          {
              auto* desc = createImagePublisherDescriptor(
                  topicName, frameId, nodeNamespace, queueSize, qosProfile);
              if (!desc)
              {
                  throw std::runtime_error("Failed to initialize Ros2SrtxImagePublisher");
              }
              return wrapDescriptorAsCapsule(desc);
          },
          py::arg("topic_name"),
          py::arg("frame_id"),
          py::arg("node_namespace"),
          py::arg("queue_size"),
          py::arg("qos_profile") = "",
          R"pbdoc(
              Create a ROS 2 Image publisher and return a PyCapsule wrapping
              the C-ABI callback descriptor.

              The capsule is named "SrtxFrameCallbackDescriptor" and is intended
              to be passed to omni.replicator.srtx's register_frame_callback().

              Args:
                  topic_name: ROS 2 topic name to publish on.
                  frame_id: TF frame_id for the published message header.
                  node_namespace: ROS 2 node namespace.
                  queue_size: Publisher queue depth.
                  qos_profile: JSON-encoded QoS profile (empty string for defaults).

              Returns:
                  PyCapsule containing the callback descriptor.
          )pbdoc");

    m.def("create_camera_info_publisher_capsule",
          [](const std::string& topicName,
             const std::string& frameId,
             const std::string& nodeNamespace,
             uint64_t queueSize,
             const std::string& qosProfile,
             uint32_t width,
             uint32_t height,
             const std::string& distortionModel,
             const std::vector<double>& k,
             const std::vector<double>& r,
             const std::vector<double>& p,
             const std::vector<double>& d) -> py::capsule
          {
              auto* desc = createCameraInfoPublisherDescriptor(
                  topicName, frameId, nodeNamespace, queueSize, qosProfile, width, height, distortionModel, k, r, p, d);
              if (!desc)
              {
                  throw std::runtime_error("Failed to initialize Ros2SrtxCameraInfoPublisher");
              }
              return wrapDescriptorAsCapsule(desc);
          },
          py::arg("topic_name"),
          py::arg("frame_id"),
          py::arg("node_namespace"),
          py::arg("queue_size"),
          py::arg("qos_profile"),
          py::arg("width"),
          py::arg("height"),
          py::arg("distortion_model"),
          py::arg("k"),
          py::arg("r"),
          py::arg("p"),
          py::arg("d"),
          R"pbdoc(
              Create a ROS 2 CameraInfo publisher and return a PyCapsule wrapping
              the C-ABI callback descriptor.

              The capsule is named "SrtxFrameCallbackDescriptor" and is intended
              to be passed to omni.replicator.srtx's register_frame_callback().
          )pbdoc");

    m.def("create_lidar_publisher_capsule",
          [](const std::string& topicName,
             const std::string& frameId,
             const std::string& nodeNamespace,
             uint64_t queueSize,
             const std::string& qosProfile) -> py::capsule
          {
              auto* desc = createLidarPublisherDescriptor(
                  topicName, frameId, nodeNamespace, queueSize, qosProfile);
              if (!desc)
              {
                  throw std::runtime_error("Failed to initialize Ros2SrtxLidarPublisher");
              }
              return wrapDescriptorAsCapsule(desc);
          },
          py::arg("topic_name"),
          py::arg("frame_id"),
          py::arg("node_namespace"),
          py::arg("queue_size"),
          py::arg("qos_profile") = "",
          R"pbdoc(
              Create a ROS 2 PointCloud2 (lidar) publisher and return a PyCapsule
              wrapping the C-ABI callback descriptor.

              The capsule is named "SrtxFrameCallbackDescriptor" and is intended
              to be passed to omni.replicator.srtx's register_frame_callback().

              Args:
                  topic_name: ROS 2 topic name to publish on.
                  frame_id: TF frame_id for the published message header.
                  node_namespace: ROS 2 node namespace.
                  queue_size: Publisher queue depth.
                  qos_profile: JSON-encoded QoS profile (empty string for defaults).

              Returns:
                  PyCapsule containing the callback descriptor.
          )pbdoc");

    m.def("create_laser_scan_publisher_capsule",
          [](const std::string& topicName,
             const std::string& frameId,
             const std::string& nodeNamespace,
             uint64_t queueSize,
             const std::string& qosProfile,
             float azimuthRangeStart,
             float azimuthRangeEnd,
             float depthRangeMin,
             float depthRangeMax,
             float rotationRate,
             float horizontalResolution,
             float horizontalFov) -> py::capsule
          {
              auto* desc = createLaserScanPublisherDescriptor(
                  topicName, frameId, nodeNamespace, queueSize, qosProfile,
                  azimuthRangeStart, azimuthRangeEnd, depthRangeMin, depthRangeMax,
                  rotationRate, horizontalResolution, horizontalFov);
              if (!desc)
              {
                  throw std::runtime_error("Failed to initialize Ros2SrtxLaserScanPublisher");
              }
              return wrapDescriptorAsCapsule(desc);
          },
          py::arg("topic_name"),
          py::arg("frame_id"),
          py::arg("node_namespace"),
          py::arg("queue_size"),
          py::arg("qos_profile") = "",
          py::arg("azimuth_range_start") = -180.0f,
          py::arg("azimuth_range_end") = 180.0f,
          py::arg("depth_range_min") = 0.0f,
          py::arg("depth_range_max") = 100.0f,
          py::arg("rotation_rate") = 20.0f,
          py::arg("horizontal_resolution") = 1.0f,
          py::arg("horizontal_fov") = 360.0f,
          R"pbdoc(
              Create a ROS 2 LaserScan publisher and return a PyCapsule
              wrapping the C-ABI callback descriptor.

              The capsule is named "SrtxFrameCallbackDescriptor" and is intended
              to be passed to omni.replicator.srtx's register_frame_callback().

              Args:
                  topic_name: ROS 2 topic name to publish on.
                  frame_id: TF frame_id for the published message header.
                  node_namespace: ROS 2 node namespace.
                  queue_size: Publisher queue depth.
                  qos_profile: JSON-encoded QoS profile (empty string for defaults).
                  azimuth_range_start: Scan start angle in degrees.
                  azimuth_range_end: Scan end angle in degrees.
                  depth_range_min: Minimum range in meters.
                  depth_range_max: Maximum range in meters.
                  rotation_rate: Scan frequency in Hz.
                  horizontal_resolution: Angular resolution in degrees.
                  horizontal_fov: Horizontal field of view in degrees.

              Returns:
                  PyCapsule containing the callback descriptor.
          )pbdoc");

    m.def(
        "fill_point_cloud_buffer",
        [](py::buffer destination, py::buffer xyz, const std::vector<std::pair<py::buffer, size_t>>& fields,
           size_t pointStep)
        {
            constexpr size_t kXyzBytes = isaacsim::ros2::nodes::kPointCloudXyzBytes;

            py::buffer_info destinationInfo = destination.request(true);
            const size_t destinationBytes = getContiguousBufferSize(destinationInfo, "destination");

            py::buffer_info xyzInfo = xyz.request();
            const size_t xyzBytes = getContiguousBufferSize(xyzInfo, "xyz");
            // Buffer-protocol format for float32 is "f", optionally prefixed with a
            // native/little-endian byte-order character (all supported platforms are LE).
            std::string xyzFormat = xyzInfo.format;
            if (!xyzFormat.empty() && (xyzFormat[0] == '@' || xyzFormat[0] == '=' || xyzFormat[0] == '<'))
            {
                xyzFormat.erase(0, 1);
            }
            if (xyzFormat != py::format_descriptor<float>::format() ||
                static_cast<size_t>(xyzInfo.itemsize) != sizeof(float))
            {
                throw std::invalid_argument("xyz buffer must contain native-endian 32-bit floats (got format '" +
                                            xyzInfo.format + "')");
            }
            if (xyzBytes % kXyzBytes != 0)
            {
                throw std::invalid_argument("xyz buffer size must be a multiple of 3 floats (x, y, z per point)");
            }
            const size_t numPoints = xyzBytes / kXyzBytes;
            if (overlapsDestination(destinationInfo.ptr, destinationBytes, xyzInfo.ptr, xyzBytes))
            {
                throw std::invalid_argument("xyz buffer must not overlap the destination buffer");
            }

            if (pointStep < kXyzBytes)
            {
                throw std::invalid_argument("point_step must be at least " + std::to_string(kXyzBytes) +
                                            " bytes (xyz occupies bytes [0, " + std::to_string(kXyzBytes) + "))");
            }
            // Division instead of numPoints * pointStep keeps the comparison overflow-safe.
            if (numPoints > 0 && pointStep > destinationBytes / numPoints)
            {
                throw std::invalid_argument("Destination buffer is smaller than num_points * point_step (" +
                                            std::to_string(destinationBytes) + " bytes for " +
                                            std::to_string(numPoints) + " points of " + std::to_string(pointStep) +
                                            " bytes)");
            }

            // buffer_info instances own their Py_buffer views; keep them alive until the copy is done.
            std::vector<py::buffer_info> fieldInfos;
            fieldInfos.reserve(fields.size());
            std::vector<std::tuple<void*, size_t, size_t>> orderedFields;
            orderedFields.reserve(fields.size());
            for (size_t fieldIndex = 0; fieldIndex < fields.size(); ++fieldIndex)
            {
                const std::string fieldName = "fields[" + std::to_string(fieldIndex) + "]";
                py::buffer_info fieldInfo = fields[fieldIndex].first.request();
                const size_t fieldBytes = getContiguousBufferSize(fieldInfo, fieldName.c_str());
                const size_t offset = fields[fieldIndex].second;
                if (offset < kXyzBytes)
                {
                    throw std::invalid_argument(fieldName + " offset (" + std::to_string(offset) +
                                                ") overlaps the xyz bytes [0, " + std::to_string(kXyzBytes) + ")");
                }
                if (numPoints == 0)
                {
                    // A zero-point cloud carries no field bytes; the per-point size (and thus the
                    // fit within point_step) is indeterminate, so only require an empty source.
                    if (fieldBytes != 0)
                    {
                        throw std::invalid_argument(fieldName + " buffer size (" + std::to_string(fieldBytes) +
                                                    " bytes) is not a multiple of the point count (0)");
                    }
                    continue;
                }
                if (fieldBytes % numPoints != 0)
                {
                    throw std::invalid_argument(fieldName + " buffer size (" + std::to_string(fieldBytes) +
                                                " bytes) is not a multiple of the point count (" +
                                                std::to_string(numPoints) + ")");
                }
                const size_t perPointBytes = fieldBytes / numPoints;
                // Subtraction instead of offset + perPointBytes keeps the comparison overflow-safe.
                if (perPointBytes > pointStep || offset > pointStep - perPointBytes)
                {
                    throw std::invalid_argument(fieldName + " does not fit within point_step (offset " +
                                                std::to_string(offset) + " + " + std::to_string(perPointBytes) +
                                                " bytes per point > " + std::to_string(pointStep) + ")");
                }
                if (overlapsDestination(destinationInfo.ptr, destinationBytes, fieldInfo.ptr, fieldBytes))
                {
                    throw std::invalid_argument(fieldName + " buffer must not overlap the destination buffer");
                }
                orderedFields.emplace_back(fieldInfo.ptr, perPointBytes, offset);
                fieldInfos.push_back(std::move(fieldInfo));
            }

            {
                py::gil_scoped_release releaseGil;
                isaacsim::ros2::nodes::fillPointCloudBufferHost(static_cast<uint8_t*>(destinationInfo.ptr),
                                                                static_cast<const float*>(xyzInfo.ptr), orderedFields,
                                                                pointStep, numPoints);
            }
            return numPoints;
        },
        py::arg("destination"), py::arg("xyz"), py::arg("fields") = std::vector<std::pair<py::buffer, size_t>>(),
        py::arg("point_step") = 3 * sizeof(float),
        R"pbdoc(
            Interleave xyz positions and per-point fields into a PointCloud2-style buffer.

            Gathers separate per-field host arrays into the packed ``point_step`` layout
            expected by ``sensor_msgs/PointCloud2.data``: each point occupies ``point_step``
            bytes, the first 12 bytes are the xyz position (3 consecutive float32), and each
            extra field is written at its byte offset within the point. The interleave runs
            in parallel and the GIL is released while it does.

            Args:
                destination: Writable, C-contiguous host buffer of at least
                    ``num_points * point_step`` bytes (e.g. a ``numpy.ndarray`` of uint8 or
                    an ``array.array('B')`` used as a ROS 2 message ``data`` field).
                xyz: C-contiguous float32 buffer of the point positions, ``num_points * 3``
                    elements (any shape). Defines the number of points.
                fields: Extra per-point fields as ``(source, offset)`` pairs. Each source is a
                    C-contiguous buffer whose total size is a per-point size times
                    ``num_points``; ``offset`` is the field's byte offset within a point and
                    must lie in ``[12, point_step - per_point_size]``.
                point_step: Bytes per point in the destination layout. Defaults to 12
                    (xyz only). Destination bytes not covered by xyz or a field (padding)
                    are left untouched.

            Returns:
                The number of points interleaved.

            Raises:
                ValueError: If a buffer is not C-contiguous, the destination is too small,
                    xyz is not native-endian float32 or not a multiple of 3 elements, a
                    field's size or offset is inconsistent with ``point_step``, or a source
                    buffer overlaps the destination.
        )pbdoc");
}
} // namespace anonymous
