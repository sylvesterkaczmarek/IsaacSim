// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// clang-format off
#include <pch/UsdPCH.hpp>
// clang-format on

#include <isaacsim/core/includes/ScopedCudaDevice.hpp>
#include <isaacsim/zmq/nodes/ZmqPublishNode.hpp>

#include <OgnZMQPublishImageDatabase.h>
#include <cuda_runtime_api.h>
#include <image.pb.h>

// ScopedCudaDevice.hpp defines CUDA_CHECK as a log-only macro; redefine here to also return false.
#undef CUDA_CHECK
#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            CARB_LOG_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err));                    \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

// Per-encoding pixel layout used to size copies and describe the IPC array.
struct ImageLayout
{
    const char* dtype; // warp/numpy dtype token the consumer reconstructs with
    uint32_t channels; // 4 for rgba8, 3 for rgb8, 1 for 32FC1
    size_t elemSize; // bytes per channel element
};

static bool layoutForEncoding(const std::string& encoding, ImageLayout& out)
{
    if (encoding == "rgba8")
    {
        out = { "uint8", 4, 1 };
    }
    else if (encoding == "rgb8")
    {
        out = { "uint8", 3, 1 };
    }
    else if (encoding == "32FC1")
    {
        out = { "float32", 1, sizeof(float) };
    }
    else
    {
        return false;
    }
    return true;
}

/**
 * @brief Publishes a camera image (color or depth) over ZMQ as an Image protobuf.
 *
 * Two pixel paths, selected by ``useIpc``:
 *  - useIpc && GPU source (default): zero-copy CUDA IPC. The render output is
 *    D2D-copied into a persistent, IPC-exportable cudaMalloc buffer; the buffer's
 *    cudaIpcMemHandle_t + an interprocess event handle + a frame id ride in the
 *    proto. Same host only. (The render output is a cudaArray / pooled buffer that
 *    is not itself IPC-exportable, so the copy into our own linear buffer is required.)
 *  - otherwise: raw bytes (CPU source, or useIpc disabled for cross-host streaming).
 */
class OgnZMQPublishImage : public isaacsim::zmq::nodes::ZmqPublishNode
{
public:
    ~OgnZMQPublishImage()
    {
        // Explicitly call overridden reset() before the vtable reverts to ZmqPublishNode
        // during base class destruction. Without this, the CUDA stream and IPC
        // resources would leak because ~ZmqPublishNode() only calls ZmqPublishNode::reset().
        OgnZMQPublishImage::reset();
    }

    static void releaseInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state = OgnZMQPublishImageDatabase::sPerInstanceState<OgnZMQPublishImage>(nodeObj, instanceId);
        state.reset();
    }

    virtual void reset() override
    {
        if (m_streamCreated)
        {
            isaacsim::core::includes::ScopedDevice scopedDev(m_streamDevice);
            destroyIpcResources();
            cudaStreamDestroy(m_stream);
            m_stream = nullptr;
            m_streamDevice = -1;
            m_streamCreated = false;
        }
        ZmqPublishNode::reset();
    }

    static bool compute(OgnZMQPublishImageDatabase& db)
    {
        auto& state = db.template perInstanceState<OgnZMQPublishImage>();
        return state.computeImpl(db);
    }

private:
    bool computeImpl(OgnZMQPublishImageDatabase& db)
    {
        const std::string ip = db.inputs.ip();
        const uint16_t port = static_cast<uint16_t>(db.inputs.port());
        const std::string topicName = db.inputs.topicName();
        if (!ensureSocketReady(db, ip, port))
        {
            return false;
        }

        const uint32_t width = db.inputs.width();
        const uint32_t height = db.inputs.height();
        if (width == 0 || height == 0)
        {
            db.logWarning("OgnZMQPublishImage: invalid width/height (%u x %u)", width, height);
            return false;
        }

        if (db.inputs.dataPtr() == 0 && db.inputs.data.size() == 0)
        {
            db.logError("OgnZMQPublishImage: no image data (dataPtr==0 and data array is empty)");
            return false;
        }

        const std::string encoding = db.tokenToString(db.inputs.encoding());
        ImageLayout layout;
        if (!layoutForEncoding(encoding, layout))
        {
            db.logError("OgnZMQPublishImage: unsupported encoding '%s'", encoding.c_str());
            return false;
        }

        const int cudaDeviceIndex = db.inputs.cudaDeviceIndex();
        const size_t bufferSize = static_cast<size_t>(db.inputs.bufferSize());
        const size_t rowBytes = static_cast<size_t>(width) * layout.channels * layout.elemSize;

        // Resolve the byte size of one frame.
        size_t dataSize = bufferSize;
        if (dataSize == 0)
        {
            if (cudaDeviceIndex != -1)
            {
                // GPU texture (bufferSize unknown): infer from the encoding layout.
                dataSize = rowBytes * height;
            }
            else if (db.inputs.data.size() > 0)
            {
                dataSize = db.inputs.data.size();
            }
        }
        if (dataSize == 0)
        {
            db.logError("OgnZMQPublishImage: cannot determine image data size");
            return false;
        }

        isaacsim::zmq::Image proto;
        proto.set_timestamp(db.inputs.timeStamp());
        proto.set_width(width);
        proto.set_height(height);
        proto.set_encoding(encoding);

        const bool useIpc = db.inputs.useIpc() && cudaDeviceIndex != -1;
        if (useIpc)
        {
            isaacsim::core::includes::ScopedDevice scopedDev(cudaDeviceIndex);
            if (!ensureCudaStream(cudaDeviceIndex) || !ensureIpcResources(dataSize))
            {
                return false;
            }
            // D2D: render output -> our IPC-exportable buffer (no host round trip).
            if (!copyGpuImage(db, m_ipcBuf, bufferSize, height, rowBytes, cudaMemcpyDeviceToDevice))
            {
                return false;
            }
            // Order the consumer's read after this write; do NOT block the producer.
            CUDA_CHECK(cudaEventRecord(m_ipcEvent, m_stream));

            isaacsim::zmq::GpuIpcImage* gpu = proto.mutable_gpu();
            isaacsim::zmq::GpuIpcArray* ipc = gpu->mutable_array();
            ipc->set_mem_handle(&m_memHandle, sizeof(m_memHandle));
            ipc->set_dtype(layout.dtype);
            ipc->add_shape(height);
            ipc->add_shape(width);
            if (layout.channels > 1)
            {
                ipc->add_shape(layout.channels);
            }
            gpu->set_ipc_event_handle(&m_evtHandle, sizeof(m_evtHandle));
            proto.set_frame_id(++m_frameId);
        }
        else
        {
            // Bytes path: stage pixels on the host and embed them in the proto.
            std::vector<uint8_t> hostData(dataSize);
            if (cudaDeviceIndex == -1)
            {
                if (db.inputs.dataPtr() != 0 && bufferSize > 0)
                {
                    std::memcpy(hostData.data(), reinterpret_cast<const void*>(db.inputs.dataPtr()), dataSize);
                }
                else if (db.inputs.data.size() > 0)
                {
                    const auto& arr = db.inputs.data.cpu();
                    std::memcpy(hostData.data(), arr.data(), dataSize);
                }
                else
                {
                    db.logError("OgnZMQPublishImage: no valid CPU data source");
                    return false;
                }
            }
            else
            {
                isaacsim::core::includes::ScopedDevice scopedDev(cudaDeviceIndex);
                if (!ensureCudaStream(cudaDeviceIndex))
                {
                    return false;
                }
                if (!copyGpuImage(db, hostData.data(), bufferSize, height, rowBytes, cudaMemcpyDeviceToHost))
                {
                    return false;
                }
                CUDA_CHECK(cudaStreamSynchronize(m_stream));
            }
            proto.set_data(hostData.data(), hostData.size());
        }

        if (!publishProto(db, topicName, proto))
        {
            return false;
        }

        db.outputs.execOut() = kExecutionAttributeStateEnabled;
        return true;
    }

    // Copy the image source into dst. GPU buffer (bufferSize>0) → linear copy;
    // GPU texture (bufferSize==0) → 2D copy from the mipmapped array's level 0.
    bool copyGpuImage(OgnZMQPublishImageDatabase& db,
                      void* dst,
                      size_t bufferSize,
                      uint32_t height,
                      size_t rowBytes,
                      cudaMemcpyKind kind)
    {
        if (bufferSize == 0)
        {
            cudaArray_t levelArray = nullptr;
            CUDA_CHECK(cudaGetMipmappedArrayLevel(
                &levelArray, reinterpret_cast<cudaMipmappedArray_t>(db.inputs.dataPtr()), 0));
            CUDA_CHECK(cudaMemcpy2DFromArrayAsync(dst, rowBytes, levelArray, 0, 0, rowBytes, height, kind, m_stream));
        }
        else
        {
            CUDA_CHECK(cudaMemcpyAsync(dst, reinterpret_cast<void*>(db.inputs.dataPtr()), bufferSize, kind, m_stream));
        }
        return true;
    }

    bool ensureCudaStream(int deviceIndex)
    {
        if (m_streamDevice != deviceIndex && m_streamCreated)
        {
            isaacsim::core::includes::ScopedDevice oldDev(m_streamDevice);
            destroyIpcResources();
            cudaStreamDestroy(m_stream);
            m_stream = nullptr;
            m_streamCreated = false;
            m_streamDevice = -1;
        }
        if (!m_streamCreated)
        {
            cudaError_t err = cudaStreamCreate(&m_stream);
            if (err != cudaSuccess)
            {
                CARB_LOG_ERROR("OgnZMQPublishImage: cudaStreamCreate failed: %s", cudaGetErrorString(err));
                return false;
            }
            m_streamCreated = true;
            m_streamDevice = deviceIndex;
        }
        return true;
    }

    // Allocate (or resize) the IPC-exportable buffer and lazily create the
    // interprocess event. mem/event handle bytes are stable for the buffer's
    // lifetime — the consumer treats the event handle as a producer epoch.
    // Caller must hold ScopedDevice(m_streamDevice) and a live stream.
    bool ensureIpcResources(size_t dataSize)
    {
        if (m_ipcBuf != nullptr && m_ipcBufSize != dataSize)
        {
            // Drain any in-flight async copy into the old buffer first — cudaFree
            // does not synchronize non-default streams.
            CUDA_CHECK(cudaStreamSynchronize(m_stream));
            CUDA_CHECK(cudaFree(m_ipcBuf));
            m_ipcBuf = nullptr;
            m_ipcBufSize = 0;
            m_memValid = false;
        }
        if (m_ipcBuf == nullptr)
        {
            CUDA_CHECK(cudaMalloc(&m_ipcBuf, dataSize));
            m_ipcBufSize = dataSize;
        }
        if (!m_memValid)
        {
            CUDA_CHECK(cudaIpcGetMemHandle(&m_memHandle, m_ipcBuf));
            m_memValid = true;
        }
        if (!m_evtValid)
        {
            CUDA_CHECK(cudaEventCreateWithFlags(&m_ipcEvent, cudaEventInterprocess | cudaEventDisableTiming));
            CUDA_CHECK(cudaIpcGetEventHandle(&m_evtHandle, m_ipcEvent));
            m_evtValid = true;
        }
        return true;
    }

    // Free IPC buffer + event. Caller must hold ScopedDevice(m_streamDevice).
    void destroyIpcResources()
    {
        if (m_stream)
        {
            // Best effort: drain in-flight copies before freeing their target buffer.
            cudaStreamSynchronize(m_stream);
        }
        if (m_ipcEvent)
        {
            cudaEventDestroy(m_ipcEvent);
            m_ipcEvent = nullptr;
        }
        if (m_ipcBuf)
        {
            cudaFree(m_ipcBuf);
            m_ipcBuf = nullptr;
        }
        m_ipcBufSize = 0;
        m_memValid = false;
        m_evtValid = false;
    }

    cudaStream_t m_stream = nullptr;
    int m_streamDevice = -1;
    bool m_streamCreated = false;

    // CUDA-IPC state (zero-copy path).
    //
    // NOTE: single export buffer, last-frame-wins. The per-frame event lets the
    // consumer wait for THIS frame's write to complete (read-after-write), but does
    // NOT stop the next frame's D2D copy from overwriting the buffer while a slow
    // consumer is still reading it (write-after-read). This mirrors the upstream
    // IsaacSimZMQ single-scratch design and is acceptable for a ~60Hz, HWM=1 viewer
    // (worst case: an occasional torn frame). For a hard guarantee, cycle a small
    // ring of export buffers keyed by frame_id instead of reusing one.
    void* m_ipcBuf = nullptr; //!< IPC-exportable cudaMalloc buffer (D2D target)
    size_t m_ipcBufSize = 0;
    cudaIpcMemHandle_t m_memHandle{};
    bool m_memValid = false;
    cudaEvent_t m_ipcEvent = nullptr;
    cudaIpcEventHandle_t m_evtHandle{};
    bool m_evtValid = false;
    uint64_t m_frameId = 0;
};

REGISTER_OGN_NODE()
