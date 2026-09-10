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

#include <carb/logging/Log.h>

#include <hololink/core/hololink.hpp>
#include <hololink/emulation/emulator_utils.hpp>
#include <hololink/emulation/hsb_config.hpp>
#include <isaacsim/hsb/core/HSBSender.hpp>

#include <cstring>
#include <endian.h>
#include <mutex>
#include <unordered_map>

using namespace hololink::emulation;

namespace isaacsim::hsb::core
{

namespace detail
{

namespace
{
// Build the HSB configuration for the shared emulator. Starts from the Leopard Eagle defaults
// shipped with hsb_emulator and bumps `hsb_ip_version` so hololink data-channel consumers
// (which enforce `MINIMUM_HSB_IP_VERSION` in hololink's data_channel.cpp) accept the emulated
// board.
//
// The version is encoded as YYMM (e.g. 0x2602 = Feb 2026). The shipped emulator's default
// (0x2508 / Aug 2025) predates the current floor in hololink and would be rejected; bump
// safely past it.
HSBConfiguration buildSharedConfig()
{
    HSBConfiguration config = HSB_LEOPARD_EAGLE_CONFIG;
    config.hsb_ip_version = 0x2700; // > current MINIMUM_HSB_IP_VERSION (0x2602) in hololink
    return config;
}
} // namespace

SharedHSBEmulator::SharedHSBEmulator(const std::string& ip) : emulator(buildSharedConfig()), ipAddress(ip)
{
}

SharedHSBEmulator::~SharedHSBEmulator()
{
    // Stop the emulator before its members are destroyed. HSBEmulator::stop() is documented
    // as idempotent, so calling here is safe even if no DataPlane ever attached. catch(...)
    // (not catch(std::exception&)) — a destructor must never let anything escape, even a
    // non-std::exception throw.
    try
    {
        emulator.stop();
    }
    catch (const std::exception& e)
    {
        CARB_LOG_ERROR("[HSB Bridge] SharedHSBEmulator::~ ip=%s stop() threw: %s", ipAddress.c_str(), e.what());
    }
    catch (...)
    {
        CARB_LOG_ERROR("[HSB Bridge] SharedHSBEmulator::~ ip=%s stop() threw non-std exception", ipAddress.c_str());
    }
}

namespace
{

// Process-wide registry of live shared emulators, keyed on IP. weak_ptr lets the emulator be
// destroyed automatically when the last HSBSender for an IP drops its shared_ptr; the next
// connect() on the same IP then sees an expired weak_ptr and constructs a fresh emulator.
std::mutex g_registryMutex;
std::unordered_map<std::string, std::weak_ptr<SharedHSBEmulator>> g_registry;

} // namespace

} // namespace detail

std::shared_ptr<detail::SharedHSBEmulator> HSBSender::acquireSharedEmulator(const std::string& ipAddress)
{
    std::lock_guard<std::mutex> lock(detail::g_registryMutex);

    auto it = detail::g_registry.find(ipAddress);
    if (it != detail::g_registry.end())
    {
        if (auto existing = it->second.lock())
        {
            // Another HSBSender on this IP is alive; reuse its emulator.
            return existing;
        }
        // weak_ptr expired (last sender for this IP was destroyed); fall through to create a
        // fresh one and overwrite the stale entry.
    }

    auto shared = std::make_shared<detail::SharedHSBEmulator>(ipAddress);
    // Do NOT call emulator.start() here. The hololink I2CController refuses peripheral
    // attachments while the emulator is running, so each HSBSender::connect() needs to attach
    // its Vb1940Emulator *before* start() is reached. We let the senders drive the
    // stop-attach-start cycle (see HSBSender::connect() below).
    detail::g_registry[ipAddress] = shared;
    return shared;
}

bool HSBSender::hasSharedEmulatorWith(const HSBSender& other) const
{
    return m_shared && m_shared == other.m_shared;
}

HSBSender::HSBSender(const std::string& ipAddress, uint8_t dataPlaneId, uint8_t sensorId, const std::string& dataPlaneType)
    : m_ipAddress(ipAddress),
      m_dataPlaneId(dataPlaneId),
      m_sensorId(sensorId),
      m_dataPlaneType(dataPlaneType.empty() ? "linux" : dataPlaneType),
      m_connected(false),
      m_streamingWarningLogged(false)
{
}

HSBSender::~HSBSender()
{
    // Destructors must never let anything escape. disconnect() catches std::exception, but a
    // non-std throw from std::mutex::lock() / smart-ptr resets would still propagate and call
    // std::terminate(). Mirror the catch(...) pattern used in SharedHSBEmulator::~.
    try
    {
        if (m_connected)
        {
            disconnect();
        }
    }
    catch (const std::exception& e)
    {
        CARB_LOG_ERROR("[HSB Bridge] ~HSBSender: disconnect() threw: %s", e.what());
    }
    catch (...)
    {
        CARB_LOG_ERROR("[HSB Bridge] ~HSBSender: disconnect() threw non-std exception");
    }
}

bool HSBSender::connect()
{
    CARB_LOG_INFO("[HSB Bridge] connect() ip=%s dataPlaneId=%u sensorId=%u dataPlaneType='%s'", m_ipAddress.c_str(),
                  (unsigned)m_dataPlaneId, (unsigned)m_sensorId, m_dataPlaneType.c_str());

    if (m_connected)
    {
        return true;
    }

    try
    {
        // Parse IP address
        IPAddress source_ip = IPAddress_from_string(m_ipAddress);

        // Get-or-create the shared emulator for this IP. The first caller starts the emulator;
        // subsequent senders targeting the same IP reuse the same virtual Hololink board so the
        // hololink receiver sees a single board with multiple sensors —
        // matching real stereo VB1940 hardware on Leopard Eagle.
        m_shared = acquireSharedEmulator(m_ipAddress);

        // Serialize the entire connect lifecycle (slot reservation → dataplane create →
        // is_running/stop/attach/start) across all senders sharing this emulator. Two senders
        // connecting concurrently on the same IP would otherwise interleave stop/start and
        // could leave the emulator half-configured (e.g. sender A calls stop() just after
        // sender B called start()).
        std::lock_guard<std::mutex> connectLock(m_shared->connectMutex);

        // Claim our sensor slot on the shared emulator. Two senders on the same IP must use
        // different sensor IDs (real Leopard Eagle: sensor 0 and 1 sit at different I²C bus
        // offsets — see CAM_I2C_BUS + m_sensorId below). Reject duplicates so a misconfigured
        // scene fails early instead of silently producing one stream on top of another.
        if (!m_shared->takenSensorIds.insert(m_sensorId).second)
        {
            CARB_LOG_ERROR("[HSB Bridge] sensorId %u already in use on ip %s; reject connect", (unsigned)m_sensorId,
                           m_ipAddress.c_str());
            m_shared.reset();
            return false;
        }

        // Create data plane (Linux/RoCEv2 or COE IEEE 1722B), bound to the shared emulator.
        if (m_dataPlaneType == "coe")
        {
            m_dataPlane = std::make_unique<COEDataPlane>(m_shared->emulator, source_ip, m_dataPlaneId, m_sensorId);
        }
        else
        {
            m_dataPlane = std::make_unique<LinuxDataPlane>(m_shared->emulator, source_ip, m_dataPlaneId, m_sensorId);
        }

        // The I2CController refuses peripheral attachments while the emulator is running.
        // For stereo: the second HSBSender to connect will find the emulator already started
        // by the first; stop it transiently so we can attach our Vb1940Emulator, then start
        // again. HSBEmulator::start()/stop() are documented idempotent so this is safe.
        if (m_shared->emulator.is_running())
        {
            m_shared->emulator.stop();
        }

        // Create and attach Vb1940Emulator for linux_vb1940_player compatibility.
        // On Leopard Eagle, the i2c bus address is the sensor_id offset from CAM_I2C_BUS — so
        // two senders sharing one emulator end up at distinct peripheral addresses.
        m_vb1940 = std::make_unique<sensors::Vb1940Emulator>();
        m_vb1940->attach_to_i2c(m_shared->emulator.get_i2c(hololink::I2C_CTRL), hololink::CAM_I2C_BUS + m_sensorId);

        // Attach the rig calibration EEPROM at peripheral address 0x51 on the same I²C bus.
        // Real hardware has this as a separate IC next to the VB1940; without it, the live
        // receiver's read_calibration_from_eeprom() returns zero intrinsics/extrinsics, and
        // downstream cuVSLAM rejects with "Focal length must be > 0" / "identical poses".
        // Pre-populate from any setCalibration() call made before connect(); otherwise the
        // EEPROM is zero-initialized and setCalibration() can be called later to update.
        // Program the Vb1940Emulator's rig calibration EEPROM region (peripheral 0x51).
        // Vb1940Emulator already claims address 0x51 in its attach_to_i2c() — the
        // upstream stub no-ops EEPROM reads, but our local fork serves real bytes
        // from a per-sensor backing store. See
        // third_party/hololink/vb1940-serve-eeprom-region.patch.
        if (m_hasCalibration)
        {
            std::array<uint8_t, 256> eepromBytes{};
            encodeCalibrationBytes(m_pendingCalibration, eepromBytes);
            m_vb1940->set_eeprom_data(eepromBytes.data(), eepromBytes.size());
        }

        m_shared->emulator.start();

        CARB_LOG_INFO("[HSB Bridge] Connected successfully (dataPlane: %s) ip=%s sensorId=%u", m_dataPlaneType.c_str(),
                      m_ipAddress.c_str(), (unsigned)m_sensorId);
        m_connected = true;
        return true;
    }
    catch (const std::exception& e)
    {
        CARB_LOG_ERROR("[HSB Bridge] Failed to connect: %s", e.what());
        // Roll back partial state so a retry starts clean. Same ordering constraint as
        // disconnect(): detach the Vb1940Emulator BEFORE freeing the sensor slot, and hold
        // connectMutex across both steps — otherwise a concurrent connect() could reclaim the
        // slot and attach a new peripheral at the same I²C address while our (possibly-
        // attached) one is still alive.
        if (m_shared)
        {
            std::lock_guard<std::mutex> connectLock(m_shared->connectMutex);
            m_vb1940.reset();
            m_dataPlane.reset();
            m_shared->takenSensorIds.erase(m_sensorId);

            // If we transiently stopped the emulator to attach our (now-aborted) peripheral
            // and there are other senders still attached to this shared emulator, restart it
            // so their streams don't silently freeze after our failed connect. Guard the
            // restart in its own try/catch — a nested throw here would leak out of connect()
            // as a different exception than the original.
            if (!m_shared->takenSensorIds.empty() && !m_shared->emulator.is_running())
            {
                try
                {
                    m_shared->emulator.start();
                }
                catch (const std::exception& restartEx)
                {
                    CARB_LOG_ERROR("[HSB Bridge] rollback restart of shared emulator failed: %s", restartEx.what());
                }
            }
        }
        m_shared.reset();
        m_connected = false;
        return false;
    }
}

bool HSBSender::disconnect()
{
    if (!m_connected)
    {
        return true;
    }

    try
    {
        // Order matters: destroy the Vb1940 and DataPlane BEFORE releasing the sensor slot,
        // and hold connectMutex across both steps. If we released the slot first, a concurrent
        // connect() on the same (ipAddress, sensorId) could pass the duplicate check and call
        // attach_to_i2c() at the same peripheral address while our Vb1940Emulator is still
        // attached — double-registering the I²C peripheral. Holding connectMutex through the
        // reset+erase makes the detach-then-free-slot pair atomic w.r.t. concurrent connects.
        if (m_shared)
        {
            std::lock_guard<std::mutex> connectLock(m_shared->connectMutex);
            m_vb1940.reset(); // detaches from the shared emulator's I²C bus
            m_dataPlane.reset(); // detaches from the shared emulator's data plane registry
            m_shared->takenSensorIds.erase(m_sensorId);
        }
        // Drop the shared_ptr last (outside the lock): its release may or may not destroy the
        // emulator — depends on whether other HSBSenders on the same IP still hold references.
        m_shared.reset();
        m_connected = false;
        return true;
    }
    catch (const std::exception& e)
    {
        CARB_LOG_ERROR("[HSB Bridge] Failed to disconnect: %s", e.what());
        return false;
    }
}

bool HSBSender::isStreaming() const
{
    if (m_vb1940)
    {
        return m_vb1940->is_streaming();
    }
    return false;
}

namespace
{

// Write a 64-bit IEEE-754 double into `dst` in big-endian byte order, matching what
// hololink's NativeVb1940Sensor::bytes_to_double() decodes from the EEPROM bytes.
// htobe64 is a POSIX (BSD-derived) conversion in <endian.h> — no-op on big-endian
// hosts, byte-swap on little-endian. Isaac Sim is Linux-only so glibc availability
// is guaranteed; on the (unsupported) big-endian hypothetical, htobe64 correctly
// leaves `raw` untouched.
void writeBigEndianDouble(uint8_t* dst, double value)
{
    uint64_t raw;
    std::memcpy(&raw, &value, sizeof(raw));
    raw = htobe64(raw);
    std::memcpy(dst, &raw, sizeof(raw));
}

} // namespace

void HSBSender::encodeCalibrationBytes(const CameraCalibration& calib, std::array<uint8_t, 256>& out)
{
    // Layout consumed by NativeVb1940Sensor::read_calibration_from_eeprom() — 30 doubles
    // at 8 bytes each, packed contiguously starting at byte 0. See
    // src/hololink/sensors/camera/vb1940/native_vb1940_sensor.cpp:795-851 for the decoder.
    static_assert(sizeof(double) == 8, "expecting 64-bit double");
    out.fill(0);

    constexpr size_t kStep = 8;
    size_t off = 0;
    const auto emit = [&](double v)
    {
        writeBigEndianDouble(out.data() + off, v);
        off += kStep;
    };

    // Left intrinsics (fx, fy, cx, cy), then left distortion (k1..k6, p1, p2 — note the
    // upstream layout interleaves: k1, k2, p1, p2, k3, k4, k5, k6).
    emit(calib.intrinsics[0][0]);
    emit(calib.intrinsics[0][1]);
    emit(calib.intrinsics[0][2]);
    emit(calib.intrinsics[0][3]);
    for (double d : calib.distortion[0])
    {
        emit(d);
    }
    // Right intrinsics + distortion.
    emit(calib.intrinsics[1][0]);
    emit(calib.intrinsics[1][1]);
    emit(calib.intrinsics[1][2]);
    emit(calib.intrinsics[1][3]);
    for (double d : calib.distortion[1])
    {
        emit(d);
    }
    // Rig rotation (Rx, Ry, Rz) then translation (Tx, Ty, Tz).
    emit(calib.R[0]);
    emit(calib.R[1]);
    emit(calib.R[2]);
    emit(calib.T[0]);
    emit(calib.T[1]);
    emit(calib.T[2]);
}

void HSBSender::setCalibration(const CameraCalibration& calib)
{
    // Stage for next connect() and apply immediately if already connected.
    m_pendingCalibration = calib;
    m_hasCalibration = true;
    if (m_vb1940)
    {
        std::array<uint8_t, 256> bytes{};
        encodeCalibrationBytes(calib, bytes);
        m_vb1940->set_eeprom_data(bytes.data(), bytes.size());
    }
}

bool HSBSender::isConnected() const
{
    return m_connected && m_dataPlane != nullptr;
}

bool HSBSender::send(const DLTensor& tensor)
{
    if (!isConnected())
    {
        CARB_LOG_ERROR("[HSB Bridge] Failed to send: not connected");
        return false;
    }

    // Check if host has configured streaming - if not, data won't actually be sent
    // (DataPlane::send returns 0 when no receiver has connected). Latch is per-instance
    // so stereo senders don't mask each other's warnings and there's no cross-thread race
    // on a shared flag.
    if (!isStreaming() && !m_streamingWarningLogged)
    {
        CARB_LOG_ERROR("[HSB Bridge] Vb1940 not in streaming mode - waiting for host to connect (ip=%s sensorId=%u)",
                       m_ipAddress.c_str(), (unsigned)m_sensorId);
        m_streamingWarningLogged = true;
    }
    else if (isStreaming() && m_streamingWarningLogged)
    {
        CARB_LOG_INFO("[HSB Bridge] Vb1940 now streaming - host connected (ip=%s sensorId=%u)", m_ipAddress.c_str(),
                      (unsigned)m_sensorId);
        m_streamingWarningLogged = false;
    }

    int64_t bytes_sent = m_dataPlane->send(tensor);
    return bytes_sent >= 0;
}

} // namespace isaacsim::hsb::core
