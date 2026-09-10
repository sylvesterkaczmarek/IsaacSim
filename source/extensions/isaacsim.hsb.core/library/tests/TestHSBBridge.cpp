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

#include <doctest/doctest.h>
#include <isaacsim/hsb/core/HSBSender.hpp>

#include <cstdint>
#include <cstring>

using namespace isaacsim::hsb::core;

namespace
{
// Mirror of the private writeBigEndianDouble helper used inside encodeCalibrationBytes.
// Reads 8 consecutive bytes as an IEEE-754 big-endian double so tests can verify what
// the encoder wrote at each byte offset.
double readBigEndianDouble(const uint8_t* p)
{
    uint64_t bits = 0;
    for (int i = 0; i < 8; ++i)
    {
        bits = (bits << 8) | static_cast<uint64_t>(p[i]);
    }
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}
} // namespace

TEST_SUITE("HSBSender")
{
    TEST_CASE("HSBSender - construction")
    {
        // Test that HSBSender can be constructed
        HSBSender sender("10.0.0.1", 1, 2);

        // Initial state should be disconnected
        CHECK(!sender.isConnected());
    }

    TEST_CASE("HSBSender - construction with COE")
    {
        HSBSender sender("127.0.0.1", 0, 0, "coe");
        CHECK(!sender.isConnected());
    }

    TEST_CASE("HSBSender - connect and disconnect lifecycle")
    {
        HSBSender sender("127.0.0.1", 0, 0);

        // Initial state
        CHECK(!sender.isConnected());

        // Connect
        bool connected = sender.connect();
        // Note: May fail if HSB emulator setup fails, which is OK for unit test
        if (connected)
        {
            CHECK(sender.isConnected());

            // Disconnect
            CHECK(sender.disconnect());
            CHECK(!sender.isConnected());

            // Multiple disconnect calls should be safe
            CHECK(sender.disconnect());
            CHECK(!sender.isConnected());
        }
    }

    TEST_CASE("HSBSender - send without connection fails")
    {
        HSBSender sender("127.0.0.1", 0, 0);

        // Try to send without connecting
        int64_t shape[3] = { 10, 10, 3 };
        std::vector<uint8_t> data(10 * 10 * 3, 0);

        DLTensor tensor = { .data = data.data(),
                            .device = { .device_type = DLDeviceType::kDLCPU, .device_id = 0 },
                            .ndim = 3,
                            .dtype = DLDataType{ .code = DLDataTypeCode::kDLUInt, .bits = 8, .lanes = 1 },
                            .shape = shape,
                            .strides = nullptr,
                            .byte_offset = 0 };

        // Should fail when not connected
        CHECK(!sender.send(tensor));
    }

    TEST_CASE("HSBSender - multiple connect calls are safe")
    {
        HSBSender sender("127.0.0.1", 0, 0);

        bool first_connect = sender.connect();
        if (first_connect)
        {
            CHECK(sender.isConnected());

            // Second connect should succeed (idempotent)
            CHECK(sender.connect());
            CHECK(sender.isConnected());

            // Cleanup
            sender.disconnect();
        }
    }

    TEST_CASE("HSBSender - COE connect and disconnect best-effort")
    {
        HSBSender sender("127.0.0.1", 0, 0, "coe");
        CHECK(!sender.isConnected());
        bool connected = sender.connect();
        if (connected)
        {
            CHECK(sender.isConnected());
            CHECK(sender.disconnect());
            CHECK(!sender.isConnected());
        }
    }

    TEST_CASE("HSBSender - stereo: two senders share one emulator on same IP")
    {
        // Real Leopard Eagle: one Hololink board, two VB1940 imagers at sensor_id 0 and 1.
        // Equivalent in sim: two HSBSender instances on the same IP with different sensor_ids.
        HSBSender left("127.0.0.1", 0, 0);
        HSBSender right("127.0.0.1", 1, 1);

        const bool leftOk = left.connect();
        if (leftOk)
        {
            CHECK(left.isConnected());

            // Right must succeed without spinning up a second emulator on the same address.
            CHECK(right.connect());
            CHECK(right.isConnected());

            // Directly pin the sharing invariant: both senders must resolve to the SAME
            // underlying HSBEmulator in the process-wide registry. Coexistence alone doesn't
            // prove sharing (SO_REUSEADDR on loopback could let both bind separately).
            CHECK(left.hasSharedEmulatorWith(right));

            // Either order of disconnect must be safe; the shared emulator stays up while
            // any sender still references it.
            CHECK(left.disconnect());
            CHECK(!left.isConnected());
            CHECK(right.isConnected());

            // After left disconnects, the shared_ptr is released on left's side, so the
            // identity check no longer holds — right still holds the last live reference.
            CHECK_FALSE(left.hasSharedEmulatorWith(right));

            CHECK(right.disconnect());
            CHECK(!right.isConnected());
        }
    }

    TEST_CASE("HSBSender - duplicate sensorId on same IP is rejected")
    {
        HSBSender first("127.0.0.1", 0, 0);
        HSBSender duplicate("127.0.0.1", 1, 0); // same IP, same sensorId

        if (first.connect())
        {
            // Duplicate sensor_id on the same emulator would silently double-attach at the
            // same I²C peripheral address — the live receiver would then see one of them
            // ignored. Refuse instead.
            CHECK(!duplicate.connect());
            CHECK(!duplicate.isConnected());
            CHECK(first.isConnected());

            first.disconnect();
        }
    }

    TEST_CASE("HSBSender - emulator slot is freed after disconnect")
    {
        // After a sender disconnects, the next sender on the same (IP, sensorId) must be able
        // to claim that slot — this validates that connect()'s rollback and disconnect()'s
        // cleanup both release the takenSensorIds entry.
        HSBSender first("127.0.0.1", 0, 0);
        if (first.connect())
        {
            CHECK(first.disconnect());

            HSBSender second("127.0.0.1", 0, 0);
            CHECK(second.connect());
            CHECK(second.isConnected());
            second.disconnect();
        }
    }
}

TEST_SUITE("Integration")
{
    TEST_CASE("HSBSender - send DLTensor after connect")
    {
        HSBSender sender("127.0.0.1", 0, 0);

        bool connected = sender.connect();
        if (connected)
        {
            // Create a simple test image
            int64_t shape[3] = { 480, 640, 3 };
            std::vector<uint8_t> data(480 * 640 * 3, 128); // Gray image

            DLTensor tensor = { .data = data.data(),
                                .device = { .device_type = DLDeviceType::kDLCPU, .device_id = 0 },
                                .ndim = 3,
                                .dtype = DLDataType{ .code = DLDataTypeCode::kDLUInt, .bits = 8, .lanes = 1 },
                                .shape = shape,
                                .strides = nullptr,
                                .byte_offset = 0 };

            // Should succeed (or fail gracefully if HSB emulator isn't fully configured)
            sender.send(tensor);
            // We don't assert success here as it depends on HSB runtime state
            // The important part is that it doesn't crash

            sender.disconnect();
        }
    }
}

// Encoding correctness tests for the calibration EEPROM layout served at I²C 0x51.
// The decoder that consumes this buffer lives in hololink:
// src/hololink/sensors/camera/vb1940/native_vb1940_sensor.cpp — any layout change here
// must stay in sync with that decoder or cuVSLAM silently misreads calibration.
//
// Buffer layout (30 IEEE-754 big-endian doubles = 240 bytes; bytes 240..255 zero-padded):
//   0    L_fx   |   8   L_fy   |  16   L_cx   |  24   L_cy
//   32   L_k1   |  40   L_k2   |  48   L_p1   |  56   L_p2
//   64   L_k3   |  72   L_k4   |  80   L_k5   |  88   L_k6
//   96   R_fx   | 104   R_fy   | 112   R_cx   | 120   R_cy
//   128  R_k1   | 136   R_k2   | 144   R_p1   | 152   R_p2
//   160  R_k3   | 168   R_k4   | 176   R_k5   | 184   R_k6
//   192  Rx     | 200   Ry     | 208   Rz     | 216   Tx     | 224   Ty     | 232   Tz
TEST_SUITE("CameraCalibration encoding")
{
    TEST_CASE("encodeCalibrationBytes - zero calibration produces zeroed buffer")
    {
        CameraCalibration calib{}; // all fields default-constructed to 0.0
        std::array<uint8_t, 256> out{};
        out.fill(0xFF); // sentinel — proves encode overwrites the whole buffer
        HSBSender::encodeCalibrationBytes(calib, out);
        for (size_t i = 0; i < out.size(); ++i)
        {
            CHECK(out[i] == 0);
        }
    }

    TEST_CASE("encodeCalibrationBytes - big-endian byte order")
    {
        // 3.0 in IEEE-754 = 0x4008000000000000. In big-endian the MSB lands at byte 0.
        CameraCalibration calib{};
        calib.intrinsics[0][0] = 3.0;
        std::array<uint8_t, 256> out{};
        HSBSender::encodeCalibrationBytes(calib, out);
        CHECK(out[0] == 0x40);
        CHECK(out[1] == 0x08);
        CHECK(out[2] == 0x00);
        CHECK(out[3] == 0x00);
        CHECK(out[4] == 0x00);
        CHECK(out[5] == 0x00);
        CHECK(out[6] == 0x00);
        CHECK(out[7] == 0x00);
    }

    TEST_CASE("encodeCalibrationBytes - documented offset layout")
    {
        // Program every field with a distinct value so decoding by offset unambiguously
        // pins which field lives at which byte offset. If someone reorders emit() calls
        // in encodeCalibrationBytes, one or more of these assertions fires immediately.
        CameraCalibration calib{};
        calib.intrinsics[0] = { 1.0, 2.0, 3.0, 4.0 };
        for (size_t i = 0; i < 8; ++i)
        {
            calib.distortion[0][i] = 10.0 + static_cast<double>(i);
        }
        calib.intrinsics[1] = { 20.0, 21.0, 22.0, 23.0 };
        for (size_t i = 0; i < 8; ++i)
        {
            calib.distortion[1][i] = 30.0 + static_cast<double>(i);
        }
        calib.R = { 40.0, 41.0, 42.0 };
        calib.T = { 50.0, 51.0, 52.0 };

        std::array<uint8_t, 256> out{};
        HSBSender::encodeCalibrationBytes(calib, out);

        // Left intrinsics + distortion
        CHECK(readBigEndianDouble(out.data() + 0) == doctest::Approx(1.0)); // L_fx
        CHECK(readBigEndianDouble(out.data() + 8) == doctest::Approx(2.0)); // L_fy
        CHECK(readBigEndianDouble(out.data() + 16) == doctest::Approx(3.0)); // L_cx
        CHECK(readBigEndianDouble(out.data() + 24) == doctest::Approx(4.0)); // L_cy
        for (size_t i = 0; i < 8; ++i)
        {
            CHECK(readBigEndianDouble(out.data() + 32 + i * 8) == doctest::Approx(10.0 + static_cast<double>(i)));
        }
        // Right intrinsics + distortion
        CHECK(readBigEndianDouble(out.data() + 96) == doctest::Approx(20.0)); // R_fx
        CHECK(readBigEndianDouble(out.data() + 104) == doctest::Approx(21.0)); // R_fy
        CHECK(readBigEndianDouble(out.data() + 112) == doctest::Approx(22.0)); // R_cx
        CHECK(readBigEndianDouble(out.data() + 120) == doctest::Approx(23.0)); // R_cy
        for (size_t i = 0; i < 8; ++i)
        {
            CHECK(readBigEndianDouble(out.data() + 128 + i * 8) == doctest::Approx(30.0 + static_cast<double>(i)));
        }
        // Rig rotation, then translation
        CHECK(readBigEndianDouble(out.data() + 192) == doctest::Approx(40.0)); // Rx
        CHECK(readBigEndianDouble(out.data() + 200) == doctest::Approx(41.0)); // Ry
        CHECK(readBigEndianDouble(out.data() + 208) == doctest::Approx(42.0)); // Rz
        CHECK(readBigEndianDouble(out.data() + 216) == doctest::Approx(50.0)); // Tx
        CHECK(readBigEndianDouble(out.data() + 224) == doctest::Approx(51.0)); // Ty
        CHECK(readBigEndianDouble(out.data() + 232) == doctest::Approx(52.0)); // Tz
    }

    TEST_CASE("encodeCalibrationBytes - trailing bytes are zero-padded")
    {
        // 30 doubles = 240 bytes; buffer is 256. Verify bytes 240..255 stay zero even
        // when the payload is fully populated (guards against accidental overrun by
        // future layout changes).
        CameraCalibration calib{};
        calib.intrinsics[0].fill(1.0);
        calib.intrinsics[1].fill(1.0);
        calib.distortion[0].fill(1.0);
        calib.distortion[1].fill(1.0);
        calib.R.fill(1.0);
        calib.T.fill(1.0);
        std::array<uint8_t, 256> out{};
        HSBSender::encodeCalibrationBytes(calib, out);
        for (size_t i = 240; i < 256; ++i)
        {
            CHECK(out[i] == 0);
        }
    }

    TEST_CASE("encodeCalibrationBytes - realistic Eagle-like values round-trip")
    {
        // Plausible values matching the cuvslam_vb1940 sim scene (fx=1828, 2560x1984 imagers,
        // 88 mm baseline). Sanity check that realistic-magnitude doubles survive the encoding
        // and land at the expected offsets — catches issues that only fire on real data.
        CameraCalibration calib{};
        for (auto& img : calib.intrinsics)
        {
            img[0] = 1828.0; // fx
            img[1] = 1828.0; // fy
            img[2] = 1280.0; // cx = width/2 for 2560
            img[3] = 992.0; // cy = height/2 for 1984
        }
        calib.T[0] = -0.088; // Eagle stereo baseline in meters

        std::array<uint8_t, 256> out{};
        HSBSender::encodeCalibrationBytes(calib, out);

        CHECK(readBigEndianDouble(out.data() + 0) == doctest::Approx(1828.0)); // L_fx
        CHECK(readBigEndianDouble(out.data() + 8) == doctest::Approx(1828.0)); // L_fy
        CHECK(readBigEndianDouble(out.data() + 16) == doctest::Approx(1280.0)); // L_cx
        CHECK(readBigEndianDouble(out.data() + 24) == doctest::Approx(992.0)); // L_cy
        CHECK(readBigEndianDouble(out.data() + 96) == doctest::Approx(1828.0)); // R_fx
        CHECK(readBigEndianDouble(out.data() + 216) == doctest::Approx(-0.088)); // Tx (baseline)
        CHECK(readBigEndianDouble(out.data() + 224) == doctest::Approx(0.0)); // Ty
        CHECK(readBigEndianDouble(out.data() + 232) == doctest::Approx(0.0)); // Tz
    }
}
