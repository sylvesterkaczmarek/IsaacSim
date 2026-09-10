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
#include <isaacsim/common/string/String.hpp>

#include <unordered_map>

using namespace isaacsim::common::string;

namespace
{

void checkColor(const std::string& color, const std::vector<float>& expected)
{
    const std::vector<float> rgb = resolveColor(color);
    REQUIRE_EQ(rgb.size(), 3);
    for (std::size_t i = 0; i < 3; ++i)
    {
        CHECK_EQ(rgb[i], doctest::Approx(expected[i]));
    }
}

} // namespace

TEST_SUITE("String")
{

    TEST_CASE("VectorStringHasher")
    {
        VectorStringHasher hasher;
        CHECK_EQ(hasher({ "a", "b", "c" }), hasher({ "a", "b", "c" }));
        CHECK_NE(hasher({ "a", "b", "c" }), hasher({ "c", "b", "a" }));
        CHECK_NE(hasher({}), hasher({ "" }));

        std::unordered_map<std::vector<std::string>, int, VectorStringHasher> map;
        map[{ "x", "y" }] = 1;
        CHECK_EQ(map.at({ "x", "y" }), 1);
        CHECK_EQ(map.count({ "y", "x" }), 0);
    }

    TEST_CASE("toLower")
    {
        CHECK_EQ(toLower("Hello, World!"), "hello, world!");
    }

    TEST_CASE("toUpper")
    {
        CHECK_EQ(toUpper("Hello, World!"), "HELLO, WORLD!");
    }

    TEST_CASE("join")
    {
        CHECK_EQ(join({ "a", "b", "c" }, ", "), "a, b, c");
        CHECK_EQ(join({ "a", "b", "c" }, "-"), "a-b-c");
        CHECK_EQ(join({ "a" }, "-"), "a");
        CHECK_EQ(join({}, "-"), "");
        CHECK_EQ(join({ "a", "", "c" }, "-"), "a--c");
    }

    TEST_CASE("split")
    {
        CHECK_EQ(split("a, b, c", ", "), std::vector<std::string>{ "a", "b", "c" });
        CHECK_EQ(split("a-b-c", "-"), std::vector<std::string>{ "a", "b", "c" });
        CHECK_EQ(split("abc", "-"), std::vector<std::string>{ "abc" });
        CHECK_EQ(split("", "-"), std::vector<std::string>{ "" });
        CHECK_EQ(split("a--b", "-"), (std::vector<std::string>{ "a", "", "b" }));
        CHECK_EQ(split("a--b", ""), std::vector<std::string>{ "a--b" });
    }

    TEST_CASE("sort")
    {
        CHECK_EQ(sort({ "banana", "apple", "cherry" }), (std::vector<std::string>{ "apple", "banana", "cherry" }));
        CHECK_EQ(sort({ "banana", "apple", "cherry" }, false), (std::vector<std::string>{ "cherry", "banana", "apple" }));
        CHECK_EQ(sort({}), std::vector<std::string>{});
    }

    TEST_CASE("resolveColor")
    {
        SUBCASE("Hex - short form")
        {
            // Format #abc expands to #aabbcc.
            checkColor("#abc", { 0xaa / 255.0f, 0xbb / 255.0f, 0xcc / 255.0f });
            checkColor("#000", { 0.0f, 0.0f, 0.0f });
            checkColor("#fff", { 1.0f, 1.0f, 1.0f });
            // Case-insensitive.
            checkColor("#aBc", resolveColor("#ABC"));
            // Alpha nibble is accepted and discarded (#rgba).
            checkColor("#abcf", { 0xaa / 255.0f, 0xbb / 255.0f, 0xcc / 255.0f });
        }

        SUBCASE("Hex - full form")
        {
            checkColor("#0A1b2C", { 0x0a / 255.0f, 0x1b / 255.0f, 0x2c / 255.0f });
            checkColor("#000000", { 0.0f, 0.0f, 0.0f });
            checkColor("#FFFFFF", { 1.0f, 1.0f, 1.0f });
            // Case-insensitive.
            checkColor("#0a1B2c", resolveColor("#0A1B2C"));
            // Alpha byte is accepted and discarded (#rrggbbaa).
            checkColor("#0A1b2C80", { 0x0a / 255.0f, 0x1b / 255.0f, 0x2c / 255.0f });
        }

        SUBCASE("Grayscale string")
        {
            checkColor("0.5", { 0.5f, 0.5f, 0.5f });
            checkColor("0", { 0.0f, 0.0f, 0.0f });
            checkColor("1", { 1.0f, 1.0f, 1.0f });
        }

        SUBCASE("Base colors")
        {
            checkColor("k", { 0.0f, 0.0f, 0.0f });
            checkColor("r", { 1.0f, 0.0f, 0.0f });
            checkColor("g", { 0.0f, 0.5f, 0.0f });
            checkColor("w", { 1.0f, 1.0f, 1.0f });
            // Single-letter base colors are not case-folded (matches matplotlib).
            CHECK_THROWS_AS(resolveColor("K"), std::invalid_argument);
        }

        SUBCASE("CSS4 named colors - case-insensitive")
        {
            checkColor("AquaMarine", { 0x7f / 255.0f, 0xff / 255.0f, 0xd4 / 255.0f });
            checkColor("aquamarine", resolveColor("AQUAMARINE"));
            checkColor("RebeccaPurple", { 0x66 / 255.0f, 0x33 / 255.0f, 0x99 / 255.0f });
            // 'gray' / 'grey' spellings both resolve.
            checkColor("gray", resolveColor("grey"));
        }

        SUBCASE("Tableau colors - case-insensitive")
        {
            checkColor("tab:Green", { 0x2c / 255.0f, 0xa0 / 255.0f, 0x2c / 255.0f });
            checkColor("tab:green", resolveColor("TAB:GREEN"));
            checkColor("tab:gray", resolveColor("tab:grey"));
        }

        SUBCASE("CN cycle specification")
        {
            // C0 is the first cycle color (tab:blue), C2 the third (tab:green).
            checkColor("C0", resolveColor("tab:blue"));
            checkColor("C2", resolveColor("tab:green"));
            // The index wraps around the 10-color cycle.
            checkColor("C12", resolveColor("C2"));
            checkColor("C10", resolveColor("C0"));
        }

        SUBCASE("none - fully transparent resolves to black")
        {
            checkColor("none", { 0.0f, 0.0f, 0.0f });
            checkColor("None", { 0.0f, 0.0f, 0.0f });
            checkColor("NONE", { 0.0f, 0.0f, 0.0f });
        }

        SUBCASE("Invalid inputs throw")
        {
            CHECK_THROWS_AS(resolveColor(""), std::invalid_argument);
            CHECK_THROWS_AS(resolveColor("notacolor"), std::invalid_argument);
            CHECK_THROWS_AS(resolveColor("#12"), std::invalid_argument); // wrong hex length
            CHECK_THROWS_AS(resolveColor("#gggggg"), std::invalid_argument); // non-hex digits
            CHECK_THROWS_AS(resolveColor("1.5"), std::invalid_argument); // grayscale out of range
            CHECK_THROWS_AS(resolveColor("-0.1"), std::invalid_argument); // grayscale out of range
            CHECK_THROWS_AS(resolveColor("nan"), std::invalid_argument); // NaN is not in the 0-1 range
            CHECK_THROWS_AS(resolveColor("inf"), std::invalid_argument); // infinity is not in the 0-1 range
            CHECK_THROWS_AS(resolveColor("0.5 "), std::invalid_argument); // trailing junk after float
            CHECK_THROWS_AS(resolveColor("xkcd:eggshell"), std::invalid_argument); // xkcd not supported
        }
    }
}
