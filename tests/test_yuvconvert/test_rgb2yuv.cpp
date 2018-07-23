/* Copyright(c) 2018 Steven Hoving
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files(the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions :
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <yuvconvert.h>
#include <yuvconvert/yuvconvert_common.h>

#include "test_utilities.h"
#include <fmt/printf.h>
#include <vector>
#include <cstdint>
#include <array>
//#include <limits>
#include <algorithm>

template<typename T>
std::vector<uint8_t> create_rest_rbgx_buffer(const int width, const int height)
{
    int color = 0;
    std::vector<uint8_t> result(width * height * sizeof(T));
    for (auto &itr : result)
        itr = color++ % 255;

    return result;
}

template <typename T>
class rgbx2yuv_fixture_internal : public testing::TestWithParam<int>
{
public:
    ~rgbx2yuv_fixture_internal() override = default;
    void SetUp() override
    {
        const auto width = GetParam();
        const auto height = abs(width);
        const auto total = abs(width * height);
        const auto pixel_size = sizeof(T);
        rgb_buffer = create_rest_rbgx_buffer<T>(abs(width), abs(width));

        // although yuv 420 only requires 1.5x total (1x width x height + 1/4 + 1/4) .... we just
        // use 2 here for now.
        yuv420_sse.resize(total * 2);
        yuv420_c.resize(total * 2);

        auto sse_y = yuv420_sse.data();
        auto sse_u = sse_y + total;
        auto sse_v = sse_u + (total >> 1);

        auto c_y = yuv420_c.data();
        auto c_u = c_y + total;
        auto c_v = c_u + (total >> 1);

        destination_sse[0] = sse_y;
        destination_sse[1] = sse_u;
        destination_sse[2] = sse_v;

        destination_c[0] = c_y;
        destination_c[1] = c_u;
        destination_c[2] = c_v;

        destination_stride[0] = abs(width);
        destination_stride[1] = abs(width) >> 1;
        destination_stride[2] = abs(width) >> 1;

        if (width < 0)
            src[0] = rgb_buffer.data() + (rgb_buffer.size() + (width * sizeof(T)));
        else
            src[0] = rgb_buffer.data();

        src_stride[0] = width * sizeof(T);
    }

    void TearDown() override {}

protected:
    std::vector<uint8_t> rgb_buffer;
    std::vector<uint8_t> yuv420_sse;
    std::vector<uint8_t> yuv420_c;

    uint8_t *destination_sse[3]{nullptr, nullptr, nullptr};
    uint8_t *destination_c[3]{nullptr, nullptr, nullptr};
    int destination_stride[3]{0, 0, 0};

    uint8_t *src[3]{nullptr, nullptr, nullptr};
    int src_stride[3]{0, 0, 0};
};

using rgb2yuv_fixture = rgbx2yuv_fixture_internal<rgb>;
using rgba2yuv_fixture = rgbx2yuv_fixture_internal<rgba>;

TEST_P(rgb2yuv_fixture, test_rgb)
{
    const auto width = GetParam();
    const auto height = width;

    yuvconvert::bgr_to_420(destination_c, destination_stride, src, width, height, src_stride, yuvconvert::simd_mode::plain_c);
    yuvconvert::bgr_to_420(destination_sse, destination_stride, src, width, height, src_stride, yuvconvert::simd_mode::ssse3);
    EXPECT_TRUE(std::equal(yuv420_c.begin(), yuv420_c.end(), yuv420_sse.begin()));
}

TEST_P(rgba2yuv_fixture, test_rgba)
{
    const auto width = GetParam();
    const auto height = width;

    yuvconvert::bgra_to_420(destination_c, destination_stride, src, width, height, src_stride, yuvconvert::simd_mode::plain_c);
    yuvconvert::bgra_to_420(destination_sse, destination_stride, src, width, height, src_stride, yuvconvert::simd_mode::ssse3);
    EXPECT_TRUE(std::equal(yuv420_c.begin(), yuv420_c.end(), yuv420_sse.begin()));
}

INSTANTIATE_TEST_CASE_P(rgb_test_sequence, rgb2yuv_fixture, ::testing::ValuesIn(std::vector<int>{
    128,
    256,
    512,
    4096 - 2,
    -128,
    -4096 - 2,
}));

INSTANTIATE_TEST_CASE_P(rgba_test_sequence, rgba2yuv_fixture, ::testing::ValuesIn(std::vector<int>{
    128,
    256,
    512,
    4096 - 2,
    -128,
    -4096 - 2,
}));
