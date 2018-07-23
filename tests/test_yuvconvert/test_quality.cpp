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
#include <yuvconvert.h>
#include <yuvconvert/yuvconvert_common.h>
#include <ximage.h>
#include "test_utilities.h"
#include <fmt/printf.h>
#include <vector>
#include <cstdint>
#include <array>
#include <limits>
#include <algorithm>

class yuv420
{
public:
    yuv420(uint8_t *data, int width, int height)
        : data_(data)
        , width_(width)
        , height_(height)
    {
        source_ = { data_, nullptr, nullptr };
        source_stride_ = { width_ * 4, 0, 0 };

        const auto total = width_ * height_;
        yuv_buffer_.resize(total * 3);
        auto y = const_cast<uint8_t*>(yuv_buffer_.data());
        auto u = y + total;
        auto v = u + (total >> 1);
        destination_ = { y, u, v };
        destination_stride_ = { width_, width_ >> 1, width_ >> 1 };
    }

    std::array<uint8_t*, 3> &source()
    {
        return source_;
    }

    std::array<int, 3> &source_stride()
    {
        return source_stride_;
    }

    std::array<uint8_t*, 8> &destination()
    {
        return destination_;
    }

    std::array<int, 3> &destination_stride()
    {
        return destination_stride_;
    }

    int width() const
    {
        return width_;
    }

    int height() const
    {
        return height_;
    }

    uint8_t *data_;
    std::vector<uint8_t> yuv_buffer_;
    int width_;
    int height_;

    std::array<uint8_t*, 3> source_;
    std::array<int, 3> source_stride_;
    std::array<uint8_t*, 8> destination_;
    std::array<int, 3> destination_stride_;
};

#define YUV2R(y, u, v) clamp<uint8_t>((298 * ((y)-16) + 409 * ((v)-128) + 128) >> 8)
#define YUV2G(y, u, v) clamp<uint8_t>((298 * ((y)-16) - 100 * ((u)-128) - 208 * ((v)-128) + 128) >> 8)
#define YUV2B(y, u, v) clamp<uint8_t>((298 * ((y)-16) + 516 * ((u)-128) + 128) >> 8)

rgb YUVtoBGR(int y, int u, int v)
{
#if 0
    int32_t U, V, R, G, B;
    float Y = u;
    //lYUVColor.rgbRed;
    U = u - 128;//lYUVColor.rgbGreen - 128;
    V = v - 128;//lYUVColor.rgbBlue - 128;

    //    R = (int32_t)(1.164 * Y + 2.018 * U);
    //    G = (int32_t)(1.164 * Y - 0.813 * V - 0.391 * U);
    //    B = (int32_t)(1.164 * Y + 1.596 * V);
    R = (int32_t)(Y + 1.403f * V);
    G = (int32_t)(Y - 0.344f * U - 0.714f * V);
    B = (int32_t)(Y + 1.770f * U);

    R = std::min(255, std::max(0, R));
    G = std::min(255, std::max(0, G));
    B = std::min(255, std::max(0, B));
    rgb result = { (uint8_t)B,(uint8_t)G,(uint8_t)R };
    return result;
#endif
    return {
        yuv2b(y, u, v),
        yuv2g(y, u, v),
        yuv2r(y, u, v)
    };

    //return {
    //    YUV2B(y, u, v),
    //    YUV2G(y, u, v),
    //    YUV2R(y, u, v)
    //};
}

void yuv420_to_rgb(std::array<uint8_t*, 8> source, std::array<int, 3> source_stride, rgb* destination, int width, int height)
{
    auto y = source[0];
    auto u = source[1];
    auto v = source[2];

    for (int line = 0; line < height; ++line)
    {
        const auto offset = line * width;
        const auto offset_y = line * source_stride[0];
        const auto offset_uv = (line * source_stride[1]) >> 1;
        for (int x = 0; x < width; ++x)
        {
            const auto y_value = y[offset_y + x];
            const auto u_value = u[offset_uv + (x >> 1)];
            const auto v_value = v[offset_uv + (x >> 1)];

            destination[offset + x] = YUVtoBGR(y_value, u_value, v_value);
        }
    }
}

static uint32_t SumSquareError_C(const uint8_t* src_a,
    const uint8_t* src_b,
    int count) {
    uint32_t sse = 0u;
    for (int x = 0; x < count; ++x) {
        int diff = src_a[x] - src_b[x];
        sse += static_cast<uint32_t>(diff * diff);
    }
    return sse;
}

double ComputeSumSquareError(const uint8_t* src_a, const uint8_t* src_b, int count)
{
    uint32_t(*SumSquareError)(const uint8_t* src_a, const uint8_t* src_b, int count) =
        SumSquareError_C;

    const int kBlockSize = 1 << 15;
    uint64_t sse = 0;

    for (int i = 0; i < (count - (kBlockSize - 1)); i += kBlockSize) {
        sse += SumSquareError(src_a + i, src_b + i, kBlockSize);
    }
    src_a += count & ~(kBlockSize - 1);
    src_b += count & ~(kBlockSize - 1);
    int remainder = count & (kBlockSize - 1) & ~15;
    if (remainder) {
        sse += SumSquareError(src_a, src_b, remainder);
        src_a += remainder;
        src_b += remainder;
    }
    remainder = count & 15;
    if (remainder) {
        sse += SumSquareError_C(src_a, src_b, remainder);
    }
    return static_cast<double>(sse);
}

static const double kMaxPSNR = 128.0;

// PSNR formula: psnr = 10 * log10 (Peak Signal^2 * size / sse)
// Returns 128.0 (kMaxPSNR) if sse is 0 (perfect match).
double ComputePSNR(double sse, double size) {
    const double kMINSSE = 255.0 * 255.0 * size / pow(10.0, kMaxPSNR / 10.0);
    if (sse <= kMINSSE)
        sse = kMINSSE;  // Produces max PSNR of 128
    return 10.0 * log10(255.0 * 255.0 * size / sse);
}


TEST(test_quality, psnr)
{
    CxImage test;
    test.Load(L"D:/dev/camstudio/CamEncoder/tests/samples/test.bmp");
    const auto dib = reinterpret_cast<BITMAPINFOHEADER*>(test.GetDIB());
    const auto dib_data = reinterpret_cast<rgb*>(test.GetImageData());

    const auto source_width = dib->biWidth;
    const auto source_height = dib->biHeight;
    auto source_buffer = std::vector<rgba>(source_width * source_height);
    auto source_rgba = reinterpret_cast<uint8_t*>(source_buffer.data());
    for (int i = 0; i < source_width * source_height; i++)
    {
        const auto &pxl = dib_data[i];
        source_buffer[i] = rgba{ pxl.b, pxl.g, pxl.r, 0 };
    }

    yuv420 yuv_a(
        source_rgba,
        source_width,
        source_height
    );

    yuv420 yuv_b(
        source_rgba,
        source_width,
        source_height
    );

    yuvconvert::bgra_to_420(
        yuv_a.destination().data(),
        yuv_a.destination_stride().data(),
        yuv_a.source().data(),
        yuv_a.width(),
        yuv_a.height(),
        yuv_a.source_stride().data(),
        yuvconvert::simd_mode::plain_c
    );

    yuvconvert::bgra_to_420(
        yuv_b.destination().data(),
        yuv_b.destination_stride().data(),
        yuv_b.source().data(),
        yuv_b.width(),
        yuv_b.height(),
        yuv_b.source_stride().data(),
        yuvconvert::simd_mode::ssse3
    );

    std::vector<rgb> rgb_a(yuv_a.width() * yuv_a.height());
    yuv420_to_rgb(yuv_a.destination(), yuv_a.destination_stride(), rgb_a.data(), yuv_a.width(), yuv_a.height());

    std::vector<rgb> rgb_b(yuv_b.width() * yuv_b.height());
    yuv420_to_rgb(yuv_b.destination(), yuv_b.destination_stride(), rgb_b.data(), yuv_b.width(), yuv_b.height());

    std::vector<uint8_t> red_a(yuv_a.width() * yuv_a.height());
    auto dst_a = red_a.begin();
    for (const auto &pixel : rgb_a)
        *dst_a++ = pixel.r;

    std::vector<uint8_t> red_b(yuv_b.width() * yuv_b.height());
    auto dst_b = red_b.begin();
    for (const auto &pixel : rgb_b)
        *dst_b++ = pixel.r;


    const auto sse_a = ComputeSumSquareError(source_rgba, (uint8_t*)red_a.data(), (int)red_a.size());
    const auto psnr_a = ComputePSNR(sse_a, (int)rgb_a.size());
    fmt::print("psnr a: {}  sse: {}\n", psnr_a, sse_a);

    const auto sse_b = ComputeSumSquareError(source_rgba, (uint8_t*)red_b.data(), (int)red_b.size());
    const auto psnr_b = ComputePSNR(sse_b, (int)rgb_b.size());
    fmt::print("psnr b: {} sse: {}\n", psnr_b, sse_b);
}
