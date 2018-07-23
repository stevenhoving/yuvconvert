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

#include "to_420.h"
#include "to_420_c.h"
#include "to_420_ssse3.h"
#include "yuvconvert.h"

namespace yuvconvert
{
using bgrx_row_to_y_row = void(const unsigned char *src, unsigned char *dst, const int width);
using bgrx_row_to_yuv_row = void(const unsigned char *src, unsigned char *dst_y, unsigned char *dst_u, unsigned char *dst_v, const int width);


void bgra_to_420(unsigned char *destination[3], const int dst_stride[3],
                 const unsigned char *const source[3], const int width, const int height,
                 const int src_stride[3])
{
    auto src = source[0];
    auto y = destination[0];
    auto u = destination[1];
    auto v = destination[2];

    const auto raw_stride = src_stride[0];
    const auto y_stride = dst_stride[0];
    const auto u_stride = dst_stride[1];
    const auto v_stride = dst_stride[2];

    for (int line = 0; line < height; line += 2)
    {
        bgra_row_to_yuv_row_c(src, y, u, v, width);
        src += raw_stride;
        y += y_stride;
        u += u_stride;
        v += v_stride;
        bgra_row_to_y_row_c(src, y, width);
        src += raw_stride;
        y += y_stride;
    }
}

void bgr_to_420(unsigned char *destination[3], const int dst_stride[3],
                const unsigned char *const source[3], const int width, const int height,
                const int src_stride[3])
{
    auto src = source[0];
    auto y = destination[0];
    auto u = destination[1];
    auto v = destination[2];

    const auto raw_stride = src_stride[0];
    const auto y_stride = dst_stride[0];
    const auto u_stride = dst_stride[1];
    const auto v_stride = dst_stride[2];

    for (int line = 0; line < height; line += 2)
    {
        bgr_row_to_yuv_row_c(src, y, u, v, width);
        src += raw_stride;
        y += y_stride;
        u += u_stride;
        v += v_stride;
        bgr_row_to_y_row_c(src, y, width);
        src += raw_stride;
        y += y_stride;
    }
}

void bgr_to_420(unsigned char *destination[3], const int dst_stride[3], const unsigned char *const source[3],
    const int width, const int height, const int src_stride[3], simd_mode mode /*= simd_mode::plain_c*/)
{
    auto src = source[0];
    auto y = destination[0];
    auto u = destination[1];
    auto v = destination[2];

    const auto raw_stride = src_stride[0];
    const auto y_stride = dst_stride[0];
    const auto u_stride = dst_stride[1];
    const auto v_stride = dst_stride[2];

    bgrx_row_to_yuv_row *yuv_row_converter = nullptr;
    bgrx_row_to_y_row *y_row_converter = nullptr;
    if (mode == simd_mode::plain_c)
    {
        yuv_row_converter = bgr_row_to_yuv_row_c;
        y_row_converter = bgr_row_to_y_row_c;
    }
    else if (mode == simd_mode::ssse3)
    {
        yuv_row_converter = bgr_row_to_yuv_row_ssse3;
        y_row_converter = bgr_row_to_y_row_ssse3;
    }

    for (int line = 0; line < height; line += 2)
    {
        yuv_row_converter(src, y, u, v, width);
        src += raw_stride;
        y += y_stride;
        u += u_stride;
        v += v_stride;
        y_row_converter(src, y, width);
        src += raw_stride;
        y += y_stride;
    }
}

void bgra_to_420(unsigned char *destination[3], const int dst_stride[3], const unsigned char *const source[3],
    const int width, const int height, const int src_stride[3], simd_mode mode /*= simd_mode::plain_c*/)
{
    auto src = source[0];
    auto y = destination[0];
    auto u = destination[1];
    auto v = destination[2];

    const auto raw_stride = src_stride[0];
    const auto y_stride = dst_stride[0];
    const auto u_stride = dst_stride[1];
    const auto v_stride = dst_stride[2];

    bgrx_row_to_yuv_row *yuv_row_converter = nullptr;
    bgrx_row_to_y_row *y_row_converter = nullptr;
    if (mode == simd_mode::plain_c)
    {
        yuv_row_converter = bgra_row_to_yuv_row_c;
        y_row_converter = bgra_row_to_y_row_c;
    }
    else if (mode == simd_mode::ssse3)
    {
        yuv_row_converter = bgra_row_to_yuv_row_ssse3;
        y_row_converter = bgra_row_to_y_row_ssse3;
    }

    for (int line = 0; line < height; line += 2)
    {
        yuv_row_converter(src, y, u, v, width);
        src += raw_stride;
        y += y_stride;
        u += u_stride;
        v += v_stride;
        y_row_converter(src, y, width);
        src += raw_stride;
        y += y_stride;
    }
}

} // namespace yuvconvert
