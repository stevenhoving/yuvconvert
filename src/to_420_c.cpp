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

#include "to_420_c.h"
#include "yuvconvert_common.h"

// c implementation for converting a rgbx row to y
template<int pixel_width>
constexpr void bgrx_row_to_y_row(const unsigned char *src, unsigned char *dst, const int width)
{
    for (int x = 0; x < width; ++x)
    {
        *dst++ = rgb2y(
            src[2], // r
            src[1], // g
            src[0]);// b
        src += pixel_width;
    }
}

template<int pixel_width>
constexpr void bgrx_row_to_yuv_row(const unsigned char *src, unsigned char *dst_y,
                                   unsigned char *dst_u, unsigned char *dst_v, const int width)
{
    for (int x = 0; x < width; x += 2)
    {
        auto r = src[2];
        auto g = src[1];
        auto b = src[0];
        *dst_y++ = rgb2y(r, g, b);
        *dst_u++ = rgb2u(r, g, b);
        *dst_v++ = rgb2v(r, g, b);
        src += pixel_width;

        r = src[2];
        g = src[1];
        b = src[0];
        *dst_y++ = rgb2y(r, g, b);
        src += pixel_width;
    }
}

void bgra_row_to_y_row_c(const unsigned char *src, unsigned char *dst, const int width)
{
    bgrx_row_to_y_row<4>(src, dst, width);
}

void bgra_row_to_yuv_row_c(const unsigned char *src, unsigned char *dst_y, unsigned char *dst_u,
                           unsigned char *dst_v, const int width)
{
    bgrx_row_to_yuv_row<4>(src, dst_y, dst_u, dst_v, width);
}

void bgr_row_to_y_row_c(const unsigned char *src, unsigned char *dst, const int width)
{
    bgrx_row_to_y_row<3>(src, dst, width);
}

void bgr_row_to_yuv_row_c(const unsigned char *src, unsigned char *dst_y, unsigned char *dst_u,
                          unsigned char *dst_v, const int width)
{
    bgrx_row_to_yuv_row<3>(src, dst_y, dst_u, dst_v, width);
}
