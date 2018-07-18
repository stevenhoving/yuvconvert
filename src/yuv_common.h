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

#pragma once

#include <algorithm>

template<typename T, typename Y>
static constexpr T clamp(Y a)
{
    constexpr auto max_val = std::numeric_limits<T>::max();
    constexpr auto min_val = std::numeric_limits<T>::min();
    return std::clamp<T>(a, min_val, max_val);
}

/* commonly found approximate yuv rgb convention functions */
static constexpr auto rgb2y(const uint8_t r, const uint8_t g, const uint8_t b) -> uint8_t
{
    return ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
}

static constexpr auto rgb2u(const uint8_t r, const uint8_t g, const uint8_t b) -> uint8_t
{
    return ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
}

static constexpr auto rgb2v(const uint8_t r, const uint8_t g, const uint8_t b) -> uint8_t
{
    return ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
}

static constexpr auto yuv2r(const uint8_t y, const uint8_t u, const uint8_t v) -> uint8_t
{
    return clamp<uint8_t>((298 * (y - 16) + 409 * (v - 128) + 128) >> 8);
}

static constexpr auto yuv2g(const uint8_t y, const uint8_t u, const uint8_t v) -> uint8_t
{
    return clamp<uint8_t>((298 * ((y)-16) - 100 * (u - 128) - 208 * (v - 128) + 128) >> 8);
}

static constexpr auto yuv2b(const uint8_t y, const uint8_t u, const uint8_t v) -> uint8_t
{
    return clamp<uint8_t>((298 * (y - 16) + 516 * (u - 128) + 128) >> 8);
}
