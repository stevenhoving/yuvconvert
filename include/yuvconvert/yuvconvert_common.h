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
#include <array>

#include <static_math/cmath.h>

/* \note the idea behind this clamp function is good, but there is room for improvement. */
template<typename ClampType, typename ValueType>
static constexpr ClampType clamp(ValueType value)
{
    constexpr auto max_val = std::numeric_limits<ClampType>::max();
    constexpr auto min_val = std::numeric_limits<ClampType>::min();
    return std::clamp<ValueType>(value, min_val, max_val);
}

//Y = 0.299 * R + 0.587 * G + 0.114 * B;
//U = -0.14713 * R - 0.28886 * G + 0.436 * B;
//V = 0.615 * R - 0.51499 * G - 0.10001 * B;


// Full range
//{ { 0.299f, 0.587f, 0.114f},
//{ -0.169f, -0.331f,  0.500f },
//{ 0.500f, -0.419f, -0.081f }},
//
//// bt601
//{ { 0.257f,  0.504f,  0.098f},
// {-0.148f, -0.291f,  0.439f},
// { 0.439f, -0.368f, -0.071f} },
//
//    // bt709
//{ { 0.183f,  0.614f,  0.062f},
// {-0.101f, -0.339f,  0.439f},
// { 0.439f, -0.399f, -0.040f} },
//    };

/* currently we convert to BT601 colorspace */
//BT601

/*
Y = (0.257 * R) + (0.504 * G) + (0.098 * B) + 16
Cr = V = (0.439 * R) - (0.368 * G) - (0.071 * B) + 128
Cb = U = -(0.148 * R) - (0.291 * G) + (0.439 * B) + 128

*/

struct fixed_point
{
};

struct floating_point
{
};

template<typename ConvertionType>
class rgbconvert
{
public:
    static constexpr auto to_y(const uint8_t r, const uint8_t g, const uint8_t b) -> uint8_t;
    static constexpr auto to_u(const uint8_t r, const uint8_t g, const uint8_t b) -> uint8_t;
    static constexpr auto to_v(const uint8_t r, const uint8_t g, const uint8_t b) -> uint8_t;
};

// ITU-R BT.601 standard.
constexpr std::array<double, 3> luma_factor = { 0.2569, 0.5044, 0.0979 };
constexpr std::array<double, 3> chroma_u_factor = { -0.1483, -0.2911, 0.4394 };
constexpr std::array<double, 3> chroma_v_factor = { 0.4394, - 0.3679, -0.0715 };

// ITU-R BT.601 studio swing.
// todo

// ITU-R BT.601 full swing.
// todo

constexpr auto fixed_point_precision = 8;
constexpr auto fixed_point_half = 1 << (fixed_point_precision - 1);

constexpr auto luma_r_factor = (unsigned int)smath::round(luma_factor[0] * (1 << fixed_point_precision));
constexpr auto luma_g_factor = (int)smath::round(luma_factor[1] * (1 << fixed_point_precision));
constexpr auto luma_b_factor = (int)smath::round(luma_factor[2] * (1 << fixed_point_precision));

constexpr auto chroma_u_r_factor = (int)smath::round(chroma_u_factor[0] * (1 << fixed_point_precision));
constexpr auto chroma_u_g_factor = (int)smath::round(chroma_u_factor[1] * (1 << fixed_point_precision));
constexpr auto chroma_u_b_factor = (int)smath::round(chroma_u_factor[2] * (1 << fixed_point_precision));

// fixed point specialization.
template<>
static constexpr auto rgbconvert<fixed_point>::to_y(const uint8_t r, const uint8_t g, const uint8_t b) -> uint8_t
{
    const auto luma = (luma_r_factor * r) + (luma_g_factor * g) + (luma_b_factor * b);
    return (luma + fixed_point_half + (16 << fixed_point_precision)) >> fixed_point_precision;
}

template<>
static constexpr auto rgbconvert<fixed_point>::to_u(const uint8_t r, const uint8_t g, const uint8_t b) -> uint8_t
{
    //return ((-38 * r + -74 * g + 112 * b + 128) >> 8) + 128;
    //return ((-38 * static_cast<int>(r) + -74 * static_cast<int>(g) + 112 * static_cast<int>(b) + 128) >> 8) + 128;

    auto u = (chroma_u_r_factor * r) + (chroma_u_g_factor * g) + (chroma_u_b_factor * b);
    u = (u + fixed_point_half + (128 << (fixed_point_precision + 2))) >> (fixed_point_precision + 2);
    return ((u & ~0xff) == 0) ? u : (u < 0) ? 0 : 255;
}

template<>
static constexpr auto rgbconvert<fixed_point>::to_v(const uint8_t r, const uint8_t g, const uint8_t b) -> uint8_t
{
    //return ((112 * r + -94 * g + -18 * b + 128) >> 8) + 128;
    return ((112 * static_cast<int>(r) + -94 * static_cast<int>(g) + -18 * static_cast<int>(b) + 128) >> 8) + 128;
}

// floating point specialization

template<>
static constexpr auto rgbconvert<floating_point>::to_y(const uint8_t r, const uint8_t g, const uint8_t b) -> uint8_t
{
    const auto y = (luma_factor[0] * r) + (luma_factor[1] * g) + (luma_factor[2] * b) + 16;
    return static_cast<uint8_t>(std::clamp(y, 0.0, 255.0));
}

template<>
static constexpr auto rgbconvert<floating_point>::to_u(const uint8_t r, const uint8_t g, const uint8_t b) -> uint8_t
{
    const auto u = (chroma_u_factor[0] * r) - (chroma_u_factor[1] * g) + (chroma_u_factor[2] * b) + 128;
    return static_cast<uint8_t>(std::clamp(u, 0.0, 255.0));
}

template<>
static constexpr auto rgbconvert<floating_point>::to_v(const uint8_t r, const uint8_t g, const uint8_t b) -> uint8_t
{
    const auto v = (0.615 * r) - (0.51499 * g) - (0.10001 * b) + 128;
    return static_cast<uint8_t>(std::clamp(v, 0.0, 255.0));
}

//////////////////////////////////////////////////////////////////////////

static constexpr auto rgb2y(const uint8_t r, const uint8_t g, const uint8_t b) -> uint8_t
{
    return ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
    //const auto y = 0.299 * r + 0.587 * g + 0.114 * b;
    //return static_cast<uint8_t>(std::clamp(y, 0.0, 255.0));
}

static constexpr auto rgb2u(const uint8_t r, const uint8_t g, const uint8_t b) -> uint8_t
{
    return ((-38 * r + -74 * g + 112 * b + 128) >> 8) + 128;
    //const auto u = -0.14713 * r - 0.28886 * g + 0.436 * b;
    //return static_cast<uint8_t>(std::clamp(u, 0.0, 255.0));
}

static constexpr auto rgb2v(const uint8_t r, const uint8_t g, const uint8_t b) -> uint8_t
{
    return ((112 * r + -94 * g + -18 * b + 128) >> 8) + 128;
    //return 0.615 * r - 0.51499 * g - 0.10001 * b;
    //const auto v = 0.615 * r - 0.51499 * g - 0.10001 * b;
    //return static_cast<uint8_t>(std::clamp(v, 0.0, 255.0));
}

//B = 1.164(Y - 16) + 2.018(U - 128)
//G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
//R = 1.164(Y - 16) + 1.596(V - 128)

//C = Y - 16
//D = U - 128
//E = V - 128
//
//R = clip((298 * C + 409 * E + 128) >> 8)
//G = clip((298 * C - 100 * D - 208 * E + 128) >> 8)
//B = clip((298 * C + 516 * D + 128) >> 8)

static constexpr auto yuv2r(const uint8_t y, const uint8_t u, const uint8_t v) -> uint8_t
{
    return clamp<uint8_t>((298 * (y - 16) + 409 * (v - 128) + 128) >> 8);
    //const auto r = 1.164 * (int(y) - 16) + 1.596 * (int(v) - 128);
    //return static_cast<uint8_t>(std::clamp(r, 0.0, 255.0));
}

static constexpr auto yuv2g(const uint8_t y, const uint8_t u, const uint8_t v) -> uint8_t
{
    return clamp<uint8_t>((298 * (y-16) - 100 * (u - 128) - 208 * (v - 128) + 128) >> 8);
    //const auto g = 1.164 * (int(y) - 16) - 0.813 * (int(v) - 128) - 0.391 * (int(u) - 128);
    //return static_cast<uint8_t>(std::clamp(g, 0.0, 255.0));
}

static constexpr auto yuv2b(const uint8_t y, const uint8_t u, const uint8_t v) -> uint8_t
{
    return clamp<uint8_t>((298 * (y - 16) + 516 * (u - 128) + 128) >> 8);
    //const auto b = 1.164 * (int(y) - 16) + 2.018 * (int(u) - 128);
    //return static_cast<uint8_t>(std::clamp(b, 0.0, 255.0));
}

static constexpr uint32_t adder_scaler(const uint32_t a, const uint32_t b)
{
    return (((a ^ b) & 0x7f7f7f7f) >> 1) | (a & b);
}