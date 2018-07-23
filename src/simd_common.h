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

#include "simd_vec.h"

// BT.601 Studio swing.
namespace simd
{
// *y++ = ((66 * r1 + 129 * g1 + 25 * b1 + 128) >> 8) + 16;
static const vec3 y_mul = simd::vec3_set(25, 129, 66);

// *u++ = ((-38 * r1 + -74 * g1 + 112 * b1 + 128) >> 8) + 128;
static const vec3 u_mul = simd::vec3_set(112, -74, -38);

// *v++ = ((112 * r1 + -94 * g1 + -18 * b1 + 128) >> 8) + 128;
static const vec3 v_mul = simd::vec3_set(-18, -94, 112);

static const auto y_add = _mm_set1_epi8(16);
static const auto uv_add = _mm_set1_epi16(128);

} // namespace simd
