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

#include <emmintrin.h>

#if defined(__clang__)
#define COMPILER_CLANG 1
#define COMPILER_MSVC 0
#elif defined(_MSC_VER)
#define COMPILER_CLANG 0
#define COMPILER_MSVC 1
#endif

#if COMPILER_MSVC
#define __packed_struct(x) __pragma(pack(push, 1)) struct x
#define __packed_struct_end __pragma(pack(pop))
#define __no_unroll __pragma(loop(no_vector))
#else
#define __packed_struct(x) struct __attribute__((__packed__)) x
#define __packed_struct_end
#define __no_unroll _Pragma("nounroll")
#endif

#if 0
#if defined(_MSC_VER) && !defined(__clang__)
#define __packed_struct(x) __pragma(pack(push, 1)) struct x
#define __packed_struct_end __pragma(pack(pop))

#define __no_unroll __pragma(loop(no_vector))
#else
#define __packed_struct(x) struct __attribute__((__packed__)) x
#define __packed_struct_end

#define __no_unroll _Pragma("nounroll")
#endif
#endif

namespace simd
{
template <typename T>
static inline T align_up(const T size, const T align) noexcept
{
    return ((size + (align - 1)) & ~(align - 1));
}

template <typename T>
static inline T align_down(const T size, const T align) noexcept
{
    return size & ~(align - 1);
}

} // namespace simd
