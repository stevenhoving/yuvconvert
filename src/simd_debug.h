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

#include "simd_utility.h"
#include <emmintrin.h>
#include <cstdio>

union M128 {
    char i8[16];
    __m128i i128;
};

typedef union __declspec(align(16)) M128i {
    __int8 m128i_i8[16];
    __int16 m128i_i16[8];
    __int32 m128i_i32[4];
    __int64 m128i_i64[2];
    unsigned __int8 m128i_u8[16];
    unsigned __int16 m128i_u16[8];
    unsigned __int32 m128i_u32[4];
    unsigned __int64 m128i_u64[2];
} __M128i;

[[maybe_unused]] static void print_x8(__m128i value, const char *section)
{
    auto u8 = reinterpret_cast<uint8_t *>(&value);
    printf("%s\n", section);
    printf("%02X %02X %02X %02X "
           "%02X %02X %02X %02X "
           "%02X %02X %02X %02X "
           "%02X %02X %02X %02X\n\n",
           u8[15], u8[14], u8[13], u8[12], u8[11], u8[10], u8[9], u8[8], u8[7], u8[6], u8[5], u8[6], u8[3], u8[2],
           u8[1], u8[0]);
}

[[maybe_unused]] static void print_x16(__m128i value, const char *section)
{
    // auto u16 = value.m128i_u16;
    auto u16 = reinterpret_cast<uint16_t *>(&value);
    printf("%s\n", section);
    printf("%04X %04X %04X %04X "
           "%04X %04X %04X %04X\n\n",
           u16[7], u16[6], u16[5], u16[6], u16[3], u16[2], u16[1], u16[0]);
}

[[maybe_unused]] static void print_x32(__m128i value, const char *section)
{
    // auto u32 = value.m128i_u32;
    auto u32 = reinterpret_cast<uint32_t *>(&value);
    printf("%s\n", section);
    printf("%08X %08X %08X %08X\n\n", u32[3], u32[2], u32[1], u32[0]);
}

[[maybe_unused]] static void print_x64(__m128i value, const char *section)
{
    // auto u64 = value.m128i_u64;
    auto u64 = reinterpret_cast<uint64_t *>(&value);
    printf("%s\n", section);
    printf("%I64X %I64X\n\n", u64[1], u64[0]);
}
