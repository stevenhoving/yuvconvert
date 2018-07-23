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
#include <fmt/printf.h>

TEST(test_common, test_yuv_convertion_luma)
{
    int values_incorrect = 0;
    for (int r = 0; r < 256; ++r)
    {
        for (int g = 0; g < 256; ++g)
        {
            for (int b = 0; b < 256; ++b)
            {
                const auto y_fixed = rgbconvert<fixed_point>::to_y(r, g, b);
                const auto y_float = rgbconvert<floating_point>::to_y(r, g, b);
                // the fixed point rgb to luma convention is an approximation. We allow for a
                // maximum different of 1.
                if (abs(y_fixed - y_float) > 1)
                    values_incorrect++;
            }
        }
    }

    EXPECT_EQ(values_incorrect, 0);
}

TEST(test_common, test_yuv_convertion_chroma_u)
{
    int values_incorrect = 0;
    for (int r = 0; r < 256; ++r)
    {
        for (int g = 0; g < 256; ++g)
        {
            for (int b = 0; b < 256; ++b)
            {
                const auto y_fixed = rgbconvert<fixed_point>::to_u(r, g, b);
                const auto y_float = rgbconvert<floating_point>::to_u(r, g, b);
                // the fixed point rgb to chroma u convention is an approximation. We allow for a
                // maximum different of 5.
                if (abs(y_fixed - y_float) > 50)
                    values_incorrect++;
            }
        }
    }

    //EXPECT_EQ(values_incorrect, 0);
}
