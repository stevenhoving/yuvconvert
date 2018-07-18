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

#include <benchmark/benchmark.h>
#include <yuvconvert.h>

class bgrx_to_420_fixture : public ::benchmark::Fixture
{
public:
    void SetUp(const ::benchmark::State& state)
    {
        width = state.range(0);
        height = state.range(1);
        const auto total = width * height;

        rgb_buffer.resize(total * 4);
        yuv420_buffer.resize(total * 3);

        source[0] = rgb_buffer.data();
        source_stride[0] = width * 4;

        auto y = yuv420_buffer.data();
        auto u = y + total;
        auto v = u + (total >> 1);

        destination[0] = y;
        destination[1] = u;
        destination[2] = v;

        destination_stride[0] = width;
        destination_stride[1] = width >> 1;
        destination_stride[2] = width >> 1;
    }

    void TearDown(const ::benchmark::State& state)
    {
    }

public:
    std::vector<unsigned char> rgb_buffer;
    unsigned char * source[3] = {};
    int source_stride[3] = {};

    std::vector<unsigned char> yuv420_buffer;
    unsigned char *destination[3] = {};
    int destination_stride[3] = {};

    int width{0};
    int height{0};
};

BENCHMARK_DEFINE_F(bgrx_to_420_fixture, bgra_to_420)(benchmark::State& st)
{
    for (auto _ : st) {
        yuvconvert::bgra_to_420(destination, destination_stride, source, width, height, source_stride);
    }
}

BENCHMARK_REGISTER_F(bgrx_to_420_fixture, bgra_to_420)
    ->Args({ 128, 128})
    ->Args({ 256, 256})
    ->Args({ 512, 512})
    ->Args({ 1024, 1024})
    ->Args({ 2048, 2048})
    ->Args({ 4096, 4096});

BENCHMARK_DEFINE_F(bgrx_to_420_fixture, bgr_to_420)(benchmark::State& st)
{
    for (auto _ : st) {
        yuvconvert::bgr_to_420(destination, destination_stride, source, width, height, source_stride);
    }
}

BENCHMARK_REGISTER_F(bgrx_to_420_fixture, bgr_to_420)
    ->Args({ 128, 128 })
    ->Args({ 256, 256 })
    ->Args({ 512, 512 })
    ->Args({ 1024, 1024 })
    ->Args({ 2048, 2048 })
    ->Args({ 4096, 4096 });

#if 0
BENCHMARK_DEFINE_F(bgrx_to_420_fixture, bgra2yuv420p_sse)(benchmark::State& st)
{
    for (auto _ : st) {
        bgra2yuv420p_sse(destination, destination_stride, source, width, height, source_stride);
    }
}

BENCHMARK_REGISTER_F(bgrx_to_420_fixture, bgra2yuv420p_sse)
    ->Args({ 128, 128})
    ->Args({ 256, 256})
    ->Args({ 512, 512})
    ->Args({ 1024, 1024})
    ->Args({ 2048, 2048})
    ->Args({ 4096, 4096});

#endif