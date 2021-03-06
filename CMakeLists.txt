# Copyright(c) 2018 Steven Hoving
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

add_subdirectory(dep)

set(YUVCONVERT_SOURCE
    src/to_420.cpp
    src/to_420.h
    src/to_420_c.cpp
    src/to_420_c.h
    src/to_420_ssse3.cpp
    src/to_420_ssse3.h
    src/simd_common.h
    src/simd_debug.h
    src/simd_utility.h
    src/simd_vec.h
    src/yuv_pixel_type.h
    include/yuvconvert/yuvconvert_common.h
)

set(YUVCONVERT_INTERFACE
    include/yuvconvert.h
)

source_group(yuvconvert FILES
    ${YUVCONVERT_SOURCE}
    ${YUVCONVERT_INTERFACE}
)

add_library(yuvconvert STATIC
    ${YUVCONVERT_SOURCE}
    ${YUVCONVERT_INTERFACE}
)

target_include_directories(yuvconvert
  PUBLIC
    include
  PRIVATE
    include/yuvconvert
)

target_link_libraries(yuvconvert
  PUBLIC
    static_math
)

#set_target_properties(yuvconvert PROPERTIES FOLDER "External/yuvconvert")
add_subdirectory(tests)
add_subdirectory(benchmark)
