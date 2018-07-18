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

set(YUVCONVERT_SOURCE
    src/to_420.cpp
    src/to_420.h
    src/to_420_c.cpp
    src/to_420_c.h
    src/to_420_ssse3.cpp
    src/to_420_ssse3.h
    src/yuv_common.h
    include/yuvconvert.h
)

source_group(yuvconvert FILES
    ${YUVCONVERT_SOURCE}
)

add_library(yuvconvert STATIC
    ${YUVCONVERT_SOURCE}
)

target_include_directories(yuvconvert
  PUBLIC
    include
  PRIVATE
    include/yuvconverte
)

#set_target_properties(yuvconvert PROPERTIES FOLDER "External/yuvconvert")
add_subdirectory(benchmark)
