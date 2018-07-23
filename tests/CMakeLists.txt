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

include(Unittests)

add_unit_test_suite(
    TARGET test_yuvconvert
    SOURCES
        test_rgb2yuv.cpp
        test_common.cpp
        test_utilities.h
        test_quality.cpp
    INCLUDES
        ${CMAKE_CURRENT_BINARY_DIR}
    LIBRARIES yuvconvert CxImage fmt
    FOLDER tests/yuvconvert
)


target_compile_definitions(test_yuvconvert
  PRIVATE
    NOMINMAX
    _UNICODE
    UNICODE
)

#set_target_properties(test_cam_encoder PROPERTIES
#    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
#    VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$(Configuration)
#)

#--gtest_catch_exceptions=0