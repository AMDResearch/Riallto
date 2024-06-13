# Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import os
from npu.build.kernel import Kernel
from .test_applications import check_npu
from .test_callgraphs import _test_callgraph_singlekernel_build

gray_out_src = '''
#include "gray2rgba.cpp"
#include "rgba2gray.cpp"

extern "C" {

#define N 720

void grayout(uint8_t *in_buffer, uint8_t *out_buffer) {
    uint8_t buffer[N];
    rgba2gray_aie(in_buffer, buffer, N*4);
    gray2rgba_aie(buffer, out_buffer, N);
}

} // extern "C"
'''

color_detect_src = '''
#include "rgba2hue.cpp"
#include "inrange.cpp"
#include "gray2rgba.cpp"
#include "bitwiseAnd.cpp"

extern "C" {

#define N 720

void colordetect(uint8_t *in_buffer, uint8_t *out_buffer) {
    uint8_t rgba2hue_buff[N];
    uint8_t in_range_buff[N];
    uint8_t gray2rgba_buff[N];

    rgba2hue_aie(in_buffer, rgba2hue_buff, N*4);
    in_range_aie(rgba2hue_buff, in_range_buff, N, 50, 151);
    gray2rgba_aie(in_range_buff, gray2rgba_buff, N);
    bitwiseAND_aie<uint8_t, 64>(in_buffer, gray2rgba_buff, out_buffer, N*4);
}

} // extern "C"
'''

edge_overlaid_src = '''
#include "rgba2gray.cpp"
#include "gray2rgba.cpp"
#include "filter2d_720p.cpp"
#include "addWeighted.cpp"

extern "C" {

#define N 720

void edge(uint8_t *in_buffer, uint8_t *out_buffer) {
    uint8_t graybuffer[N];
    uint8_t f2dbuffer[N];
    uint8_t rgbabuffer[N*4];

    rgba2gray_aie(in_buffer, graybuffer, N*4);
    filter2d_720p_aie(graybuffer, f2dbuffer, 0, -4096, 0, -4096, 16384, -4096, 0, -4096, 0);
    gray2rgba_aie(f2dbuffer, rgbabuffer, N);
    addweighted_aie<uint8_t, 64, UINT8_MAX>(in_buffer, rgbabuffer, out_buffer, N*4, 0, 0, 0);
}

} // extern "C"
'''

lib_src = '''
#include "plus1.cpp"
#include "plusn.cpp"

extern "C" {

#define N 720

void pluschain(uint8_t *in_buffer, uint8_t *out_buffer) {
    uint8_t outbuf[N];
    plusone_aie(in_buffer, outbuf, N);
    plusn_aie(outbuf, outbuf, N, 2);
}

} // extern "C"
'''

def function_behavior(invobj):
    invobj.out_buffer.array = invobj.in_buffer.array


@pytest.mark.parametrize('kernel_fusion', ['gray_out_src', 'color_detect_src',
                                           'edge_overlaid_src', 'lib_src'])

def test_kernel_fusion_build(kernel_fusion):
    check_npu()
    krnobj = Kernel(eval(kernel_fusion))
    krnobj.behavioralfx = function_behavior
    # lunch build and then assert
    assert (objfile := krnobj.objfile)
    _test_callgraph_singlekernel_build(krnobj)
    # remove object file
    os.remove(objfile)
