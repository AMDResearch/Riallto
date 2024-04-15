# Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import os
from npu.build.kernel import Kernel
from .test_applications import check_npu

gray_out_src = '''
#include "kernels.hpp"

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
#include "kernels.hpp"

extern "C" {

#define N 720

void colordetect(uint8_t *in_buffer, uint8_t *out_buffer) {
    uint8_t rgba2hue_buff[N];
    uint8_t in_range_buff[N];
    uint8_t gray2rgba_buff[N];

    rgba2hue_aie(in_buffer, rgba2hue_buff, N*4);
    in_range_aie(rgba2hue_buff, in_range_buff, N, 50, 151);
    gray2rgba_aie(in_range_buff, gray2rgba_buff, N);
    bitwiseAND_aie(in_buffer, gray2rgba_buff, out_buffer, N);
}

} // extern "C"
'''


def function_behavior(invobj):
    invobj.out_buffer.array = invobj.in_buffer.array


@pytest.mark.parametrize('superkernel', ['gray_out_src', 'color_detect_src'])
def test_superkernel_build(superkernel):
    check_npu()
    krnobj = Kernel(eval(superkernel))
    krnobj.behavioralfx = function_behavior
    # lunch build and then assert
    assert (objfile := krnobj.objfile)
    #_test_callgraph_singlekernel_nbytes_build(krnobj)
    # remove object file
    os.remove(objfile)
