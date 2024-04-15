# Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import os
from npu.build.kernel import Kernel

gray_out_src = '''
#include "kernels.hpp"

extern "C" {

#define N 720

void passthrough(uint8_t *in_buffer, uint8_t *out_buffer) {
    uint8_t buffer[N];
    rgba2gray_aie(in_buffer, buffer, N*4);
    gray2rgba_aie(buffer, out_buffer, N);
}

} // extern "C"
'''


def function_behavior(invobj):
    invobj.out_buffer.array = invobj.in_buffer.array


def test_superkernel_build():
    krnobj = Kernel(gray_out_src)
    krnobj.behavioralfx = function_behavior
    # lunch build and then assert
    assert (objfile := krnobj.objfile)
    # remove header files
    os.remove(objfile)
