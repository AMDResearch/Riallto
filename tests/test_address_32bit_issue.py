# Copyright 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import numpy as np
import pytest

from npu.build.kernel import Kernel
from npu.build.appbuilder import AppBuilder
from .test_applications import check_npu, manage_testing
from npu.runtime import AppRunner

def test_32bit_transfers(manage_testing):
    """ test to check if 32bit transfers are aligned correctly. """
    check_npu()

    set_value_src= """
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

extern "C" {

void set_value(uint32_t *in_buffer, uint32_t *out_buffer){
    for(int i =0; i<16; i++) {
        out_buffer[i] = i;
    }
}

} // extern C
    """

    def setval_behavioral(obj):
         obj.out_buffer.array = np.array([range(16)])

    set_value = Kernel(set_value_src,setval_behavioral)

    A_buff = np.zeros(shape=(4,4), dtype=np.uint32)
    B_buff = np.zeros(shape=(4,4), dtype=np.uint32)

    set_value.tloc = (0,2)

    class SimpleInt32App(AppBuilder):
        def callgraph(self, x):
            return set_value(x)

    trace_app = SimpleInt32App()
    trace_app.build(A_buff)

    A = np.random.randint(0, 256, (4, 4), dtype=np.uint32)
    B = np.random.randint(0, 256, (4, 4), dtype=np.uint32)

    app = AppRunner('SimpleInt32App.xclbin')
    A_in = app.allocate(shape=(4, 4), dtype=np.uint32)
    B_out = app.allocate(shape=(4, 4), dtype=np.uint32)

    A_in[:] = A
    A_in.sync_to_npu()

    app.call(A_in, B_out)

    B_out.sync_from_npu()
    B_npu = np.array(B_out[:])

    del A_in
    del B_out
    del app

    golden_output = np.arange(0, 16, dtype=np.uint32).reshape(4, 4)

    if not np.allclose(B_npu, golden_output, atol=0):
        print(f"{B_npu=} exptecting={golden_output}")
        raise RuntimeError("Test failed")
