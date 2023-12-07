# Copyright 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import numpy as np
from npu.build.kernel import Kernel
import npu.runtime as ipr
from .test_applications import check_npu, manage_testing
from .test_applications import TwoOutputsAppMTPassThrough
from npu.runtime import AppRunner

ident_src = '''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

extern "C" {

void ident(uint8_t *in_buffer, uint8_t *out_buffer, uint32_t nbytes) {
    ::aie::vector<uint8_t, 32> buffer0;
    uint16_t loop_count = nbytes >> 5;
    for(int j=0; j<loop_count; j++) {
        buffer0 = ::aie::load_v<32>(in_buffer);
        in_buffer += 32;
        ::aie::store_v((uint8_t*)out_buffer, buffer0);
        out_buffer += 32;
    }
}

} // extern "C"
'''

def test_1in2out(manage_testing):
    """ End-to-end build-and-run test that builds a callgraph with one input buffer, broadcasted to two separate kernels, and one output buffer (MTConcat) in memtile."""
    check_npu()

    ident0 = Kernel(ident_src)
    ident1 = Kernel(ident_src)

    def ident_behavior(invobj):
        invobj.out_buffer.array = invobj.in_buffer.array

    ident0.behavioralfx = ident_behavior
    ident1.behavioralfx = ident_behavior

    ubuff0 = np.arange(256,dtype=np.uint8)

    trace_app = TwoOutputsAppMTPassThrough(ident0, ident1)
    trace_app.build(ubuff0)

    app = AppRunner("TwoOutputsAppMTPassThrough.xclbin")

    test_data = np.ones(256,dtype=np.uint8)
    bo0_in = app.allocate(shape=(256), dtype=np.uint8)
    bo_out0 = app.allocate(shape=(256), dtype=np.uint8)
    bo_out1 = app.allocate(shape=(256), dtype=np.uint8)

    bo0_in[:] = test_data
    bo0_in.sync_to_npu()

    app._refresh_sequence()
    app.call(bo0_in, bo_out0, bo_out1)

    bo_out0.sync_from_npu()
    test_out0 = np.array(bo_out0).reshape(256)

    bo_out1.sync_from_npu()
    test_out1 = np.array(bo_out1).reshape(256)

    del app

    assert(np.allclose(test_data, test_out0, atol=0))
    assert(np.allclose(test_data, test_out1, atol=0))
