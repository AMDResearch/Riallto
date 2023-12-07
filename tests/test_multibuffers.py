# Copyright 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import numpy as np

from npu.build.kernel import Kernel
import npu.runtime as ipr
from .test_applications import check_npu, manage_testing
from npu.runtime import AppRunner
from .test_applications import TwoInputsApp, TwoInputsAppMTPassThrough
from npu.lib import BitwiseOr, BitwiseAnd

def test_2in1out(manage_testing):
    """ End-to-end build-and-run test that creates a callgraph with two input buffers going to different kernels. Direct comms CT <-> IT."""
    check_npu()

    two_port = BitwiseOr()
    two_port.tloc = (0,2)

    ubuff0 = np.arange(256,dtype=np.uint8)
    ubuff1 = np.arange(256,dtype=np.uint8)

    trace_app = TwoInputsApp()
    trace_app.build(two_port, ubuff0, ubuff1, ubuff0.nbytes)

    app = AppRunner("TwoInputsApp.xclbin")

    test_data0 = np.ones(256,dtype=np.uint8)
    test_data1 = np.arange(256,dtype=np.uint8)
    bo0_in = app.allocate(shape=(256), dtype=np.uint8)
    bo1_in = app.allocate(shape=(256), dtype=np.uint8)
    bo_out = app.allocate(shape=(256), dtype=np.uint8)

    bo0_in[:] = test_data0
    bo1_in[:] = test_data1
    bo0_in.sync_to_npu()
    bo1_in.sync_to_npu()

    app._refresh_sequence()
    app.call(bo0_in, bo1_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out).reshape(256)

    del app

    assert(np.allclose(test_data0 | test_data1, test_out, atol=0))


def test_2in1out_memtile(manage_testing):
    """ End-to-end build-and-run test that creates a callgraph with two input buffers going to different kernels. Comms via memorytile CT <-> MT <-> IT."""
    check_npu()
    ubuff0 = np.ones(256,dtype=np.uint8)
    ubuff1 = np.arange(256,dtype=np.uint8)

    two_port = BitwiseAnd()
    two_port.tloc = (0,2)

    trace_app = TwoInputsAppMTPassThrough()
    trace_app.build(two_port, ubuff0, ubuff1, ubuff0.nbytes)

    app = AppRunner("TwoInputsAppMTPassThrough.xclbin")

    test_data0 = np.ones(256,dtype=np.uint8)
    test_data1 = np.arange(256,dtype=np.uint8)
    bo0_in = app.allocate(shape=(256), dtype=np.uint8)
    bo1_in = app.allocate(shape=(256), dtype=np.uint8)
    bo_out = app.allocate(shape=(256), dtype=np.uint8)

    bo0_in[:] = test_data0
    bo1_in[:] = test_data1
    bo0_in.sync_to_npu()
    bo1_in.sync_to_npu()

    app._refresh_sequence()
    app.call(bo0_in, bo1_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out).reshape(256)

    del app

    assert(np.allclose(test_data0 & test_data1, test_out, atol=0))
