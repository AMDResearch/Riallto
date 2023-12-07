# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npu.build.appbuilder import AppBuilder
from npu.lib import PlusN
from npu.build.mtkernel import MTPassThrough, MTConcat, MTSplit

from .test_applications import check_npu, manage_testing
from npu.runtime import AppRunner

def test_broadcast(manage_testing):
    """Tests broadcasting the same data from the MT to multiple kernels."""
    check_npu()
    array_in = np.zeros(32, dtype=np.uint8)
    n = 0

    class MTPassThroughBroadcastConcat4AIEsPlusN(AppBuilder):
        def callgraph(self,x):
            x = MTPassThrough(x)
            xs = [k(x, x.nbytes, n) for k in [PlusN() for _ in range(4)]]
            x = MTConcat(xs)
            return x

    trace_app = MTPassThroughBroadcastConcat4AIEsPlusN()
    trace_app.build(array_in)
    app = AppRunner("MTPassThroughBroadcastConcat4AIEsPlusN.xclbin")

    test_data = np.ones(shape=(32), dtype=np.uint8)

    bo_in = app.allocate(shape=(32), dtype=np.uint8)
    bo_out = app.allocate(shape=(4,32), dtype=np.uint8)

    bo_in[:] = test_data
    bo_in.sync_to_npu()

    app._refresh_sequence()
    app.call(bo_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out).reshape(4,32)

    del app

    for i in range(4):
        if not np.array_equal(test_out[i], test_data):
            raise RuntimeError(f"Data mismatch! {test_data=}  {test_out[i]=}")

def test_broadcast_nomemtile_in(manage_testing):
    """Tests broadcasting the same data to two different tiles directly from IT without MT."""
    check_npu()

    array_in = np.zeros(32, dtype=np.uint8)
    n = 0

    class ITBroadcastMTConcatPlusN(AppBuilder):
        def callgraph(self,x):
            xs = [k(x, x.nbytes, n) for k in [PlusN() for _ in range(4)]]
            x = MTConcat(xs)
            return x

    trace_app = ITBroadcastMTConcatPlusN()
    trace_app.build(array_in, debug=True)
    app = AppRunner("ITBroadcastMTConcatPlusN.xclbin")

    test_data = np.ones(shape=(32), dtype=np.uint8)

    bo_in = app.allocate(shape=(32), dtype=np.uint8)
    bo_out = app.allocate(shape=(4,32), dtype=np.uint8)

    bo_in[:] = test_data
    bo_in.sync_to_npu()

    app._refresh_sequence()
    app.call(bo_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out).reshape(4, 32)

    del app

    for i in range(4):
        if not np.array_equal(test_out[i], test_data):
            raise RuntimeError(f"Data mismatch! {test_data=}  {test_out[i]=}")
