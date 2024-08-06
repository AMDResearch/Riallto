# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT


import numpy as np
from npu.build.appbuilder import AppBuilder
from npu.lib import PlusN

from .test_applications import check_npu, manage_testing
from npu.runtime import AppRunner
from npu.build.itkernel import ITWrite

def test_slicing(manage_testing):
    """ End-to-end build-and-run test where only a slice of the input is set via IT. """
    check_npu()

    ubuff_in = np.arange(256, dtype=np.uint8)
    ubuff_out = np.arange(256, dtype=np.uint8)

    n = 5

    class SimpleAppSlicing(AppBuilder):
        def callgraph(self,x,z):
            z[32:64] = PlusN(x[0:32], 32, n)

    app = SimpleAppSlicing()

    app.build(ubuff_in, ubuff_out)

    app = AppRunner("SimpleAppSlicing.xclbin")

    bo_in = app.allocate(shape=(256), dtype=np.uint8)
    bo_out = app.allocate(shape=(256), dtype=np.uint8)

    bo_in[:] = ubuff_in
    bo_in.sync_to_npu()

    app._refresh_sequence()
    app.call(bo_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out).reshape(256)

    del app
    assert np.array_equal(test_out[32:64], ubuff_in[0:32] + 5)
