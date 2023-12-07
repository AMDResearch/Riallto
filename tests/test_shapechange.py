# Copyright 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import numpy as np
from npu.build.appbuilder import AppBuilder
from .test_applications import check_npu, manage_testing
from npu.runtime import AppRunner
from npu.lib import Rgba2Gray, Gray2Rgba


def test_rgba2gray_resize(manage_testing):
    """ End-to-end build-and-run test for Rgba2Gray kernel

    Input data is 1024-Byte in and output data is 256-Byte.
    """
    check_npu()

    array_in = np.arange(1024, dtype=np.uint8)

    class SimpleApp(AppBuilder):
        def callgraph(self, x):
            return Rgba2Gray(x, x.nbytes)

    trace_app = SimpleApp()
    trace_app.build(array_in)

    app = AppRunner("SimpleApp.xclbin")

    test_data = np.ones(1024, dtype=np.uint8)
    bo_in = app.allocate(shape=(1024), dtype=np.uint8)
    bo_out = app.allocate(shape=(256), dtype=np.uint8)

    bo_in[:] = test_data
    bo_in.sync_to_npu()

    app._refresh_sequence()
    app.call(bo_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out).reshape(256)

    del app


def test_gray2rgba_resize(manage_testing):
    """ End-to-end build-and-run test for Gray2Rgba kernel

    Input data is 256-Byte and output data 1024-Byte.
    """
    check_npu()

    array_in = np.arange(256, dtype=np.uint8)

    class SimpleApp(AppBuilder):
        def callgraph(self, x):
            return Gray2Rgba(x, x.nbytes)

    trace_app = SimpleApp()
    trace_app.build(array_in)

    app = AppRunner("SimpleApp.xclbin")

    bo_in = app.allocate(shape=(256), dtype=np.uint8)
    bo_out = app.allocate(shape=(1024), dtype=np.uint8)

    bo_in[:] = array_in
    bo_in.sync_to_npu()

    app._refresh_sequence()
    app.call(bo_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out).reshape(256, 4)

    del app
    reshaped = np.zeros((4, 256), dtype=np.uint8)
    for i in range(4):
        reshaped[i] = test_out[:, i]

    assert np.array_equal(reshaped[0], array_in) and \
        np.array_equal(reshaped[1], array_in) and \
        np.array_equal(reshaped[2], array_in)
