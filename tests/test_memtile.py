# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import numpy as np
from npu.utils.test_device import get_device_status
import npu.runtime as ipr
from npu.runtime import AppRunner

from .test_applications import manage_testing, _test_appbuild, check_npu
from .test_applications import AppMTInPlusN, MtSplitConcat4AIEsPlusN, AppMTBroadcastConcat
from .test_applications import AppITTilingMTIoPlusN, SimpleInvAppMTPassThrough, AppMTSplitConcatInit
from .test_applications import SimpleMTPassThroughApplication
import pytest
from npu.build.appbuilder import AppBuilder
from npu.build.mtkernel import MTPassThrough
from npu.build.mtkernel import MTPassThrough


def test_mtkernel_passthrough_simple_behavioral_build():
    """ Test the simplest single kernel app with mtpassthrough."""
    imgbuffer = np.zeros(shape=(256),dtype=np.uint8)

    app = SimpleInvAppMTPassThrough()
    app.callgraph(imgbuffer)
    assert np.all(app.callgraph(imgbuffer) == 255)
    app.build(imgbuffer)


def test_mtkernel_passthrough_behavioral():
    """ Behavioral test of using MTPassThrough in a simple callgraph. """
    imgin = np.zeros(shape=(256,2),dtype=np.uint8)
    imgout = np.zeros(shape=(256,2),dtype=np.uint8)

    app = AppITTilingMTIoPlusN()
    app(imgin, imgout, 5)
    assert np.all(imgout == 5)


def test_mtkernel_splitconcat_behavioral():
    """ Behavioral test of using MTSplit/MTConcat. """
    imgin = np.zeros(shape=(256),dtype=np.uint8)

    app = MtSplitConcat4AIEsPlusN()
    imgout = app(imgin,7)
    assert np.all(imgout == 7)


def test_plusn_via_memtile(manage_testing):
    """End-to-end build-and-run test that uses MTPassThrough and PlusN kernel"""
    check_npu()
    array = np.zeros(shape=(256),dtype=np.uint8)

    n = 5

    trace_app = AppMTInPlusN()
    trace_app.build(array, n)

    app = AppRunner("AppMTInPlusN.xclbin")

    test_data = np.random.randint(0,255,256, dtype=np.uint8)
    bo_in = app.allocate(shape=(256), dtype=np.uint8)
    bo_out = app.allocate(shape=(256), dtype=np.uint8)

    bo_in[:] = test_data
    bo_in.sync_to_npu()

    app._refresh_sequence()
    app.call(bo_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out).reshape(256)

    print(np.allclose(test_data+n,test_out, atol=0))
    print(f"{test_data=}")
    print(f"{test_out=}")

    del app

    assert(np.allclose(test_data+n,test_out, atol=0))

@pytest.mark.parametrize('size', [256, 15360])
def test_memtile_distribute_join_4(size):
    """End-to-end build-and-run test that uses MTSplit/MTConcat with 4 kernels

    It uses the PlusN kernel, and we test sizes [256, 15360])
    """

    check_npu()

    tile_width = size
    n = 5

    array = np.zeros(shape=(4,tile_width),dtype=np.uint8)

    trace_app = MtSplitConcat4AIEsPlusN()
    trace_app.build(array,n)
    app = AppRunner("MtSplitConcat4AIEsPlusN.xclbin")

    test_data = np.random.randint(0,255,size=(4,tile_width), dtype=np.uint8)
    bo_in = app.allocate(shape=(4,tile_width), dtype=np.uint8)
    bo_out = app.allocate(shape=(4,tile_width), dtype=np.uint8)

    bo_in[:] = test_data
    bo_in.sync_to_npu()

    app._refresh_sequence()
    app.call(bo_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out).reshape(4,tile_width)

    print(np.allclose(test_data+n,test_out, atol=0))
    print(f"{test_data=}")
    print(f"{test_out=}")

    del app

    assert(np.allclose(test_data+n,test_out, atol=0))


def test_callgraph_memtile_broadcast():
    """Build test for a broadcast via the memtile"""
    array = np.zeros(shape=(15360,),dtype=np.uint8)

    app = AppMTBroadcastConcat()
    _test_appbuild(app, array)


def test_cgv3_mtpassthrough():
    """ Build test for MTPassTrhough with those kernels in the AppBuild init. """

    app = SimpleMTPassThroughApplication()

    x_in = np.zeros(shape=(720, 1280*4), dtype=np.uint8)
    x_out = np.zeros(shape=(720, 1280*4), dtype=np.uint8)

    app.build(x_in, x_out, debug=True)


def test_cgv3_mtsplitconcatinit():
    """ Build test for MTSplitConcat with those kernels in the AppBuild init. """
    imgbuffer = np.zeros(shape=(4,64),dtype=np.uint8)
    outbuffer = np.zeros(shape=(4,64),dtype=np.uint8)

    app = AppMTSplitConcatInit()

    app.callgraph(imgbuffer, outbuffer)
    assert np.all(outbuffer == 255)
    app.to_mlir(imgbuffer, outbuffer)
    app.build(imgbuffer, outbuffer)
