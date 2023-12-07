# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT


import numpy as np
from .test_applications import check_npu, manage_testing
from npu.runtime import AppRunner
from npu.build.appbuilder import AppBuilder
from npu.build.itkernel import ITWrite
from npu.lib import PlusN
from .test_applications import AppITTiling, ITReadWriteApp, AppITMTInit

def test_itread_build():
    """ Test calling behavioral model using ITReads."""
    imgbuffer = np.zeros(shape=(256), dtype=np.uint8)

    app = ITReadWriteApp()
    assert np.all(app(imgbuffer) == 255)
    app.build(imgbuffer)    

def test_itread_itwrite_tiling_build():
    """ Test calling behavioral model using ITReads."""
    inbuffer = np.zeros(shape=(256,256),dtype=np.uint8)
    outbuffer = np.zeros(shape=(256,256),dtype=np.uint8)

    app = AppITMTInit()
    app(inbuffer, outbuffer)
    assert np.all(outbuffer == 1)
    app.build(inbuffer, outbuffer)   


def test_tiled_output():
    """Examines sequence metadata from the AppBuilder class to ensure tiling is correct."""
    imgin = np.zeros(shape=(540, 15360), dtype=np.uint8)
    imgout = np.zeros(shape=(540, 15360), dtype=np.uint8)

    app = AppITTiling()
    j = app.to_json(imgin, imgout)

    # verify that tiling is incrementing for the ITwrite
    validation = True
    for ix in range(3, len(j['sequence']), 3):
        validation &= j['sequence'][ix-1]['snkoffset'] == 15360*(ix//3 - 1)

    assert validation


def test_2d_interfacetile_transfer(manage_testing):
    """Full end-to-end build-and-run test to ensure that 2D tiling of an image with the interface tile is functioning correctly."""
    check_npu()

    ain = np.zeros(shape=(720, 1280), dtype=np.uint8)
    aout = np.zeros(shape=(720, 1280), dtype=np.uint8)

    v_tile = 48
    h_tile = 10
    tile_width = 1280//h_tile
    tile_height = 720//v_tile
    n = 5

    class App2DSlicing(AppBuilder):
        def __init__(self):
            self.pn = PlusN()
            self.pn.tloc = (0, 2)
            super().__init__()

        def callgraph(self, x, z, n):
            for vt in range(v_tile):
                for ht in range(h_tile):
                    tvi = vt * tile_height
                    thi = ht * tile_width

                    s2d = slice(tvi, tvi+tile_height), slice(thi, thi+tile_width)
                    y = self.pn(x[s2d], x[s2d].nbytes, n)
                    z[s2d] = y

    trace_app = App2DSlicing()
    trace_app.build(ain, aout, n)
    app = AppRunner("App2DSlicing.xclbin")

    test_data = np.zeros(shape=(720, 1280), dtype=np.uint8)
    i = 0
    for vt in range(v_tile):
        for ht in range(h_tile):
            tvi = vt * tile_height
            thi = ht * tile_width
            test_data[tvi:tvi+tile_height, thi:thi+tile_width] = i
            i = (i + 1) % 16

    bo_in = app.allocate(shape=(720, 1280), dtype=np.uint8)
    bo_out = app.allocate(shape=(720, 1280), dtype=np.uint8)

    bo_in[:] = test_data
    bo_in.sync_to_npu()

    app._refresh_sequence()
    app.call(bo_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out).reshape(720, 1280)

    del app
    assert(np.allclose(test_data+n, test_out, atol=0))
