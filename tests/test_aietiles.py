# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
import numpy as np
from npu.build.appbuilder import AppBuilder
from npu.lib import Inverse
from .test_applications import SimpleInvApp, _test_appbuild


def test_simple_inverse_build():
    """ Test the simplest single kernel app."""
    imgbuffer = np.zeros(shape=(256),dtype=np.uint8)

    app = SimpleInvApp()
    app.callgraph(imgbuffer)
    assert np.all(app.callgraph(imgbuffer) == 255)
    app.build(imgbuffer)


def test_callgraph_kern2kern_sharedmem():
    """Tests building an application with kernel to kernel communication via shared memory."""
    array = np.zeros(shape=(15360),dtype=np.uint8)

    class Kern2Kern(AppBuilder):
        def __init__(self):
            self.invs = [Inverse() for _ in range(2)]
            self.invs[0].tloc = (0,2)
            self.invs[1].tloc = (0,3)

            super().__init__()

        def callgraph(self,x):
            x = self.invs[0](x, 15360)
            x = self.invs[1](x, 15360)
            return x

    app = Kern2Kern()

    assert "%tile02 = AIE.tile(0, 2)" in app.to_mlir(array)
    assert "%tile03 = AIE.tile(0, 3)" in app.to_mlir(array)
    _test_appbuild(app, array)

def test_callgraph_kern2kern_dma():
    """Tests building an application with kernel to kernel communication via DMA transfers via stream."""
    array = np.zeros(shape=(15360),dtype=np.uint8)

    class Kern2Kern(AppBuilder):
        def __init__(self):
            self.invs = [Inverse() for _ in range(2)]
            self.invs[0].tloc = (0,2)
            self.invs[1].tloc = (0,5)

            super().__init__()

        def callgraph(self,x):
            x = self.invs[0](x, 15360)
            x = self.invs[1](x, 15360)
            return x

    app = Kern2Kern()

    assert "%tile02 = AIE.tile(0, 2)" in app.to_mlir(array)
    assert "%tile05 = AIE.tile(0, 5)" in app.to_mlir(array)
    _test_appbuild(app, array)
