# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npu.build.appbuilder import AppBuilder
from npu.lib import Inverse
from npu.lib import Plus1
from npu.lib import Gray2Rgba
from npu.build.mtkernel import MTPassThrough, MTConcat, MTSplit
from npu.build.itkernel import ITWrite, ITRead
from .test_applications import check_npu, manage_testing, _test_appbuild


def test_callgraph_inverse_behavioral():
    """ Test calling behavioral model of inverse model and validates output."""
    imgbuffer = np.zeros(shape=(256), dtype=np.uint8)

    class SimpleApp(AppBuilder):
        def callgraph(self, x):
            return Inverse(x, x.nbytes)

    app = SimpleApp()
    assert np.all(app.callgraph(imgbuffer) == 255)


def test_callgraph_slicing_behavioral():
    """Test slicing within the callgraph calling the inverse behavioral."""
    imgbuffer = np.zeros(shape=(32), dtype=np.uint8)

    class SimpleAppSlicing(AppBuilder):
        def callgraph(self, x):
            return Inverse(x[8:16], 8)

    app = SimpleAppSlicing()
    imgbuffer[16:24] = app.callgraph(imgbuffer)

    assert np.all(imgbuffer[16:24] == 255)
    assert np.all(imgbuffer[:16] == 0)
    assert np.all(imgbuffer[24:32] == 0)


def test_callgraph_portslicing_behavioral():
    """Test behavioral slicing of the input over 4 CTs using the inverse"""

    # Memory tile shared buffer
    bigger_3d_array = np.zeros(shape=(4, 4, 4096), dtype=np.uint8)

    class MtSplitConcat4AIEsInverse(AppBuilder):
        def callgraph(self, x):
            xs = MTSplit(x, 4)
            xs = [k(x, 4 * 4096)
                  for k, x in zip([Inverse() for _ in range(4)], xs)]
            x = MTConcat(xs)
            return x

    app = MtSplitConcat4AIEsInverse()
    assert np.all(app.callgraph(bigger_3d_array) == 255)


def test_callgraph_kernelmultiuse_behavioral():
    """Tests behavioral model across a range of different input sizes."""

    # Memory tile shared buffer
    small_array = np.zeros(shape=(4, 4, 4096), dtype=np.uint8)
    bigger_array = np.zeros(shape=(8, 4, 4096), dtype=np.uint8)
    biggest_array = np.zeros(shape=(16, 4, 4096), dtype=np.uint8)

    class SimpleApp(AppBuilder):
        def callgraph(self, x):
            return Inverse(x, x.nbytes)

    'Call the callgraph x3 with x3 size buffers'
    for a in [small_array, bigger_array, biggest_array]:
        app = SimpleApp()
        b = app.callgraph(a)
        assert np.all(b == 255)


def test_callgraph_kern2kern_behavioral():
    """Test behavioral model of kernel-to-kernel communication using shared memory."""
    int8buffer = np.zeros(shape=(100,16384),dtype=np.uint8)
    outbuffer = np.zeros(shape=(100,16384),dtype=np.uint8)

    class Kern2KernPlus1(AppBuilder):
        def __init__(self):
            self.plusones = [Plus1() for _ in range(2)]
            super().__init__()

        def callgraph(self,xin,xout):
            for t in range(100):
                x = self.plusones[0](xin[t], 16384)
                x = self.plusones[1](x, 16384)
                xout[t] = x
        
    app = Kern2KernPlus1()
    app.callgraph(int8buffer, outbuffer)
    assert np.all(outbuffer == 2)   


def test_callgraph_appname():
    """Test callgraph metadata for naming."""
    imgbuffer = np.zeros(shape=(256), dtype=np.uint8)

    class SimpleApp(AppBuilder):
        def callgraph(self, x):
            return Inverse(x, 64 * 4)

    app = SimpleApp()

    assert app.name == 'SimpleApp'
    assert app.to_json(imgbuffer)['application'] == 'SimpleApp'


def test_callgraph_kernelio_resizing():
    """Test behavioral callgraph were we have a kernel that have different
    I/O nbytes
    """

    graybuffer = np.zeros(shape=(256), dtype=np.uint8)

    class SimpleGray2RGBA(AppBuilder):
        def callgraph(self, x):
            return Gray2Rgba(x, 64 * 4)

    app = SimpleGray2RGBA()

    rgba = app.callgraph(graybuffer)
    assert rgba.nbytes == graybuffer.nbytes * 4


def test_callgraph_inverse_itread_behavioral():
    """ Test calling behavioral model using ITReads."""
    imgbuffer = np.zeros(shape=(256), dtype=np.uint8)

    class SimpleApp(AppBuilder):
        def callgraph(self, x):
            x = ITRead(x)
            x = Inverse(x, x.nbytes)
            x = ITWrite(x)
            return x

    app = SimpleApp()
    assert np.all(app.callgraph(imgbuffer) == 255)


def _test_callgraph_singlekernel_build(krn, shape=(1024,)):
    """Behavioral test of a single kernel that has nbytes as RTPs."""

    imgbuffer = np.zeros(shape=shape, dtype=np.uint8)

    class SingleBuffer(AppBuilder):
        def callgraph(self, x):
            return krn(x)

    app = SingleBuffer()
    app.to_json(imgbuffer)

    _test_appbuild(app, imgbuffer)


def _test_callgraph_singlekernel_nbytes_build(k):
    """Behavioral test of a single kernel that has nbytes as RTPs."""

    imgbuffer = np.zeros(shape=(1024),dtype=np.uint8)

    class SingleBufferOneRtp(AppBuilder):
        def callgraph(self, x):
            return k(x, x.nbytes)

    app = SingleBufferOneRtp()
    app.to_json(imgbuffer)

    _test_appbuild(app, imgbuffer)


def _test_callgraph_2in1out_singlekernel_build(k):
    """Behavioral test of a callgraph that takes two input buffers and
    returns a single output buffer.
    """

    imgbuffer = np.zeros(8192, dtype=np.uint8)

    class Kern_2in1out(AppBuilder):
        def callgraph(self, a, b):
            return k(a, b)

    app = Kern_2in1out()
    app.to_json(imgbuffer)

    _test_appbuild(app, imgbuffer)


def _test_callgraph_1in1outNrtps_singlekernel_build(k, nrtps):
    """Behavioral tests of a kernel that has N RTPs"""

    imgbuffer = np.zeros(8192, dtype=np.uint8)
    args = range(nrtps)

    class Kern_1in1outNrtps(AppBuilder):
        def callgraph(self, a):
            return k(a, a.nbytes, *args)

    app = Kern_1in1outNrtps()
    app.to_json(imgbuffer)

    _test_appbuild(app, imgbuffer)


def _test_callgraph_2in1outNrtps_singlekernel_build(k, nrtps):
    """behavioral test of a kernel that has 2 input buffers, 1 output buffer,
    and N RTPs.
    """

    imgbuffer = np.zeros(8192, dtype=np.uint8)
    args = range(nrtps)

    class Kern_2in1outNrtps(AppBuilder):
        def callgraph(self, a, b):
            return k(a, b, a.nbytes, *args)

    app = Kern_2in1outNrtps()
    app.to_json(imgbuffer, imgbuffer)

    _test_appbuild(app, imgbuffer, imgbuffer)


def _test_callgraph_3in1outNrtps_singlekernel_build(k, nrtps):
    """behavioral test of a kernel that has 3 input buffers,
    1 output buffer, and N RTPs.
    """

    imgbuffer = np.zeros(1024, dtype=np.uint8)
    args = range(nrtps)

    class Kern_3in1outNrtps(AppBuilder):
        def callgraph(self, a, b, c):
            return k(a, b, c, *args)

    app = Kern_3in1outNrtps()
    app.to_json(imgbuffer, imgbuffer, imgbuffer)

    _test_appbuild(app, imgbuffer, imgbuffer, imgbuffer)
