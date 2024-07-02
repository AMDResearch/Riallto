# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import inspect
import pytest
import os
import shutil
import numpy as np
from pathlib import Path
from npu.utils.test_device import get_device_status, reset_npu
from npu.utils.xbutil import XBUtil
import platform

from npu.build.appbuilder import AppBuilder
from npu.build.kernelbuilder import KernelObjectBuilder
from npu.lib import Inverse
from npu.lib import PlusN
from npu.lib import RgbaInverse
from npu.build.mtkernel import MTPassThrough, MTConcat, MTSplit
from npu.build.itkernel import ITWrite, ITRead

from npu.lib import RgbaRtpThres
from npu.lib import Gray2Rgba


class SimpleInvApp(AppBuilder):
    """Callgraph to test inverse in simplest form"""
    def callgraph(self, x):
        return Inverse(x, x.nbytes)


class SimpleInvAppMTPassThrough(AppBuilder):
    """Callgraph to test MTPassThrough in simplest form"""
    def callgraph(self, x):
        x = MTPassThrough(x)
        x = Inverse(x, x.nbytes)
        x = MTPassThrough(x)
        return x


class SingleKernelCall(AppBuilder):
    """Application class for testing a single kernel call."""
    def __init__(self, k):
        self.k = k
        super().__init__()

    def callgraph(self, xin, xout):
        """ Callgraph that call the kernel once"""
        xout[:] = self.k(xin)

class SimpleKernelITTiling(AppBuilder):
    """Application class for testing tiling on a single kernel."""
    def __init__(self, k):
        self.k = k
        super().__init__()

    def callgraph(self, xin, xout):
        """ Callgraph that tiles an input x_in, passes the tiles into a kernel, and outputs the tile."""
        for t in range(xin.shape[0]):
            y = self.k(xin[t])
            xout[t] = y


class SimplePlusN(AppBuilder):
    def callgraph(self, x, n):
        """Callgraph that calls the simple PlusN kernel, where n is an RTP"""
        return PlusN(x, x.nbytes, n)


class AppMTInPlusN(AppBuilder):
    def callgraph(self, x, n):
        """Callgraph that calls the PlusN kernel, where n is an RTP,
        and ingress data is passed through an MTPassThrough"""
        x = MTPassThrough(x)
        x = PlusN(x, x.nbytes, n)
        return x


class AppITTilingMTIoPlusN(AppBuilder):
    def __init__(self):
        self.pn = PlusN()
        self.mtbuffer_in = MTPassThrough()
        self.mtbuffer_out = MTPassThrough()
        super().__init__()

    def callgraph(self, xin, xout, n):
        """Callgraph that calls the PlusN kernel, where n is an RTP,
        tiles the data, and ingress/egress data is passed through MTPassThrough"""
        for t in range(xin.shape[0]):
            x = self.mtbuffer_in(xin[t])
            x = self.pn(x, x.nbytes, n)
            x = self.mtbuffer_out(x)
            xout[t] = x


class AppITTilingPlusN(AppBuilder):
    def __init__(self):
        self.pn = PlusN()
        super().__init__()

    def callgraph(self, xin, xout, n):
        """Callgraph that calls the PlusN kernel, where n is an RTP,
        and tiles the data"""
        for t in range(xin.shape[0]):
            y = self.pn(xin[t], xin[t].nbytes, n)
            xout[t] = y


class AppITTiling(AppBuilder):
    def __init__(self):
        self.inv = Inverse()
        super().__init__()

    def callgraph(self, xin, xout):
        """Callgraph that calls the Inverse kernel and tiles the data"""
        for t in range(xin.shape[0]):
            xout[t] = self.inv(xin[t], xin[t].nbytes)


class MtSplitConcat4AIEsPlusN(AppBuilder):
    def callgraph(self, x, n):
        """ Callgraph that calls the PlusN kernel on 4 CTs, uses MTSplit to distribute
        the data over the 4 CTs, and MTConcat to join the data"""
        xs = MTSplit(x, 4)
        xs = [k(x, x.nbytes, n)
              for k, x in zip([PlusN() for _ in range(4)], xs)]
        x = MTConcat(xs)
        return x


class TwoInputsApp(AppBuilder):
    def callgraph(self, k, a, b, size):
        """Callgraph that tests a kernel with two input arguments """
        return k(a, b, size)


class TwoInputsAppMTPassThrough(AppBuilder):
    def callgraph(self, k, a, b, size):
        """Callgraph that tests a kernel with two input arguments, where data ingress
        uses an MTPassThrough."""
        a = MTPassThrough(a)
        b = MTPassThrough(b)
        return k(a, b, size)


class TwoOutputsAppMTPassThrough(AppBuilder):
    def __init__(self, k0, k1):
        self.k0 = k0
        self.k1 = k1
        super().__init__()

    def callgraph(self, x):
        """Callgraph that tests two outputs from an application."""
        a = self.k0(x, x.nbytes)
        b = self.k1(x, x.nbytes)
        a = MTPassThrough(a)
        b = MTPassThrough(b)

        return a, b


class AppITTilingMTTilingRgbaRtpThres(AppBuilder):
    def __init__(self):
        super().__init__()
        self.rgbakerns = [RgbaRtpThres() for _ in range(4)]
        self.mtbsplit = MTSplit(4)
        self.mtbconcat = MTConcat()

    def callgraph(self,xin,xout, nbytes, rthreshes, gthreshes, bthreshes):
        """ A more complicated callgraph test, that tests 4CT RGBA threshold kernels,
        where thresholds are configurable using RTPs, data is tiles, and MTSplit/MTConcat
        are used for data distribution and joining."""
        for ix, _ in enumerate(xin):
            xs = self.mtbsplit(xin[ix])
            xs = [ self.rgbakerns[i](xs[i], nbytes,
                    rthreshes[i], gthreshes[i], bthreshes[i]) for i in range(4)]
            xout[ix] = self.mtbconcat(xs)

class Kern2KernDirectMemITTIlingRGBAInverse(AppBuilder):
    def __init__(self):
        self.rgbainvs = [RgbaInverse() for _ in range(2)]
        self.rgbainvs[0].tloc = (0, 2)
        self.rgbainvs[1].tloc = (0, 3)
        super().__init__()

    def callgraph(self, xin, xout):
        """Callgraph that performs a double inverse on tiled data passed to it."""
        for t in range(xin.shape[0]):
            x = self.rgbainvs[0](xin[t], xin[t].nbytes)
            x = self.rgbainvs[1](x, x.nbytes)
            xout[t] = x


class SimpleGray2RGBA(AppBuilder):
    def callgraph(self, x):
        """Callgraph to test the Gray2Rgba kernel where input and output shapes change."""
        return Gray2Rgba(x, 64*4)


class AppMTBroadcastConcat(AppBuilder):
    def callgraph(self, x):
        """Callgraph to test broadcast of data from the MemoryTile"""
        x = MTPassThrough(x)
        xs = [Inverse(x, x.nbytes) for _ in range(4)]
        return MTConcat(xs)


class ITReadWriteApp(AppBuilder):
    """Callgraph to test use of ITRead and ITWrite"""
    def callgraph(self, x):
        x = ITRead(x)
        x = Inverse(x, x.nbytes)
        x = ITWrite(x)
        return x

class SimpleMTPassThroughApplication(AppBuilder):
    """Callgraph to test use of MTPassThrough in init"""
    def __init__(self):
        self.pn = PlusN()
        self.mtbi = MTPassThrough()
        self.mtbo = MTPassThrough()
        super().__init__()

    def callgraph(self, x_in:np.ndarray, x_out:np.ndarray)->None:
        for t in range(x_in.shape[0]):
            x = self.mtbi(x_in[t])
            x = self.pn(x, x.nbytes, 1)
            x = self.mtbo(x)
            x_out[t] = x

class AppMTSplitConcatInit(AppBuilder):
    """Callgraph to test use of MTSplit and MTConcat in init"""
    def __init__(self):
        super().__init__()
        self.invs = [Inverse() for _ in range(4)]
        self.mtbsplit = MTSplit(4)
        self.mtbconcat = MTConcat()

    def callgraph(self, xin, xout):
        for ix, _ in enumerate(xin):
            xs = self.mtbsplit(xin[ix])
            xs = [i(x,x.nbytes) for i,x in zip(self.invs, xs)]
            xout[ix] = self.mtbconcat(xs)

class AppITMTInit(AppBuilder):
    """Callgraph to test use of MT and IT kernels in init"""
    def __init__(self):
        self.pn = PlusN()
        self.mtbi = MTPassThrough()
        self.mtbo = MTPassThrough()
        self.itwr = ITWrite()
        self.itrd = ITRead()
        super().__init__()

    def callgraph(self, x_in:np.ndarray, x_out:np.ndarray)->None:
        for t in range(x_in.shape[0]):
            x = self.itrd(x_in[t])
            x = self.mtbi(x)
            x = self.pn(x, x.nbytes, 1)
            x = self.mtbo(x)
            _ = self.itwr(x, x_out[t]) # equivalent to x_out[t] = x


def _write_mlir(app, *args):
    """ Writes the MLIR to a file that was used in the test. """

    callingfx = inspect.stack()[1][3]
    mlir = app.to_mlir(*args) if len(args) > 0 else app.to_mlir(*args)
    with open(f'{callingfx}.mlir.cgv1', 'w') as fh:
        fh.write(mlir)


def _test_appbuild(app, *args):
    """ Builds an AppBuilder object while saving output to a log. """
    try:
        app.build(*args)
    except Exception as e:

        if app.ab.buildlog is not None:
            with open(app.ab.buildlog, encoding="utf8") as f:
                log = f.read()
        else:
            log = ""

        exception_msg = '\n\n\n'.join([
                        "\n---Callgraph Python",
                        inspect.getsource(app.callgraph),
                        "\n---Application MLIR",
                        app.to_mlir(*args),
                        "---Build Log",
                        log,
                        "---Original Exception",
                        str(e)])

        raise Exception(exception_msg)


def check_npu():
    """Utility to check that the IPU is available before a runtime test runs, otherwise skip test."""
    if platform.system() == 'Windows':
        if not get_device_status() == "OK":
            pytest.skip('Skipping test because the IPU device is not enabled on this device.')
        xbu = XBUtil()
        for app in xbu.list_apps():
            appname = list(app.keys())[0]
            if appname.endswith("IPURiallto"):
                pytest.skip('Skipping test because the IPU is in an unstable state.')


def reset_device():
    """Utility to reset the IPU before a test starts. (Requires to be launched in admin powershell)"""
    result = reset_npu()
    assert result.returncode == 0, "Failed to reset IPU"
    return result


@pytest.fixture
def clear_kernel_cache(request):
    """Clears the cache of built kernel objects."""
    KernelObjectBuilder.clear_cache()


@pytest.fixture
def manage_testing(request):
    """ Ensure that everything is cleaned up if the test falls over.
    Also save all the test specific artifacts to a given location."""
    yield

    if platform.system() == 'Windows':
        test_name = request.node.name
        os.makedirs(Path('logs'), exist_ok=True)
        logdir = Path(f'logs/{test_name}')
        if os.path.exists(logdir) and os.path.isdir(logdir):
            shutil.rmtree(logdir)
        os.makedirs(logdir, exist_ok=True)
        for f in Path.cwd().glob('*.xclbin'):
            shutil.move(f, logdir)
        for f in Path.cwd().glob('*.seq'):
            shutil.move(f, logdir)
        for f in Path.cwd().glob('*.mlir'):
            shutil.move(f, logdir)
