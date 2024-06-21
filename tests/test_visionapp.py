# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
from .test_callgraphs import _test_callgraph_singlekernel_nbytes_build
from .test_callgraphs import _test_callgraph_1in1outNrtps_singlekernel_build
from .test_callgraphs import _test_callgraph_2in1outNrtps_singlekernel_build
from npu.lib import AddWeighted, AddWeightedScalar
from npu.lib import BitwiseOr, BitwiseOrScalar
from npu.lib import BitwiseAnd, BitwiseAndScalar
from npu.lib import Gray2Rgba, Gray2RgbaScalar
from npu.lib import Rgba2Gray, Rgba2GrayScalar
from npu.lib import Rgba2Hue, Rgba2HueScalar
from npu.lib import Median, MedianScalar
from npu.lib import Filter2d, Filter2dScalar
from npu.lib import ThresholdRgba, ThresholdGrayscale
from npu.lib import Inverse
from npu.lib import InRange
from npu.lib import PlusN, Plus1


one_rtp = [Plus1, Inverse, Gray2Rgba, Gray2RgbaScalar, Rgba2Gray,
           Rgba2GrayScalar, Rgba2Hue, Rgba2HueScalar, Median, MedianScalar]
two_input = [BitwiseOr, BitwiseOrScalar, BitwiseAnd, BitwiseAndScalar]
multiple_rtp = [('PlusN; 1'), ('InRange; 2'), ('ThresholdRgba; 9'),
                ('ThresholdGrayscale; 3'), ('Filter2d(linewidth=1280); 8'),
                ('Filter2d(linewidth=1920); 8'),
                ('Filter2dScalar(linewidth=1280); 8'),
                ('Filter2dScalar(linewidth=1920); 8') ]

@pytest.mark.parametrize('kernel', one_rtp)
def test_callgraph_kernel_one_rtp(kernel):
    """ Build test callgraph of kernels with a single RTP"""

    _test_callgraph_singlekernel_nbytes_build(kernel)


@pytest.mark.parametrize('kernel', [AddWeighted, AddWeightedScalar])
def test_callgraph_addweighted_build(kernel):
    """ Build test callgraph of addweighed kernel."""

    _test_callgraph_2in1outNrtps_singlekernel_build(kernel, 3)


@pytest.mark.parametrize('kernel', two_input)
def test_callgraph_twobuffin_onebuffout_build(kernel):
    """ Build test callgraph of kernels with two inputs and one RTP """

    _test_callgraph_2in1outNrtps_singlekernel_build(kernel, 0)


@pytest.mark.parametrize('kernel', multiple_rtp)
def test_callgraph_multiplertp_build(kernel):
    """ Build test of callgraph kernels with multiple RTP """
    arg0, arg1 = kernel.split(';')
    _test_callgraph_1in1outNrtps_singlekernel_build(eval(arg0), eval(arg1))
