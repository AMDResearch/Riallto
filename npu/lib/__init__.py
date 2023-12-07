# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Library of prebuilts
====================

The NPU submodule `npu.lib` contains prebuilt binaries for various
vision applications, optimized C++ kernel sources and pre-defined
graphs to help quickly build custom applications.

Applications
------------

Prebuilt applications can be run as-is with no additional build
steps required.

You can list the available videoapps with

    from npu.utils import videoapps
    videoapps()

Example running edge detect

    from npu.lib import EdgeDetectVideoProcessing

    app = EdgeDetectVideoProcessing()
    app.start()

Kernels
-------

List the available kernels with

    from npu.utils import aiekernels
    aiekernels()

Graphs
------

Pre-defined graphs are classes that inherit functionality from
AppBuilder and make it simple to insert your own kernel into an
NPU application without worrying too much about setting up the 
graph for dataflow.

Example using an RGB720pBuilder class with a custom kernel object

    from npu.lib.graphs.graph_1ct import RGB720pBuilder
    app_builder = RGB720pBuilder(kernel=passthrough)

This graph is specifically designed to work with 720p RGBA data,
in applications which have one input and one output buffer.

"""

from .kernels.inverse import Inverse
from .kernels.plusn import Plus1, PlusN
from .kernels.bitwiseor import BitwiseOr, BitwiseOrScalar
from .kernels.bitwiseand import BitwiseAnd, BitwiseAndScalar
from .kernels.gray2rgba import Gray2Rgba, Gray2RgbaScalar
from .kernels.rgba2gray import Rgba2Gray, Rgba2GrayScalar
from .kernels.rgba2hue import Rgba2Hue, Rgba2HueScalar
from .kernels.median import Median, MedianScalar
from .kernels.filter2d import Filter2d, Filter2dScalar 
from .kernels.threshold import ThresholdRgba,ThresholdGrayscale, RgbaRtpThres
from .kernels.rgba_inverse import RgbaInverse
from .kernels.addweighted import AddWeighted, AddWeightedScalar
from .kernels.inrange import InRange


from .applications.videoapps import pxtype
from .applications.videoapps import VideoApplication
from .applications.videoapps import ColorThresholdVideoProcessing
from .applications.videoapps import ScaledColorThresholdVideoProcessing
from .applications.videoapps import EdgeDetectVideoProcessing
from .applications.videoapps import ColorDetectVideoProcessing
from .applications.videoapps import DenoiseTPVideoProcessing
from .applications.videoapps import DenoiseDPVideoProcessing

from .graphs.graph_1ct import RGB720pBuilder
from .graphs.image_looper_720p import ImageLooper720p
