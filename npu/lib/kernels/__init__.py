# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Kernels
=======

The `npu.lib.kernels` submodule includes pre-defined, but not pre-built
kernels that can be used in custom application graph definitions. Each
kernel also has a behavioral model associated with it allowing validation
of data shapes and functionality in Python, before compiling the application
binaries.

Example usage of inverse kernel behavioral model


    import numpy as np
    from npu.lib.kernels.inverse import Inverse

    kernel = Inverse()
    x = np.array(np.ones(100,)
    y = kernel(x, dtype=np.uint8), 100, 1)


"""

from .inverse import Inverse
from .plusn import Plus1, PlusN
from .bitwiseor import BitwiseOr, BitwiseOrScalar
from .bitwiseand import BitwiseAnd, BitwiseAndScalar
from .gray2rgba import Gray2Rgba, Gray2RgbaScalar
from .rgba2gray import Rgba2Gray, Rgba2GrayScalar
from .rgba2hue import Rgba2Hue, Rgba2HueScalar
from .median import Median, MedianScalar
from .filter2d import Filter2d, Filter2dScalar 
from .threshold import ThresholdRgba,ThresholdGrayscale, RgbaRtpThres
from .rgba_inverse import RgbaInverse
from .addweighted import AddWeighted, AddWeightedScalar
