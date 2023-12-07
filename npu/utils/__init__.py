# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Utilities
=========

The `npu.utils` submodule provides useful utility functions to visualize,
list available resources, and introspect NPU applications.

"""

from .nputop import nputop
from .videoapps import videoapps
from .aiekernels import aiekernels
from .imgplot import image_plot
from .imgread import OpenCVImageReader
