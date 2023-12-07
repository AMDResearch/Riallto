# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import inspect
import npu.lib.kernels

def aiekernels():
    """Returns a list of the optimized AIE kernels available"""
    return [x[0] for x in
            inspect.getmembers(npu.lib.kernels, predicate=inspect.isclass)]
