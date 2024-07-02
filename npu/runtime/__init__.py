# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Runtime
=======

The NPU submodule `npu.runtime` contains classes and functions to run
custom applications on the NPU device. Here APIs are provided to 
load custom xclbins, allocate numpy arrays to NPU-compatible buffers
and read processed data out.

Example usage of the AppRunner class and allocate methods to process
your data.

    import numpy as np
    from npu.runtime import AppRunner

    # Register the xclbin and program the NPU
    app = AppRunner("myapp.xclbin")

    # Generate random python data and allocate input
    # and output buffers
    test_data = np.random.randint(0, 255, 256, dtype=np.uint8)
    bo_in = app.allocate(shape=(256,), dtype=np.uint8)
    bo_out = app.allocate(shape=(256,), dtype=np.uint8)

    # Copy input data into NPU memory
    bo_in[:] = test_data
    bo_in.sync_to_npu()

    # Execute the application
    app.call(bo_in, bo_out)

    # Update the output buffer with the results
    bo_out.sync_from_npu()

    # Print the output
    print(np.array(bo_out))

    # Unload the application, free resources
    del app


"""

import platform
import os
if platform.system() == 'Windows':
    os.add_dll_directory(os.path.join('C:\\', 'Windows', 'System32', 'AMD'))

from .pyxrt import device, xclbin
from .pyxrt import hw_context, kernel
from .pyxrt import bo, xclBOSyncDirection
from .sequence import Sequence
from .aie_host_utils import print_dolphin
from .apprunner import AppRunner, PynqBuffer
