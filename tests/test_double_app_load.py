# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from .test_applications import check_npu, manage_testing
from npu.runtime import AppRunner
from npu.runtime.apprunner import IPUAppAlreadyLoaded

from .test_applications import SimplePlusN


def test_double_load(manage_testing):
    """Tests loading two applications with the same name/UUID simultaneously, should throw IPUAppAlreadyLoaded error."""
    check_npu()
    array = np.arange(256,dtype=np.uint8)
    n = 5

    trace_app = SimplePlusN()

    trace_app.build(array, n)
    app = AppRunner("SimplePlusN.xclbin")

    try:
        app_alt = AppRunner("SimplePlusN.xclbin")
    except IPUAppAlreadyLoaded:
        print(f"Test passed")
    except Exception as e:
        del app
        raise RuntimeError(f"Test Failed, wrong error received\n {e}")

    del app
