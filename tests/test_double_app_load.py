# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
import os
import pytest
from .test_applications import check_npu, manage_testing
from npu.runtime import AppRunner
from npu.runtime.apprunner import IPUAppAlreadyLoaded

from .test_applications import SimplePlusN


def test_double_load(manage_testing):
    """Tests loading two applications with the same name/UUID simultaneously, should throw IPUAppAlreadyLoaded error."""
    if os.name != "nt":
        pytest.skip("Only currently works on windows due to xbutil support")
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
    else:
        del app
        raise RuntimeError("Test Failed, wrong error received")

    del app
