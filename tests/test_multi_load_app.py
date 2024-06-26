# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import numpy as np
import os
import platform
from .test_applications import check_npu
from npu.runtime import AppRunner
from npu.utils.xbutil import XBUtil
from .test_applications import SimplePlusN


def _get_full_path(xclbin: str = None) -> str:
    binaries = os.path.dirname(os.path.abspath(__file__)) \
        + '/../npu/lib/applications/binaries/'
    return os.path.abspath(os.path.join(binaries, xclbin))


def _xbutil(appcount):
    state = True
    if platform.system() == 'Windows':
        appsreport = XBUtil()
        state = appsreport.app_count == appcount
        del appsreport
    return state


def test_double_load_custom_app():
    """Tests loading two applications with the same name/UUID simultaneously"""
    check_npu()
    array = np.arange(256, dtype=np.uint8)
    n = 5

    trace_app = SimplePlusN()

    trace_app.build(array, n)
    app = AppRunner("SimplePlusN.xclbin")
    assert app
    app1 = AppRunner("SimplePlusN.xclbin")
    assert app1
    assert _xbutil(2)
    del app, app1


@pytest.mark.parametrize('numappsreport', [2, 3, 4])
def test_videoapp_n_loads(numappsreport):
    """Load N instances of the same app. Test should pass"""
    check_npu()
    appbin = _get_full_path("color_threshold_v2_720p.xclbin")
    app = []
    for _ in range(numappsreport):
        app.append(AppRunner(appbin))

    assert _xbutil(numappsreport)

    for i in range(numappsreport):
        assert app[i]
    del app


@pytest.mark.skipif(platform.system() == 'Linux',
                    reason="Skip because we don't get app_count in Linux")
def test_videoapp_five_loads():
    """Load five instances of the same app.
    AppRunner should return a RuntimeError indicating not enough space
    """
    check_npu()
    appbin = _get_full_path("color_threshold_v1_720p.xclbin")
    app = []
    for _ in range(4):
        app.append(AppRunner(_get_full_path(appbin)))

    for i in range(4):
        assert app[i]

    assert _xbutil(4)

    with pytest.raises(RuntimeError) as verr:
        app1 = AppRunner(appbin)
        del app1
    assert 'There is currently no free space on the NPU' in str(verr.value)

    del app
