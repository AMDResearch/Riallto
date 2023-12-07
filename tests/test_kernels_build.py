# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import npu
from npu.utils import aiekernels
from npu.lib import Filter2d, Filter2dScalar


@pytest.mark.parametrize('kernel', aiekernels())
def test_kernel_build(kernel):
    """ Build test for all kernels available"""
    kernel_obj = getattr(npu.lib, kernel)()
    _test_kernel_build(kernel_obj)


def _test_kernel_build(kern):
    """ Test utility to build a kernel and log stdout/stderr. """
    try:
        kern.build()
    except Exception as e:
        with open(kern.kb.buildlog, encoding="utf8") as f:
            log = f.read()

        exception_msg = '\n\n\n'.join(["\n---Kernel Source Code",
                                       kern._srccode, "---Build Log", log,
                                       "---Original Exception", str(e)])

        raise Exception(exception_msg)


@pytest.mark.parametrize('kernel', [Filter2d, Filter2dScalar])
def test_filter2d_unsupported_resolutions(kernel):
    """ Build test for all kernels available"""
    with pytest.raises(ValueError) as excinfo:
        _ = kernel(linewidth=3840)

    assert "is not supported for the Filter2d" in str(excinfo.value)
