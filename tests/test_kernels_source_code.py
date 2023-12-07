# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import npu
from npu.utils import aiekernels


@pytest.mark.parametrize('kernel', aiekernels())
def test_src_code(kernel):
    """ Test that attempts to parse all the kernel source code. """
    kernel_obj = getattr(npu.lib, kernel)()
    assert kernel_obj.srccode
