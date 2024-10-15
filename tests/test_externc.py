# Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
from npu.build.kernel import Kernel
from npu.lib import Plus1


kernel_src = Plus1().srccode

kernel_src1 = kernel_src.replace('\n\n}', '')


def test_externc_good():
    krnl_obj = Kernel(kernel_src)
    krnl_obj.build()
    assert krnl_obj


@pytest.mark.parametrize('replacewith', [''])
def test_externc_bad(replacewith):
    src_code = kernel_src1.replace('extern "C" {', replacewith)

    with pytest.raises(SyntaxError) as excinfo:
        _ = Kernel(src_code)

    assert 'extern "C" not found.' in str(excinfo.value)
