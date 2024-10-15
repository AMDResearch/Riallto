# Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
from npu.build.kernel import Kernel
from npu.lib import Plus1


kernel_src = Plus1().srccode

kernel_src1 = kernel_src.replace('\n\n}', '')
kernel_src2 = kernel_src1.replace('extern "C" {', '')
kernel_src1 = kernel_src1.replace('extern "C"', '// extern "C"')


def test_externc_good():
    krnl_obj = Kernel(kernel_src)
    krnl_obj.build()
    assert krnl_obj


@pytest.mark.parametrize('src_code', [kernel_src1, kernel_src2])
def test_externc_bad(src_code):

    with pytest.raises(RuntimeError) as excinfo:
        _ = Kernel(src_code)

    assert 'extern "C" not found.' in str(excinfo.value)
