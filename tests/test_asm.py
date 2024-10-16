# Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
from npu.lib import Plus1, PlusN, RgbaRtpThres
from npu.build.kernelbuilder import KernelObjectBuilder


KernelObjectBuilder.clear_cache()


def test_asm_kernel_built():
    """Test if a built kernel returns a string of text"""

    kernelobj = Plus1()
    kernelobj.build()
    kernelobj.asmdisplay()
    assert kernelobj.asm


def test_asm_kernel_notbuilt_asm():
    """Test if a non built kernel returns RuntimeError when calling asm"""

    kernelobj = PlusN()
    with pytest.raises(RuntimeError) as excinfo:
        _ = kernelobj.asm

    assert 'is not built (compiled)' in str(excinfo.value)


def test_asm_kernel_notbuilt_asmdisplay():
    """Test if a non built kernel returns RuntimeError when calling asmdisplay"""

    kernelobj = RgbaRtpThres()
    with pytest.raises(RuntimeError) as excinfo:
        _ = kernelobj.asmdisplay()

    assert 'is not built (compiled)' in str(excinfo.value)
