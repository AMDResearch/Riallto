# Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
from npu.lib import Plus1, PlusN
from npu.build.kernelbuilder import KernelObjectBuilder


KernelObjectBuilder.clear_cache()


def test_asm_kernel_built():
    """Test if a built kernel returns a string of text"""

    kernelobj = Plus1()
    kernelobj.build()
    assert kernelobj.asm


def test_asm_kernel_notbuilt():
    """Test if a non built kernel returns RuntimeError"""

    kernelobj = PlusN()
    with pytest.raises(RuntimeError) as excinfo:
        _ = kernelobj.asm

    assert 'Kernel is not built (compiled)' in str(excinfo.value)
