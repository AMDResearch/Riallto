# Copyright 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from npu.build.kernel import Kernel


class KernelObjCall():
    """ Helper function that generates new kernel objects."""
    def __new__(cls, cpp, bfx, *args):
        kobj = Kernel(cpp, bfx)
        return kobj(*args) if len(args) > 0 else kobj
