# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from pathlib import Path
from .kernelgenerator import KernelObjCall


class Inverse():
    """Vectorized implementation of a invert function on grayscale images.

    This vectorized implementation operates on grayscale image inputs.
    It performs the operation 255 - image
    """

    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "inverse.cpp")
        return KernelObjCall(cpp, cls.behavioralfx, *args)

    def behavioralfx(self):
        if self.nbytes.value != self.in_buffer.array.nbytes:
            raise ValueError(f"'in_buffer' size ({self.in_buffer.array.nbytes})"
                             f" does not match 'nbytes' ({self.nbytes.value})")

        self.out_buffer.array = 255 - self.in_buffer.array
