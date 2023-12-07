# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from pathlib import Path
from .kernelgenerator import KernelObjCall


class Median():
    """Vectorized implementation of a median filter."""

    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "median.cpp")
        return KernelObjCall(cpp, cls.behavioralfx, *args)

    def behavioralfx(self):
        if self.nbytes.value != self.in_buffer.array.nbytes:
            raise ValueError(f"'in_buffer' size ({self.in_buffer.array.nbytes})"
                             f" does not match 'nbytes' ({self.nbytes.value})")

        # TODO : Add and verify behavioral model
        self.out_buffer.array = self.in_buffer.array


class MedianScalar():
    """Scalar implementation of a median filter."""

    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "median_scalar.cpp")
        return KernelObjCall(cpp, Median.behavioralfx, *args)
