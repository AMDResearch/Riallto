# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from pathlib import Path
from .kernelgenerator import KernelObjCall


class RgbaInverse():
    """Vectorized inverse operation for an RGBA image.

     For each RGBA 4-Byte pair it subtracts 255 from the R/G/B channel while
     leaving the alpha channel unchanged. This kernel accepts a single input
     for the input tile it is processing and returns a buffer for the output
     tile.
     """

    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "rgba_inverse.cpp")
        return KernelObjCall(cpp, cls.behavioralfx, *args)

    def behavioralfx(self):
        if self.nbytes.value != self.in_buffer.array.nbytes:
            raise ValueError(f"'in_buffer' size ({self.in_buffer.array.nbytes})"
                             f" does not match 'nbytes' ({self.nbytes.value})")

        self.out_buffer.array = 255 - self.in_buffer.array
