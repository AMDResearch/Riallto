# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from pathlib import Path
from .kernelgenerator import KernelObjCall


class InRange():
    """Vectorized implementation of the `cv2.inRange` function.

    This vectorized implementation operates on grayscale image inputs.
    Read more about this function in the cv2 docs:
    https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981
    """

    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "inrange.cpp")
        return KernelObjCall(cpp, cls.behavioralfx, *args)

    def behavioralfx(self):
        if self.nbytes.value != self.in_buffer.array.nbytes:
            raise ValueError(f"'in_buffer' size ({self.in_buffer.array.nbytes})"
                             f" does not match 'nbytes' ({self.nbytes.value})")

        self.out_buffer.array = self.in_buffer.array
