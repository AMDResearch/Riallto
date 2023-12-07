# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from pathlib import Path
import numpy as np
from .kernelgenerator import KernelObjCall


class BitwiseAnd():
    """Vectorized implementation of the `cv2.bitwise_and` function.

    It accepts 2 buffer inputs and outputs 1 output which is the element-wise
    logical AND operation applied on each value.

    Read more about this function in the cv2 docs:
    https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga60b4d04b251ba5eb1392c34425497e14
    """

    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "bitwiseAnd.cpp")
        return KernelObjCall(cpp, cls.behavioralfx, *args)

    def behavioralfx(self):
        if self.in_buffer1.array.shape != self.in_buffer2.array.shape:
            raise ValueError(f"'in_buffer1' shape ({self.in_buffer1.shape})"
                             " does not match in_buffer2 shape "
                             f"({self.in_buffer2.shape})")
        if self.nbytes.value != self.in_buffer1.array.nbytes:
            raise ValueError("'in_buffer1' and 'in_buffer2' size "
                             f"({self.in_buffer1.array.nbytes})"
                             f" do not match 'nbytes' ({self.nbytes.value})")

        self.out_buffer.array = np.bitwise_and(self.in_buffer1.array,
                                               self.in_buffer2.array)


class BitwiseAndScalar():
    """Scalar implementation of the `cv2.bitwise_and` function"""

    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "bitwiseAnd_scalar.cpp")
        return KernelObjCall(cpp, BitwiseAnd.behavioralfx, *args)
