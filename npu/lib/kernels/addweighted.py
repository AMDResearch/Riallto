# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import cv2
from pathlib import Path
from .kernelgenerator import KernelObjCall


class AddWeighted():
    """Vectorized implementation of the `cv2.AddWeighted` function.

    This kernel accepts 2 inputs, each input is weighted by the RTPs alpha
    and beta, then emits a single output which is the element-wise addition
    of the weighted inputs.

    Read more about this function in the cv2 docs:
    https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19
    """

    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "addWeighted.cpp")
        return KernelObjCall(cpp, cls.behavioralfx, *args)

    def behavioralfx(self):
        if self.in_buffer1.shape != self.in_buffer2.shape:
            raise ValueError(f"'in_buffer1' shape ({self.in_buffer1.shape})"
                    f" does not match in_buffer2 shape ({self.in_buffer2.shape})")
        if self.nbytes.value != self.in_buffer1.array.nbytes:
            raise ValueError("'in_buffer1' and 'in_buffer2' size "
                             f"({self.in_buffer1.array.nbytes})"
                             f" do not match 'nbytes' ({self.nbytes.value})")

        self.out_buffer.array = \
            cv2.addWeighted(self.in_buffer1.array, self.alpha.value,
                            self.in_buffer2.array, self.beta.value,
                            self.gamma.value)


class AddWeightedScalar():
    """Scalar implementation of the `cv2.AddWeighted` function"""

    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "addWeighted_scalar.cpp")
        return KernelObjCall(cpp, AddWeighted.behavioralfx, *args)
