# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import cv2
from pathlib import Path
import numpy as np
from .kernelgenerator import KernelObjCall


class Rgba2Gray():
    """Vectorized implementation of the `cv2.cvtColor` RGBA2Gray conversion.

    This kernel is used to convert 4 channel image input into a single
    channel grayscale image.

    Read more about this function in the cv2 docs:
    https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#ga397ae87e1288a81d2363b61574eb8cab
    """
    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "rgba2gray.cpp")
        return KernelObjCall(cpp, cls.behavioralfx, *args)

    def behavioralfx(self):
        if self.nbytes.value != self.in_buffer.array.nbytes:
            raise ValueError(f"'in_buffer' size ({self.in_buffer.array.nbytes})"
                             f" does not match 'nbytes' ({self.nbytes.value})")

        in_reshaped = self.in_buffer.array.reshape(1, self.nbytes.value // 4, 4)
        gray = cv2.cvtColor(in_reshaped, cv2.COLOR_RGBA2GRAY)
        self.out_buffer.array = np.squeeze(gray)


class Rgba2GrayScalar():
    """Scalar implementation of of the `cv2.cvtColor` RGBA2Gray conversion."""

    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "rgba2gray_scalar.cpp")
        return KernelObjCall(cpp, Rgba2Gray.behavioralfx, *args)
