# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from pathlib import Path
from .kernelgenerator import KernelObjCall


_supported_widths = [1280, 1920]


class Filter2d():
    """Vectorized implementation of a grayscale filter2d kernel

    It supports 720p or 1080p image resolutions.

    It accepts a row of the image at a time and uses a circular buffer to keep
    track of the last three rows. It then applies the filter to the three rows.

    Keyword Parameter
    -----------------
    linewidth: optional
        Either 1280, 1920

    Read more about this function in the cv2 docs:
    https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga27c049795ce870216ddfb366086b5a04
    """

    def __new__(cls, *args, linewidth: int = 1280):
        if linewidth == 1280:
            cpp = str(Path(__file__).parent / "cpp" / "filter2d_720p.cpp")
        elif linewidth == 1920:
            cpp = str(Path(__file__).parent / "cpp" / "filter2d_1080p.cpp")
        else:
            raise ValueError(f"Width {linewidth} is not supported for the "
                             "Filter2d kernel, supported_widths: "
                             f"{_supported_widths}")

        return KernelObjCall(cpp, cls.behavioralfx, *args)

    def behavioralfx(self):
        self.out_buffer.array = self.in_buffer.array


class Filter2dScalar():
    """Scalar implementation of a filter2d function.

    Keyword Parameter
    -----------------
    linewidth: optional
        Either 1280, 1920
    """

    def __new__(cls, *args, linewidth: int = 1280):
        if linewidth == 1280:
            cpp = str(Path(__file__).parent / "cpp" / "filter2d_720p_scalar.cpp")
        elif linewidth == 1920:
            cpp = str(Path(__file__).parent / "cpp" / "filter2d_1080p_scalar.cpp")
        else:
            raise ValueError(f"Width {linewidth} is not supported for the "
                             "Filter2dScalar kernel, supported_widths: "
                             f"{_supported_widths}")

        return KernelObjCall(cpp, Filter2d.behavioralfx, *args)
