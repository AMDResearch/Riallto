# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from pathlib import Path
from .kernelgenerator import KernelObjCall


class Plus1():
    """Vectorized implementation of a plus1 function.

    This function simply adds a constant 1 to every element of an input array.
    This is a useful function to verify that your graph is set up
    correctly as the input/output relationship is simple to interpret.
    """

    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "plus1.cpp")
        return KernelObjCall(cpp, cls.behavioralfx, *args)

    def behavioralfx(self):
        if self.nbytes.value != self.in_buffer.array.nbytes:
            raise ValueError(f"'in_buffer' size ({self.in_buffer.array.nbytes})"
                             f" does not match 'nbytes' ({self.nbytes.value})")

        self.out_buffer.array = self.in_buffer.array + 1


class PlusN():
    """Vectorized implementation of a plusN function.

    This function adds a custom value specified by the runtime parameter N
    to every value of an input array.
    """

    def __new__(cls, *args):
        cpp = str(Path(__file__).parent / "cpp" / "plusn.cpp")
        return KernelObjCall(cpp, cls.behavioralfx, *args)

    def behavioralfx(self):
        if self.nbytes.value != self.in_buffer.array.nbytes:
            raise ValueError(f"'in_buffer' size ({self.in_buffer.array.nbytes})"
                             f" does not match 'nbytes' ({self.nbytes.value})")

        self.out_buffer.array = self.n.value + self.in_buffer.array
