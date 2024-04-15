# Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import os
from npu.build.kernel import Kernel

kernel_src = '''
#include "kernels.hpp"

extern "C" {

#define N 720

void passthrough(uint8_t *in_buffer, uint8_t *out_buffer) {
    uint8_t buffer[N];
    rgba2gray_aie(in_buffer, buffer, N*4);
    gray2rgba_aie(buffer, out_buffer, N);
}

} // extern "C"
'''


def test_superkernel_build():
    cpp_file = 'passthrough.cpp'

    with open(cpp_file, 'w', encoding='utf-8', newline='\n') as file:
        file.write(kernel_src)

    class PassThrough():
        def __new__(cls, *args):
            kobj = Kernel(cpp_file, cls.behavioralfx)
            return kobj(*args) if len(args) > 0 else kobj

        def behavioralfx(self):
            self.out_buffer.array = self.in_buffer.array

    # build kernel
    passthrough = PassThrough()
    # lunch build and then assert
    assert (objfile := passthrough.objfile)
    # remove header files
    os.remove(objfile)
    os.remove(cpp_file)
