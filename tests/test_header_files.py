# Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import os
from npu.build.kernel import Kernel

header_src = '''
#include <aie_api/aie.hpp>

#define N 16

void passthrough_header(uint8_t *in_buffer, uint8_t *out_buffer) {
    for(int i=0; i < N; i++) {
        out_buffer[i] = in_buffer[i];
    }
}
'''

kernel_src = '''
#include "passthrough_header.<extension>"


extern "C" {

void passthrough(uint8_t *in_buffer, uint8_t *out_buffer) {
    passthrough_header(in_buffer, out_buffer);
}

} // extern "C"
'''


def _kernel_build(extension):
    header_file = 'passthrough_header' + extension
    cpp_file = 'passthrough.cpp'
    kernel_source = kernel_src.replace('.<extension>', extension)
    # write header file
    with open(header_file, 'w', encoding='utf-8', newline='\n') as file:
        file.write(header_src)

    with open(cpp_file, 'w', encoding='utf-8', newline='\n') as file:
        file.write(kernel_source)

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
    os.remove(header_file)
    os.remove(cpp_file)


@pytest.mark.parametrize('extension', ['.h', '.hh', '.hpp', '.hxx', '.h++'])
def test_build_kernel_known_extension(extension):
    _kernel_build(extension)
