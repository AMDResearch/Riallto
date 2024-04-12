# Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import numpy as np
from npu.build.kernel import Kernel
from npu.runtime import AppRunner
from .test_applications import check_npu
from .test_applications import SingleKernelCall


kernel_src = '''
#include <aie_api/aie.hpp>

#define N 4

extern "C" {

void passthrough(uint8_t *in_buffer, uint8_t *out_buffer) {
    for(int i=0; i < N; i++) {
        out_buffer[i] = in_buffer[i];
    }
}

} // extern "C"
'''

def passthrough_behavior(invobj):
    invobj.out_buffer.array = invobj.in_buffer.array


def _appbuild_and_test(datatype, two_dimension=False):
    check_npu()
    # get bytes per item
    bpi = np.dtype(datatype).itemsize
    shape = (4//bpi, 8//bpi) if two_dimension else (4//bpi)
    buffin = np.zeros(shape=shape, dtype=datatype)
    buffout = np.zeros(buffin.shape, dtype=buffin.dtype)

    kernel_src0 = kernel_src.replace('#define N 4', f'#define N {buffin.size}')
    datatype_txt = str(np.dtype(datatype))
    kernel_src0 = kernel_src0.replace('uint8_t', f'{datatype_txt}' + '_t' )

    passthrough = Kernel(kernel_src0)
    passthrough.behavioralfx = passthrough_behavior
    trace_app = SingleKernelCall(passthrough)
    trace_app.build(buffin, buffout)

    app = AppRunner("SingleKernelCall.xclbin")

    # generate a numpy array of random data of shape buffin.shape
    test_data = np.random.randint(0, (2**(bpi*8))-1,
                                  buffin.shape, dtype=buffin.dtype)
    res = np.zeros(buffin.shape, dtype=buffin.dtype)
    bo_in = app.allocate(shape=buffin.shape, dtype=buffin.dtype)
    bo_out = app.allocate(shape=buffin.shape, dtype=buffin.dtype)

    bo_in[:] = test_data
    bo_in.sync_to_npu()
    app._refresh_sequence()
    app.call(bo_in, bo_out)
    bo_out.sync_from_npu()
    res[:] = bo_out[:]

    del app
    assert np.array_equal(test_data, res)


@pytest.mark.parametrize('datatype', [np.uint8, np.uint16, np.uint32])
def test_appbuild_good_shapes_1d(datatype):
    _appbuild_and_test(datatype)


@pytest.mark.parametrize('datatype', [np.uint8, np.uint16, np.uint32])
def test_appbuild_good_shapes_2d(datatype):
    _appbuild_and_test(datatype, True)


@pytest.mark.parametrize('datatype', [np.uint8, np.uint16])
def test_appbuild_bad_shapes(datatype):
    check_npu()
    shape = (1, 1)
    buffin = np.zeros(shape=shape, dtype=datatype)
    buffout = np.zeros(buffin.shape, dtype=buffin.dtype)

    kernel_src0 = kernel_src.replace('#define N 4', f'#define N 1')

    passthrough = Kernel(kernel_src0)
    passthrough.behavioralfx = passthrough_behavior
    trace_app = SingleKernelCall(passthrough)

    with pytest.raises(ValueError) as valerror:
        trace_app.build(buffin, buffout)
    assert 'Cannot move non 4B array' in str(valerror.value)
