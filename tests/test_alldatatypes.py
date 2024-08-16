# Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import numpy as np
from npu.build.kernel import Kernel
from npu.runtime import AppRunner
from .test_applications import check_npu
from .test_applications import SingleKernelCall
from ml_dtypes import bfloat16


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


def _appbuild_and_test(datatype, shapein=None):
    check_npu()
    # get bytes per item
    bpi = np.dtype(datatype).itemsize
    shape = shapein if shapein else (4//bpi)
    buffin = np.zeros(shape=shape, dtype=datatype)
    buffout = np.zeros(buffin.shape, dtype=buffin.dtype)

    kernel_src0 = kernel_src.replace('#define N 4', f'#define N {buffin.size}')
    datatype_txt = str(np.dtype(datatype))
    kernel_src0 = kernel_src0.replace('uint8_t', f'{datatype_txt}' +
                                      '' if datatype == bfloat16 else '_t')

    passthrough = Kernel(kernel_src0)
    passthrough.behavioralfx = passthrough_behavior
    trace_app = SingleKernelCall(passthrough)
    trace_app.build(buffin, buffout)

    app = AppRunner("SingleKernelCall.xclbin")

    # generate a numpy array of random data of shape buffin.shape
    if datatype == bfloat16:
        test_data = bfloat16(np.random.randn(*buffin.shape))
    else:
        test_data = np.random.randint(0, (2**(bpi*8))-1, buffin.shape,
                                      dtype=buffin.dtype)
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


@pytest.mark.parametrize('dimension', ['np.uint8; (4, 12)', 'np.uint16; (8, 2)',
                                       'np.uint32; (2, 3)', 'bfloat16; (8, 2)'])
def test_appbuild_good_shapes_2d(dimension):
    dtype, shape = dimension.split(';')
    _appbuild_and_test(eval(dtype), eval(shape))


@pytest.mark.parametrize('datatype', [np.uint8, np.uint16])
def test_appbuild_bad_shapes(datatype):
    with pytest.raises(ValueError) as valerror:
        _appbuild_and_test(datatype, (1, 1))
    assert 'Cannot move non 4B array' in str(valerror.value)
