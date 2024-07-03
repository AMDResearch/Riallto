# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import numpy as np
from npu.utils.test_device import get_device_status
from .test_applications import check_npu, manage_testing

import pytest
from npu.build.kernel import Kernel
from npu.build.appbuilder import AppBuilder
from npu.runtime import AppRunner

def vbc_behavioral(obj):
    obj.c.array = obj.a.array
    obj.b.array = obj.a.array

class VectorBroadcastApp(AppBuilder):
    def __init__(self) -> None:

        self.vectorbroadcast = Kernel('''
        #include <stdint.h>
        #include <stdio.h>
        #include <stdlib.h>
        #include <aie_api/aie.hpp>

        extern "C" {
        void vectorbroadcast(uint8_t* a, uint8_t* b, uint8_t* c, const uint32_t nbytes) {

            ::aie::vector<uint8_t, 32> ai, bi, ci;

            for(int j=0; j<nbytes; j+=32) {
                ai = ::aie::load_v<32>(a);
                a += 32;
                ::aie::store_v(b, ai);
                b += 32;
                ::aie::store_v(c, ai);
                c += 32;
            }
        }
        }
        ''',vbc_behavioral)

        super().__init__()
    


    """Callgraph to test two output kernel"""
    def callgraph(self, x):
        return self.vectorbroadcast(x, x.nbytes)




def test_singlekernel_twooutput_build(manage_testing):
    """ Test the simplest single kernel app with two outputs."""

    appb = VectorBroadcastApp()

    a = np.random.randint(0, 256, size=4096, dtype=np.uint8)

    appb.build(a)

    appr = AppRunner("VectorBroadcastApp.xclbin")


    a_in = appr.allocate(shape=a.shape, dtype=a.dtype)
    b_out = appr.allocate(shape=a.shape, dtype=a.dtype)
    c_out = appr.allocate(shape=a.shape, dtype=a.dtype)    

    a_in[:] = a
    a_in.sync_to_npu()

    appr._refresh_sequence()
    appr.call(a_in, b_out, c_out)

    b_out.sync_from_npu()
    c_out.sync_from_npu()   
    print(np.array(a_in),np.array(b_out),np.array(c_out))

    assert np.array_equal(a_in, b_out)
    assert np.array_equal(a_in, c_out)

    del appr
