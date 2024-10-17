# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from pathlib import Path
import numpy as np
from npu.build.kernel import Kernel
from npu.build.mtkernel import MTPassThrough, MTSplit, MTConcat
from npu.build.appbuilder import AppBuilder
from .test_visualization import _count_class_occurrences
from npu.lib import Plus1, PlusN


imgdir = str(Path(__file__).parent / "images") + '/'


k_2in_1out = """
#include "aie_api/aie.hpp"
extern "C" {
    void aie_kernel_2_1(uint16_t* in0, uint16_t* in1, uint16_t* out_buffer) {
        auto vec_in0 = ::aie::load_v<32>(in0);
        auto vec_in1 = ::aie::load_v<32>(in1);
        in0 += 32;
        in1 += 32;
        ::aie::store_v(out_buffer, vec_in0);
        out_buffer += 32;
        ::aie::store_v(out_buffer, vec_in1);
        out_buffer += 32;
    }
}
"""

k_1in_2out = """
#include "aie_api/aie.hpp"
extern "C" {
    void aie_kernel_1_2(uint16_t* in_buffer, uint16_t* out0, uint16_t* out1) {
        auto vec_in = ::aie::load_v<32>(in_buffer);
        in_buffer += 32;
        ::aie::store_v(out0, vec_in);
        out0 += 32;
        vec_in = ::aie::load_v<32>(in_buffer);
        in_buffer += 32;
        ::aie::store_v(out1, vec_in);
        out1 += 32;
    }
}
"""


k_2in_2out = """
#include "aie_api/aie.hpp"
extern "C" {
    void aie_kernel_2_2(uint16_t* in0, uint16_t* in1, uint16_t* out0, uint16_t* out1) {
        auto vec_in0 = ::aie::load_v<32>(in0);
        in0 += 32;
        ::aie::store_v(out1, vec_in0);
        out1 += 32;
        auto vec_in1 = ::aie::load_v<32>(in1);
        in1 += 32;
        ::aie::store_v(out0, vec_in1);
        out0 += 32;
    }
}
"""


def kernel_behavior_2_1(invobj):
    invobj.out_buffer.array = np.concatenate((invobj.in0.array,
                                              invobj.in1.array))


def kernel_behavior_1_2(invobj):
    x = np.split(invobj.in_buffer.array, 2)
    invobj.out0.array = x[0]
    invobj.out1.array = x[1]


def kernel_behavior_2_2(invobj):
    invobj.out1.array = invobj.in0.array
    invobj.out0.array = invobj.in1.array


def test_viz_2in_1out():
    class SingleKernel_2_1(AppBuilder):
        def __init__(self):
            self.kernel = Kernel(k_2in_1out, kernel_behavior_2_1)
            super().__init__()

        def callgraph(self, x_in0, x_in1, x_out):
            x_out[:] = self.kernel(x_in0, x_in1)

    x_in0 = np.zeros(shape=(32, 1), dtype=np.uint16)
    x_in1 = np.zeros(shape=(32, 1), dtype=np.uint16)
    x_out = np.zeros(shape=(64, 1), dtype=np.uint16)

    app_builder = SingleKernel_2_1()
    _ = app_builder.to_metadata(x_in0, x_in1, x_out)
    app_builder.save(svgfile := f'{imgdir}{app_builder.name}.svg')
    assert _count_class_occurrences(svgfile, 'kernel') == 2
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == 6
    assert _count_class_occurrences(svgfile, 'mem_connections') == 9


def test_viz_1in_2out():
    class SingleKernel_1_2(AppBuilder):
        def __init__(self):
            self.kernel = Kernel(k_1in_2out, kernel_behavior_1_2)
            super().__init__()

        def callgraph(self, x_in, x_out0, x_out1):
            x_out0[:], x_out1[:] = self.kernel(x_in)

    x_in = np.zeros(shape=(64, 1), dtype=np.uint16)
    x_out0 = np.zeros(shape=(32, 1), dtype=np.uint16)
    x_out1 = np.zeros(shape=(32, 1), dtype=np.uint16)

    app_builder = SingleKernel_1_2()
    _ = app_builder.to_metadata(x_in, x_out0, x_out1)
    app_builder.save(svgfile := f'{imgdir}{app_builder.name}.svg')
    assert _count_class_occurrences(svgfile, 'kernel') == 2
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == 6
    assert _count_class_occurrences(svgfile, 'mem_connections') == 9


def test_viz_2in_2out():

    class SingleKernel_2_2(AppBuilder):
        def __init__(self):
            self.kernel = Kernel(k_2in_2out, kernel_behavior_2_2)
            super().__init__()

        def callgraph(self, x_in0, x_in1, x_out0, x_out1):
            x_out0[:], x_out1[:] = self.kernel(x_in0, x_in1)

    x_in0 = np.zeros(shape=(32, 1), dtype=np.uint16)
    x_in1 = np.zeros(shape=(32, 1), dtype=np.uint16)
    x_out0 = np.zeros(shape=(32, 1), dtype=np.uint16)
    x_out1 = np.zeros(shape=(32, 1), dtype=np.uint16)

    app_builder = SingleKernel_2_2()
    _ = app_builder.to_metadata(x_in0, x_in1, x_out0, x_out1)
    app_builder.save(svgfile := f'{imgdir}{app_builder.name}.svg')
    assert _count_class_occurrences(svgfile, 'kernel') == 2
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == 8
    assert _count_class_occurrences(svgfile, 'mem_connections') == 9


def test_viz_k2in_1out_mt():
    class SingleKernel_2_1_mt(AppBuilder):
        def __init__(self):
            self.split = MTSplit(2)
            self.kernel = Kernel(k_2in_1out, kernel_behavior_2_1)
            super().__init__()

        def callgraph(self, x_in, x_out):
            xs = self.split(x_in)
            x_out[:] = self.kernel(*xs)

    x_in = np.zeros(shape=(64, 1), dtype=np.uint16)
    x_out = np.zeros(shape=(64, 1), dtype=np.uint16)

    app_builder = SingleKernel_2_1_mt()
    _ = app_builder.to_metadata(x_in, x_out)
    app_builder.save(svgfile := f'{imgdir}{app_builder.name}.svg')
    assert _count_class_occurrences(svgfile, 'kernel') == 2
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == 6
    assert _count_class_occurrences(svgfile, 'mem_connections') == 9


def test_viz_k1in_2out_mt():
    class SingleKernel_1_2_mt(AppBuilder):
        def __init__(self):
            self.concat = MTConcat()
            self.kernel = Kernel(k_1in_2out, kernel_behavior_1_2)
            super().__init__()

        def callgraph(self, x_in, x_out):
            x0, x1 = self.kernel(x_in)
            x_out[:] = self.concat([x0, x1])

    x_in = np.zeros(shape=(64, 1), dtype=np.uint16)
    x_out = np.zeros(shape=(64, 1), dtype=np.uint16)

    app_builder = SingleKernel_1_2_mt()
    _ = app_builder.to_metadata(x_in, x_out)
    app_builder.save(svgfile := f'{imgdir}{app_builder.name}.svg')
    assert _count_class_occurrences(svgfile, 'kernel') == 2
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == 6
    assert _count_class_occurrences(svgfile, 'mem_connections') == 9


def test_viz_k2in_2out_mt():
    class SingleKernelMT_2_2(AppBuilder):
        def __init__(self):
            self.split = MTSplit(2)
            self.concat = MTConcat()
            self.kernel = Kernel(k_2in_2out, kernel_behavior_2_2)
            super().__init__()

        def callgraph(self, x_in, x_out):
            xs = self.split(x_in)
            x0, x1 = self.kernel(*xs)
            x_out[:] = self.concat([x0, x1])

    x_in = np.zeros(shape=(64, 1), dtype=np.uint16)
    x_out = np.zeros(shape=(64, 1), dtype=np.uint16)

    app_builder = SingleKernelMT_2_2()
    _ = app_builder.to_metadata(x_in, x_out)
    app_builder.save(svgfile := f'{imgdir}{app_builder.name}.svg')
    assert _count_class_occurrences(svgfile, 'kernel') == 2
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == 8
    assert _count_class_occurrences(svgfile, 'mem_connections') == 9
    assert _count_class_occurrences(svgfile, 'mem_tile_buffers') == 8


def test_viz_k2in_2out_inmpt():
    class SingleKernel_2_2_MPT_inbuff(AppBuilder):
        def __init__(self):
            self.kernel = Kernel(k_2in_2out, kernel_behavior_2_2)
            self.mtbi0 = MTPassThrough()
            self.mtbi1 = MTPassThrough()
            super().__init__()

        def callgraph(self, x_in0, x_in1, x_out0, x_out1):
            x_in_mpt0 = self.mtbi0(x_in0)
            x_in_mpt1 = self.mtbi1(x_in1)
            x_out0[:], x_out1[:] = self.kernel(x_in_mpt0, x_in_mpt1)

    x_in0 = np.zeros(shape=(32, 1), dtype=np.uint16)
    x_in1 = np.zeros(shape=(32, 1), dtype=np.uint16)
    x_out0 = np.zeros(shape=(32, 1), dtype=np.uint16)
    x_out1 = np.zeros(shape=(32, 1), dtype=np.uint16)

    app_builder = SingleKernel_2_2_MPT_inbuff()
    _ = app_builder.to_metadata(x_in0, x_in1, x_out0, x_out1)
    app_builder.save(svgfile := f'{imgdir}{app_builder.name}.svg')
    assert _count_class_occurrences(svgfile, 'kernel') == 2
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == 8
    assert _count_class_occurrences(svgfile, 'mem_connections') == 9
    assert _count_class_occurrences(svgfile, 'mem_tile_buffers') == 4


def test_viz_k2in_2out_outmpt():
    class SingleKernel_2_2_MPT_outbuff(AppBuilder):
        def __init__(self):
            self.kernel = Kernel(k_2in_2out, kernel_behavior_2_2)
            self.mtbi0 = MTPassThrough()
            self.mtbi1 = MTPassThrough()
            super().__init__()

        def callgraph(self, x_in0, x_in1, x_out0, x_out1):
            x0, x1 = self.kernel(x_in0, x_in1)
            x_out0[:] = self.mtbi0(x0)
            x_out1[:] = self.mtbi1(x1)

    x_in0 = np.zeros(shape=(32, 1), dtype=np.uint16)
    x_in1 = np.zeros(shape=(32, 1), dtype=np.uint16)
    x_out0 = np.zeros(shape=(32, 1), dtype=np.uint16)
    x_out1 = np.zeros(shape=(32, 1), dtype=np.uint16)

    app_builder = SingleKernel_2_2_MPT_outbuff()
    _ = app_builder.to_metadata(x_in0, x_in1, x_out0, x_out1)
    app_builder.save(svgfile := f'{imgdir}{app_builder.name}.svg')
    assert _count_class_occurrences(svgfile, 'kernel') == 2
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == 8
    assert _count_class_occurrences(svgfile, 'mem_connections') == 9
    assert _count_class_occurrences(svgfile, 'mem_tile_buffers') == 4


def test_viz_scaleup_2buffers_2kernels_mt():
    class MultiKernelScaleup_2_2_mt(AppBuilder):
        def __init__(self):
            self.concat0 = MTConcat()
            self.concat1 = MTConcat()
            self.split0 = MTSplit(2)
            self.split1 = MTSplit(2)
            self.kernel0 = [Plus1() for _ in range(2)]
            self.kernel1 = [PlusN() for _ in range(2)]
            super().__init__()

        def callgraph(self, x_in0, x_in1, x_out0, x_out1):
            xs0 = self.split0(x_in0)
            xs1 = self.split1(x_in1)

            xo0 = []
            xo1 = []
            for i in range(2):
                xo0.append(self.kernel0[i](xs0[i], xs0[i].nbytes))
                xo1.append(self.kernel1[i](xs1[i], xs1[i].nbytes, 3))

            x_out0[:] = self.concat0(xo0)
            x_out1[:] = self.concat1(xo1)

    x_in0 = np.zeros(shape=(512, 1), dtype=np.uint8)
    x_in1 = np.zeros(shape=x_in0.shape, dtype=x_in0.dtype)
    x_out0 = np.zeros(shape=x_in0.shape, dtype=x_in0.dtype)
    x_out1 = np.zeros(shape=x_in0.shape, dtype=x_in0.dtype)

    app_builder = MultiKernelScaleup_2_2_mt()
    _ = app_builder.to_metadata(x_in0, x_in1, x_out0, x_out1)
    app_builder.save(svgfile := f'{imgdir}{app_builder.name}.svg')
    assert _count_class_occurrences(svgfile, 'kernel') == 8
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == 16
    assert _count_class_occurrences(svgfile, 'mem_connections') == 9
    assert _count_class_occurrences(svgfile, 'mem_tile_buffers') == 16
