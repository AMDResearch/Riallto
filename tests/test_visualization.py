# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from pathlib import Path
import pytest
import numpy as np
from npu.build.kernel import Kernel
from npu.build.mtkernel import MTPassThrough, MTSplit, MTConcat
from npu.build.appbuilder import AppBuilder
from npu.lib.graphs.graph_1ct import RGB720pBuilder
from npu.lib import Rgba2Hue, Rgba2Gray, Gray2Rgba, BitwiseAnd
from npu.lib import InRange, RgbaRtpThres, ThresholdRgba
import re
import random
import string

imgdir = str(Path(__file__).parent / "images") + '/'
x_in = np.zeros(shape=(720, 1280, 4), dtype=np.uint8)
x_out = np.zeros(shape=(720, 1280, 4), dtype=np.uint8)


def _count_class_occurrences(svgfile, classname):
    pattern = re.compile(f'class="{classname}"')
    with open(svgfile, 'r', encoding='utf-8') as file:
        content = file.read()
    matches = pattern.findall(content)
    return len(matches)


@pytest.mark.parametrize('kernel', [RgbaRtpThres, ThresholdRgba])
def test_RGB720pBuilder(kernel):
    app_builder = RGB720pBuilder(kernel=kernel())
    svgfile = f'{imgdir}RGB720pBuilder{str(kernel().name)}.svg'
    app_builder.save(svgfile)
    assert _count_class_occurrences(svgfile, 'kernel') == 2
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == 4


def test_single_kernel():
    class SingleKernelOneCTOneIT(AppBuilder):
        def __init__(self):
            self.kernel = RgbaRtpThres()
            super().__init__()

        def callgraph(self, x_in, x_out):
            for t in range(1):
                x = self.kernel(x_in[t], x_in.shape[1]*x_in.shape[2], 128,
                                128, 128)
                x_out[t] = x

    app_builder = SingleKernelOneCTOneIT()
    _ = app_builder.to_metadata(x_in, x_out)
    app_builder.save(svgfile := f'{imgdir}{app_builder.name}.svg')
    assert _count_class_occurrences(svgfile, 'kernel') == 2
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == 4


ct2ct = [['non_neighbor_down', (0, 5), (0, 2)],
         ['non_neighbor_up', (0, 2), (0, 4)],
         ['neighbor_down', (0, 4), (0, 3)],
         ['neighbor_up', (0, 2), (0, 3)]]


@pytest.mark.parametrize('app', ct2ct)
def test_ct2ct(app):
    class Pipeline(AppBuilder):
        def __init__(self):
            self.rgba2hue = Rgba2Hue()
            self.in_range = InRange()
            self.rgba2hue.tloc = app[1]
            self.in_range.tloc = app[2]
            super().__init__()

        def callgraph(self, x_in, x_out):
            for t in range(1):
                x = self.rgba2hue(x_in[t], 1280*4)
                x = self.in_range(x, 1280, 0, 79)
                x_out[t] = x

    app_builder = Pipeline()
    x_out1 = np.zeros(shape=(720, 1280), dtype=np.uint8)
    _ = app_builder.to_metadata(x_in, x_out1)
    app_builder.save((svgfile := imgdir + app_builder.name + '_' + app[0] + '.svg'))
    assert _count_class_occurrences(svgfile, 'kernel') == 4
    aiebuff = 8 if 'non_' in app[0] else 6
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == aiebuff


@pytest.mark.parametrize('down', [True, False])
def test_color_detect(down):
    class ColorDetectApplication(AppBuilder):
        def __init__(self):
            self.rgba2hue = Rgba2Hue()
            self.in_range = InRange()
            self.gray2rgba = Gray2Rgba()
            self.bitwiseand = BitwiseAnd()
            self.mtbi = MTPassThrough()
            if not down:
                self.rgba2hue.tloc = (0, 2)
                self.in_range.tloc = (0, 3)
                self.gray2rgba.tloc = (0, 4)
                self.bitwiseand.tloc = (0, 5)
            super().__init__()

        def callgraph(self, x_in, x_out):
            for t in range(1):
                y = self.mtbi(x_in[t])
                x = self.rgba2hue(y, 1280*4)
                x = self.in_range(x, 1280, 0, 79)
                x = self.gray2rgba(x, 1280)
                x = self.bitwiseand(x, y, 1280*4)
                x_out[t] = x

    app_builder = ColorDetectApplication()
    _ = app_builder.to_metadata(x_in, x_out)
    svgfile = f"{imgdir}{app_builder.name}_{('down' if down else 'up')}.svg"
    app_builder.save(svgfile)
    assert _count_class_occurrences(svgfile, 'kernel') == 8
    assert _count_class_occurrences(svgfile, 'mem_tile_buffers') == 2
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == 12


@pytest.mark.parametrize('scale', [1, 2, 4])
def test_color_scaledup(scale):
    class ScaledUpThresholdApplication(AppBuilder):
        def __init__(self):
            self.split = MTSplit(scale)
            self.concat = MTConcat()
            self.ks = [RgbaRtpThres() for _ in range(scale)]
            super().__init__()

        def callgraph(self, xin, xout):
            r_thresh = [51, 149, 128, 19]
            g_thresh = [66, 12, 95, 128]
            b_thresh = [0, 128, 17, 33]
            for t in range(720):
                xs = self.split(xin[t])
                for i in range(scale):
                    xs[i] = self.ks[i](xs[i], 1280*4//scale, r_thresh[i],
                                       g_thresh[i], b_thresh[i])
                x = self.concat(xs)
                xout[t] = x

    app_builder = ScaledUpThresholdApplication()
    _ = app_builder.to_metadata(x_in, x_out)
    app_builder.save(svgfile := f'{imgdir}{app_builder.name}_x{scale}.svg')
    assert _count_class_occurrences(svgfile, 'kernel') == scale * 2
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == scale * 4


@pytest.mark.parametrize('dual', [True, False])
def test_mtpassthrough(dual):
    class SimpleMemTilePassthrough(AppBuilder):
        def __init__(self):
            self.mtbi = MTPassThrough()
            self.mtbo = MTPassThrough()
            self.k = RgbaRtpThres()
            self.k.tloc = (0, 3)
            super().__init__()

        def callgraph(self, x_in, x_out):
            for t in range(1):
                x = self.mtbi(x_in[t])
                x = self.k(x, 1280*4, 128, 128, 128)
                if dual:
                    x = self.mtbo(x)
                x_out[t] = x

    app_builder = SimpleMemTilePassthrough()
    _ = app_builder.to_metadata(x_in, x_out)
    svgfile = f"{imgdir}{app_builder.name}{('_dual' if dual else '')}.svg"
    app_builder.save(svgfile)
    assert _count_class_occurrences(svgfile, 'kernel') == 2
    assert _count_class_occurrences(svgfile, 'mem_tile_buffers') == 2 + 2 * dual
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == 4


def test_mixed_kernels_scaledup():
    class ScaledUpMixedKernelsApplication(AppBuilder):
        def __init__(self):
            self.ks0 = [Rgba2Hue() for _ in range(2)]
            self.ks1 = [Rgba2Gray() for _ in range(2)]
            self.split = MTSplit(4)
            self.concat = MTConcat()
            super().__init__()

        def callgraph(self, xin, xout):
            for t in range(720):
                xs = self.split(xin[t])
                for i in range(4):
                    if (i % 2) == 0:
                        xs[i] = self.ks0[i//2](xs[i], 1280)
                    else:
                        xs[i] = self.ks1[i//2](xs[i], 1280)

                xout[t] = self.concat(xs)

    x_out1 = np.zeros(shape=(720, 1280), dtype=np.uint8)
    app_builder = ScaledUpMixedKernelsApplication()
    _ = app_builder.to_metadata(x_in, x_out1)
    app_builder.save(svgfile := f'{imgdir}{app_builder.name}.svg')
    assert _count_class_occurrences(svgfile, 'kernel') == 8
    assert _count_class_occurrences(svgfile, 'mem_tile_buffers') == 16
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == 16


@pytest.mark.parametrize('tloc', ['up', 'down', 'nonneighboring'])
def test_df_pipeline_scaledup(tloc):
    class ScaledUpDfPipelineApplication(AppBuilder):
        def __init__(self):
            self.split = MTSplit(2)
            self.concat = MTConcat()

            self.ks0 = [Rgba2Hue(), Rgba2Gray()]
            self.ks1 = [Gray2Rgba() for _ in range(2)]
            if tloc == 'down':
                self.ks0[0].tloc = (0, 5)
                self.ks0[1].tloc = (0, 3)
                self.ks1[0].tloc = (0, 4)
                self.ks1[1].tloc = (0, 2)
            elif tloc == 'up':
                self.ks0[0].tloc = (0, 4)
                self.ks0[1].tloc = (0, 2)
                self.ks1[0].tloc = (0, 5)
                self.ks1[1].tloc = (0, 3)
            else:
                self.ks0[0].tloc = (0, 5)
                self.ks0[1].tloc = (0, 2)
                self.ks1[0].tloc = (0, 3)
                self.ks1[1].tloc = (0, 4)

            super().__init__()

        def callgraph(self, xin, xout):
            xo = [None] * 2
            for t in range(x_in.shape[0]):
                xs = self.split(xin[t])
                for i in range(2):
                    size = x_in.shape[1]*x_in.shape[2] // 2
                    xo[i] = self.ks0[i](xs[i], size)
                    xo[i] = self.ks1[i](xo[i], size//4)

                xout[t] = self.concat(xo)

    app_builder = ScaledUpDfPipelineApplication()
    _ = app_builder.to_metadata(x_in, x_out)
    app_builder.save(svgfile := f'{imgdir}{app_builder.name}_{tloc}.svg')
    assert _count_class_occurrences(svgfile, 'kernel') == 8
    assert _count_class_occurrences(svgfile, 'mem_tile_buffers') == 8
    aiebuff = 16 if tloc == 'nonneighboring' else 12
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == aiebuff


@pytest.mark.parametrize('randomname', [False, True])
def test_dataparallel(randomname):
    """Test a data parallel application with buffers with random names"""
    kernel_src = '''
        #include <aie_api/aie.hpp>
        #define N 720
        extern "C" {
        void passthrough(uint8_t *data_in, uint8_t *data_out) {
            for(int i=0; i < N; i++) {
                data_out[i] = data_in[i];
            }
        }
        } // extern "C"
    '''

    def _random_string():
        letters = string.ascii_letters
        return ''.join(random.choice(letters) for i in range(16))

    inname, outname = 'data_in', 'data_out'
    if randomname:
        inname, outname = _random_string(), _random_string()
        kernel_src = kernel_src.replace('data_in', inname)
        kernel_src = kernel_src.replace('data_out', outname)

    def setval_behavioral(obj):
        objout = getattr(obj, outname)
        objin = getattr(obj, inname)
        objout.array = objin.array

    class DataParallelPassthrough(AppBuilder):
        def __init__(self):
            self.split = MTSplit(4)
            self.concat = MTConcat()
            self.ks = [Kernel(kernel_src, setval_behavioral) for _ in range(4)]
            super().__init__()

        def callgraph(self, xin, xout):
            for t in range(720):
                xs = self.split(xin[t])
                for i in range(4):
                    xs[i] = self.ks[i](xs[i])
                x = self.concat(xs)
                xout[t] = x

    app_builder = DataParallelPassthrough()
    _ = app_builder.to_metadata(x_in, x_out)
    svgfile = f'{imgdir}{app_builder.name}_{inname}_{outname}.svg'
    app_builder.save(svgfile)

    assert _count_class_occurrences(svgfile, 'kernel') == 8
    assert _count_class_occurrences(svgfile, 'mem_tile_buffers') == 16
    assert _count_class_occurrences(svgfile, 'aie_tile_buffers') == 16
