# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from pathlib import Path
import pytest
import numpy as np
from npu.build.mtkernel import MTPassThrough, MTSplit, MTConcat
from npu.build.appbuilder import AppBuilder
from npu.lib.graphs.graph_1ct import RGB720pBuilder
from npu.lib import Rgba2Hue, Rgba2Gray, Gray2Rgba, BitwiseAnd
from npu.lib import InRange, RgbaRtpThres, ThresholdRgba

imgdir = str(Path(__file__).parent / "images") + '/'
x_in = np.zeros(shape=(720, 1280, 4), dtype=np.uint8)
x_out = np.zeros(shape=(720, 1280, 4), dtype=np.uint8)


@pytest.mark.parametrize('kernel', [RgbaRtpThres, ThresholdRgba])
def test_RGB720pBuilder(kernel):
    app_builder = RGB720pBuilder(kernel=kernel())
    app_builder.save(f'{imgdir}RGB720pBuilder{str(kernel().name)}.svg')


def test_single_kernel():
    class SingleKernel(AppBuilder):
        def __init__(self):
            self.kernel = RgbaRtpThres()
            super().__init__()

        def callgraph(self, x_in, x_out):
            for t in range(1):
                x = self.kernel(x_in[t], x_in.shape[1]*x_in.shape[2], 128,
                                128, 128)
                x_out[t] = x

    app_builder = SingleKernel()
    _ = app_builder.to_metadata(x_in, x_out)
    app_builder.save(f'{imgdir}single_ct_and_it.svg')


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
    app_builder.save(imgdir + app[0] + '.svg')


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

    app_bldr = ColorDetectApplication()
    _ = app_bldr.to_metadata(x_in, x_out)
    app_bldr.save(f"{imgdir}ColorDetectApplication_{('down' if down else 'up')}.svg")


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
    app_builder.save(f'{imgdir}ScaledUpThresholdApplication_x{scale}.svg')


@pytest.mark.parametrize('dual', [True, False])
def test_mtpassthrough(dual):
    class SimpleMemTileApplication(AppBuilder):
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

    app_builder = SimpleMemTileApplication()
    _ = app_builder.to_metadata(x_in, x_out)
    app_builder.save(f"{imgdir}memtile_passthrough{('_dual' if dual else '')}.svg")


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
                    if (i%2) == 0:
                        xs[i] = self.ks0[i//2](xs[i], 1280)
                    else:
                        xs[i] = self.ks1[i//2](xs[i], 1280)

                xout[t] = self.concat(xs)

    x_out1 = np.zeros(shape=(720, 1280), dtype=np.uint8)
    app_builder = ScaledUpMixedKernelsApplication()
    _ = app_builder.to_metadata(x_in, x_out1)
    app_builder.save(f'{imgdir}ScaledUpMixedKernelsApplication.svg')


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
    app_builder.save(f'{imgdir}ScaledUpDfPipelineApplication_{tloc}.svg')
