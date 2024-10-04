import pytest
import numpy as np
from pathlib import Path
from ml_dtypes import bfloat16

from npu.build.appbuilder import AppBuilder
from npu.build.mtkernel import MTConcat, MTSplit
from npu.build.kernel import Kernel
from npu.lib import Plus1
from npu.runtime import AppRunner


kernel_src = Plus1().srccode
imgdir = str(Path(__file__).parent / "images") + '/'


def plus1_behavior(invobj):
    invobj.out_buffer.array = invobj.in_buffer.array


@pytest.mark.parametrize('datatype', [np.float32, bfloat16])
def test_memtile_distribute_join_4_non_anonymous(datatype):

    datatype_txt = str(np.dtype(datatype))

    if datatype == np.float32:
        datatype_txt = 'float'
    else:
        datatype_txt = 'bfloat16'

    kernel_src0 = kernel_src.replace('uint8_t', datatype_txt)

    class Plus1Float():
        def __new__(cls, *args):
            kobj = Kernel(kernel_src0, plus1_behavior)
            return kobj(*args) if len(args) > 0 else kobj

    class MtSplitConcat4AIEsNonAnonymousFloat(AppBuilder):
        def __init__(self):
            super().__init__()
            self.kernels = [Plus1Float() for _ in range(4)]
            self.mtbsplit = MTSplit(4)
            self.mtbconcat = MTConcat()

        def callgraph(self, x_in, x_out):
            new_xs = []
            xs = self.mtbsplit(x_in)
            for i in range(4):
                new_xs.append(self.kernels[i](xs[i], xs[i].nbytes))
            x_out[:] = self.mtbconcat(new_xs)

    size = 512
    array_in = np.zeros(shape=(4, size), dtype=datatype)
    array_out = np.zeros(shape=(4, size), dtype=datatype)

    trace_app = MtSplitConcat4AIEsNonAnonymousFloat()
    trace_app.build(array_in, array_out)
    trace_app.save(f'{imgdir}{trace_app.name}_{datatype_txt}.svg')
    app = AppRunner(f"{trace_app.name}.xclbin")

    test_data = np.random.randn(*array_in.shape).astype(datatype)
    bo_in = app.allocate(shape=(4, size), dtype=datatype)
    bo_out = app.allocate(shape=(4, size), dtype=datatype)

    bo_in[:] = test_data
    bo_in.sync_to_npu()

    app.call(bo_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out).astype(np.float32)

    del app

    assert np.allclose((test_data + 1.).astype(np.float32), test_out,
                       rtol=1e-02)


vec_sum_src = '''
#include <aie_api/aie.hpp>

extern "C" {
    void reduced_sum(bfloat16 *in_buffer, bfloat16* out_buffer, uint32_t elements) {
        ::aie::vector<bfloat16, 32> buffer;
        ::aie::vector<bfloat16, 32> acc = ::aie::zeros<bfloat16, 32>();
        uint16_t loop_count = (elements) >> 5;
        for(int j=0; j<loop_count; j++) {
            buffer = ::aie::load_v<32>(in_buffer);
            acc = ::aie::add(acc, buffer);
            in_buffer += 32;
        }
        auto sum = aie::reduce_add(acc);
        out_buffer[0] = sum;
        out_buffer+=1;
    }
}
'''


@pytest.mark.parametrize('datatype', [np.float32, bfloat16])
def test_memtile_sum_mtsplit(datatype):

    datatype_txt = 'float' if datatype == np.float32 else 'bfloat16'
    vec_sum_src0 = vec_sum_src.replace('bfloat16', datatype_txt)

    class VectorSum():
        def __new__(cls, *args):
            kobj = Kernel(vec_sum_src0, cls.behavioralfx)
            return kobj(*args) if len(args) > 0 else kobj

        def behavioralfx(self):
            self.out_buffer.array = np.zeros((16), dtype=datatype)
            self.out_buffer.array[0] = np.sum(self.in_buffer.array)

    class MtSplitConcat4AIEsSum(AppBuilder):
        def __init__(self):
            super().__init__()
            self.kernels = [VectorSum() for _ in range(4)]
            self.mtbsplit = MTSplit(4)
            self.mtbconcat = MTConcat()

        def callgraph(self, x_in, x_out):
            new_xs = []
            xs = self.mtbsplit(x_in)
            for i in range(4):
                new_xs.append(self.kernels[i](xs[i], xs[i].shape[0]))
            x_out[:] = self.mtbconcat(new_xs)

    size = 256
    array_in = np.zeros(shape=(size), dtype=datatype)
    array_out = np.zeros(shape=(64), dtype=datatype)

    trace_app = MtSplitConcat4AIEsSum()
    trace_app.build(array_in, array_out)
    trace_app.save(f'{imgdir}{trace_app.name}_{datatype_txt}.svg')
    app = AppRunner(f"{trace_app.name}.xclbin")

    test_data = np.random.randn(*array_in.shape).astype(datatype)
    bo_in = app.allocate(shape=array_in.shape, dtype=datatype)
    bo_out = app.allocate(shape=array_out.shape, dtype=datatype)

    bo_in[:] = test_data
    bo_in.sync_to_npu()

    app.call(bo_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out)

    del app

    for idx, val in enumerate(np.split(test_data, 4)):
        assert np.isclose(np.sum(val), test_out[idx*16], rtol=0.25)


kernel_avrg_src = '''
#include <aie_api/aie.hpp>

extern "C" {
    void average(int8_t *in_buffer, bfloat16* out_buffer, uint32_t elements) {
        ::aie::vector<int8_t, 32> buffer;
        ::aie::accum<acc32, 32> acc;
        acc.from_vector(::aie::zeros<int8_t, 32>());
        uint16_t loop_count = (elements) >> 5;
        for(int j=0; j<loop_count; j++) {
            buffer = ::aie::load_v<32>(in_buffer);
            acc = ::aie::add(acc, buffer);
            in_buffer += 32;
        }
        auto sum = aie::reduce_add(acc.to_vector<int32_t>());
        out_buffer[0] = bfloat16(float(sum)/float(elements));
        out_buffer+=1;
    }
}
'''


@pytest.mark.parametrize('datatype', [np.float32, bfloat16])
def test_memtile_average_mtsplit(datatype):

    datatype_txt = 'float' if datatype == np.float32 else 'bfloat16'
    kernel_avrg_src0 = kernel_avrg_src.replace('bfloat16', datatype_txt)

    class Average():
        def __new__(cls, *args):
            kobj = Kernel(kernel_avrg_src0, cls.behavioralfx)
            return kobj(*args) if len(args) > 0 else kobj

        def behavioralfx(self):
            self.out_buffer.array = np.zeros((16), dtype=datatype)
            self.out_buffer.array[0] = np.average(self.in_buffer.array)

    class MtSplitConcat4AIEsAverage(AppBuilder):
        def __init__(self):
            super().__init__()
            self.kernels = [Average() for _ in range(4)]
            self.mtbsplit = MTSplit(4)
            self.mtbconcat = MTConcat()

        def callgraph(self, x_in, x_out):
            new_xs = []
            xs = self.mtbsplit(x_in)
            for i in range(4):
                new_xs.append(self.kernels[i](xs[i], xs[i].shape[0]))
            x_out[:] = self.mtbconcat(new_xs)

    size = 512
    array_in = np.zeros(shape=(size), dtype=np.int8)
    array_out = np.zeros(shape=(64), dtype=datatype)

    trace_app = MtSplitConcat4AIEsAverage()
    trace_app.build(array_in, array_out)
    trace_app.save(f'{imgdir}{trace_app.name}_{datatype_txt}.svg')
    app = AppRunner(f"{trace_app.name}.xclbin")

    np.random.seed(4102024)
    test_data = np.random.randint(-128, 127, size=array_in.shape,
                                  dtype=array_in.dtype)
    bo_in = app.allocate(shape=array_in.shape, dtype=array_in.dtype)
    bo_out = app.allocate(shape=array_out.shape, dtype=datatype)

    bo_in[:] = test_data
    bo_in.sync_to_npu()

    app.call(bo_in, bo_out)

    bo_out.sync_from_npu()
    test_out = np.array(bo_out)

    del app

    for idx, val in enumerate(np.split(test_data, 4)):
        assert np.isclose(np.average(val), test_out[idx*16], rtol=0.2)
