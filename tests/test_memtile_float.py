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
