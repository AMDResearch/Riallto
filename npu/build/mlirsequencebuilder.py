# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
from typing import Tuple, List
from dataclasses import dataclass, field
import numpy as np

STATE_DIR = os.path.dirname(__file__)

@dataclass
class UBDataMovement:
    ubname: str
    symname: str
    shape: tuple
    dtype: str
    tilesizes: List[int] = field(default_factory=lambda: [])
    dim: List[int] = field(default_factory=lambda: [])
    srcoffsets: List[int] = field(default_factory=lambda: [])
    snkoffsets: List[int] = field(default_factory=lambda: [])


class MLIRSequnceBuilder:
    """This class generates the MLIR Sequence dialect that describes datamovement to and from
    The NPU.  This is accomplished by analyzing the datamovement for IT Buffers discovered by
    running the AppBuilder callgraph.  The required sync signals are also generated.

    Attributes
    ----------
    _metadata : AppMetadata
        The application metadata.
    aietiles : list
        List of AIE tiles in this application.
    _userbuffers : dict
        Dictionary of IT Buffers discovered during application tracing.
    _ubname2externid  : dict
        Dictionary of unique incrementing IDs for IT Buffers.
    _ingress_ub : dict
        Dictionary of incoming IT Buffers to the NPU array.
    _egress_ub : dict
        Dictionary of outgoing IT Buffers from the NPU array.
    _ingress_egress_ub : dict
        Dictionary of IT Buffers if the buffer is both input and output to the NPU array.
    _constants_table : dict
        Dictionary of constants used in the MLIR sequence specification.
    _cons_broadcasts : list
        List of broadcast connection names.
    """

    def __init__(self, app_metadata, aietiles, cons_broadcasts)->None:
        """Return a new SequenceBuilder object."""
        self._metadata = app_metadata
        self.aietiles = aietiles
        self._userbuffers = {}
        self._ubname2externid = {}
        self._ingress_ub = {}
        self._egress_ub = {}
        self._ingress_egress_ub = {}
        self._constants_table = {}
        self._cons_broadcasts = cons_broadcasts

        self._populate_ingress_egress()
        self._check()
        self._populate_constants_table()

    @property
    def mlir(self)->str:
        """ Generates the MLIR sequence dialect from the callgraph traced sequence."""
        s = f"func.func @sequence({self._to_seq_portsig()}) {{\n"
        s += f"{self._generate_constants(indent='    ')}\n"
        s += f"{self._generate_rtps(indent='    ')}\n"

        for _,ub in self._egress_ub.items():
            s += f"{self._generate_ub_memcpy_nd(ub.ubname, 1, ub, indent='    ')}\n"

        for _,ub in self._ingress_ub.items():
            s += f"{self._generate_ub_memcpy_nd(ub.ubname, 0, ub, indent='    ')}\n"

        s += f"{self._generate_sync(indent='    ')}"

        s += f"    return\n"
        s += f"}}\n"
        return s

    def _generate_sync(self, indent='')->str:
        s = f'{indent}AIEX.ipu.sync {{'
        s += 'column = 0 : i32, row = 0 : i32,'
        s += ' direction = 0 : i32, channel = 0 : i32,'
        s += ' column_num = 1 : i32, row_num = 1 : i32 }\n'
        return s


    def _analyse_transfers(self, ub:UBDataMovement)->Tuple[List[int], List[int], List[int]]:
        """" Returns a list of transfer lengths too and from a userbuffer."""
        sizes = []
        num_dims = []
        steps = []

        def _visitor(dim):
            for d in dim:
                if isinstance(d, tuple):
                    num_dims = len(d)
                    if len(d) > 2:
                        raise RuntimeError("Currently restricted to 2D shim transfers")
                    sizes.append((d[0].stop - d[0].start) * (d[1].stop - d[1].start))
                elif isinstance(d, list):
                    _visitor(d)
                elif isinstance(d, slice):
                    sizes.append(d.stop - d.start)
                    if not d.step is None:
                        raise RuntimeError("Stepping when doing shim transfers not currently allowed")
                else:
                    sizes.append(max(ub.tilesizes))

        _visitor(ub.dim)
        return (sizes, steps, num_dims)

    def _check_transfers(self, ub:UBDataMovement)->None:
        sizes, steps, num_dims = self._analyse_transfers(ub)
        if len(set(sizes)) > 1:
            raise RuntimeError("[No current support] transfer sizes changing over the course of the sequence.")
        if len(set(num_dims)) > 1:
            raise RuntimeError("[No current support] Dimensionality of the sequence is changing over the course of the sequence.")
        if not self._transfer_monotonically_increasing(ub.srcoffsets):
            raise RuntimeError("[No current support] Iterating over the output/input on the IT data mover must be monotonically increasing.")
        if not self._transfer_monotonically_increasing(ub.snkoffsets):
            raise RuntimeError("[No current support] Iterating over the output/input on the IT data mover must be monotonically increasing.")

    def _transfer_monotonically_increasing(self, offset_list:List[int])->bool:
        if len(offset_list) > 1:
            diff_amount = offset_list[1] - offset_list[0]
            if diff_amount < 0:
                return False
            else:
                for i in range(len(offset_list)-1):
                    if offset_list[i+1] < offset_list[i]:
                        return False
        return True

    def _get_first_transfer(self, ub):
        if len(ub.dim) > 0:
            return ub.dim[0][0]
        else:
            raise RuntimeError("user buffer with no transfers?")

    def _extract_static_data_movement_pattern(self, ub)->List[List[List[int]]]:
        """ Extracts the static datamovement pattern from the sequence, otherwise
        throws and error that the sequence is changing over time, which we do not currently
        support.

        Output has the form:
        [ [ O3, O2, O1, O0],  <-- Offsets for each dimension
          [ L3, L2, L1, L0],  <-- Lengths for each dimension
          [ S3, S2, S1 ]   ]  <-- Strides for each dimension S0 implicitly 1

        """
        offsets = [0,0,0,0]
        lengths = [0,0,0,0]
        strides = [0,0,0]

        self._check_transfers(ub)
        init = self._get_first_transfer(ub)
        ub_shape = self._change_to_int32(ub.shape, ub.dtype)
        if isinstance(init, int):
            init = self._change_to_int32_offset(init, ub.dtype)
            offsets[0] = 0 if init <= 0 else (init - 1)
            lengths[0] = ub_shape[-1]
            lengths[1] = 1 if len(ub_shape) < 2 else ub_shape[-2]
            lengths[2] = 1 if len(ub_shape) < 3 else ub_shape[-3]
            lengths[3] = 1 if len(ub_shape) < 4 else ub_shape[-4]
        elif isinstance(init, slice):
            offsets[0] = self._change_to_int32_offset(init.start, ub.dtype)
            lengths[0] = ub_shape[-1]
            lengths[1] = 1 if len(ub_shape) < 2 else ub_shape[-2]
            lengths[2] = 1 if len(ub_shape) < 3 else ub_shape[-3]
            lengths[3] = 1 if len(ub_shape) < 4 else ub_shape[-4]
        elif isinstance(init, tuple):
            if len(init) > 2:
                raise RuntimeError("Not yet supporting transfers with more dimensions than 2")
            offsets[0] = init[0].start
            offsets[1] = init[1].start
            lengths[3] = int(ub.shape[0]/(init[0].stop - init[0].start))
            lengths[2] = int(ub.shape[1]/(init[1].stop - init[1].start))
            lengths[1] = init[0].stop - init[0].start
            lengths[0] = int((init[1].stop - init[1].start)/4)
            strides[2] = (init[0].stop - init[0].start)*int((ub.shape[1])/4)
            strides[1] = int((init[1].stop - init[1].start)/4)
            strides[0] = int((ub.shape[1])/4)
        else:
            raise RuntimeError(f"Error, unexpected transfer type {type(init)} transfer={init}")

        return [offsets, lengths, strides]

    def _generate_memcpy_nd_from_transfer(self, t:List[List[int]])->str:
        o = t[0]
        l = [1 if (x == 0) or (x == 1) else x for x in t[1]]
        s = t[2]
        ret =  f'[%c{o[3]}, %c{o[2]}, %c{o[1]}, %c{o[0]}]'
        ret += f'[%c{l[3]}, %c{l[2]}, %c{l[1]}, %c{l[0]}]'
        ret += f'[%c{s[2]}, %c{s[1]}, %c{s[0]}]'
        return ret

    def _change_to_int32_offset(self, offset, dtype):

        if dtype == "bf16":
            itemsize = 2
        else:
            itemsize = int(str(dtype)[1:])//8

        if offset % 4 != 0:
            raise ValueError(f"Must be divisible by 4 {offset=}")

        if itemsize > 4:
            offset *= (itemsize//4)
        else:
            offset //= (4//itemsize)

        return offset

    def _change_to_int32(self, shape, dtype):

        mod_shape = np.zeros(shape).squeeze()
        new_shape = list(mod_shape.shape[:])

        if dtype == "bf16":
            itemsize = 2
        else:
            itemsize = int(str(dtype)[1:])//8


        if not new_shape and itemsize == 4:
            return (1,)

        if (new_shape[-1]*itemsize) % 4 != 0:
            raise ValueError(f'Lowest dimension number of bytes has to be '
                             f'divisible by 4. lowest dimension={new_shape[-1]}'
                             f' bytes per item={itemsize}')

        if itemsize > 4:
            new_shape[-1] *= (itemsize//4)
        else:
            new_shape[-1] //= (4//itemsize)

        converted_shape = np.zeros(tuple(new_shape)).squeeze().shape
        return converted_shape if converted_shape else (1,)

    def _generate_ub_memref(self, ub:UBDataMovement)->str:
        new_shape = self._change_to_int32(ub.shape, ub.dtype)
        s = ''
        for d in new_shape:
            s += f'{d}x'
        return s + f'i32'

    def _generate_ub_memcpy_nd(self, extern_buff:str, extern_id:int, ub:UBDataMovement, indent='')->str:
        """ Generates the MLIR dma_memcpy_nd command from the UBDataMovement info."""
        self._check_transfers(ub)
        transfers = self._extract_static_data_movement_pattern(ub)
        symname = self._get_buffer_symname(ub)
        s =  f'{indent}AIEX.ipu.dma_memcpy_nd(%c0, %c0,'
        s += f'%{extern_buff}{self._generate_memcpy_nd_from_transfer(transfers)})'
        s += f'{{ metadata= @{symname}, id = {self._ubname2externid[ub.ubname]} : i32 }} :'
        s += f'(i32, i32, memref<{self._generate_ub_memref(ub)}>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])\n'
        return s

    def _get_buffer_symname(self, ub:UBDataMovement)->str:
        namesplit = ub.symname.split("___")
        srck, srcp = (namesplit[0], namesplit[1])
        if (srck, srcp) in self._cons_broadcasts:
            return f"{srck}__{srcp}"
        else:
            return ub.symname

    def _generate_rtps(self, indent='')->str:
        s = ''
        used_aie_tiles = [tile for _, tile in self.aietiles.items() if tile.is_used()]
        # Add the RTP write
        for aie in used_aie_tiles:
            has_rtp = any(e['ctype'] == 'rtp' for e in aie.kernel["ports"].values())
            if has_rtp:
                rtp_idx = 0
                for pname, p in aie.kernel["ports"].items():
                    if p['ctype'] == 'rtp':
                        sig = rtp_idx
                        s += f'{indent}AIEX.ipu.rtp_write({aie.tloc[0]}, {aie.tloc[1]}, {rtp_idx}, {sig}) {{ buffer_sym_name = "rtp_{aie.tloc[0]}_{aie.tloc[1]}" }}\n'
                    rtp_idx = rtp_idx + 1
        return s

    def _to_seq_portsig(self)->str:
        """ Generates the port signature for the sequence func.func call in the generated MLIR."""
        s = ''
        for i,ub in enumerate(self._ingress_egress_ub.values()):
            s += f"%{ub.ubname} : memref<{self._generate_ub_memref(ub)}>"
            if i != len(self._userbuffers)-1:
                s += ","
        return s


    def _populate_constants_table(self)->None:
        self._constants_table = {}
        self._add_constant(0)
        self._add_constant(1)
        ubs = {**self._ingress_ub, **self._egress_ub}
        for ub in ubs.values():
            self._add_constant(max(ub.tilesizes))
            self._add_constant(len(ub.dim))
            for s in ub.shape:
                self._add_constant(int(s/4))
            for s in self._change_to_int32(ub.shape, ub.dtype):
                self._add_constant(s)
            m = self._extract_static_data_movement_pattern(ub)
            for m_d in m:
                for m_d_d in m_d:
                    self._add_constant(m_d_d)

    def _add_constant(self, cval:int)->None:
        cname:str = f"%c{cval}"
        if cname not in self._constants_table:
            self._constants_table[cname] = cval

    def _generate_constants(self, indent='')->str:
        """ generate all the constants that need to be used in the mlir gen."""
        s = ""
        for cname, cval in self._constants_table.items():
            s += f"{indent}{cname} = arith.constant {cval} : i32\n"
        return s

    def _filter_ub(self)->None:
        """ Filters the metadata for the userbuffers."""
        self._userbuffers = {}
        self._ubname2externid = {}
        externid = 0
        for _,k in self._metadata['kernels'].items():
            if k['type'] == "IT":
                self._userbuffers[k["name"]] = k
                self._ubname2externid[k["name"]] = externid
                externid = externid + 1

    def _get_ub_dtype_mlir_str(self, ub)->str:
        typemap = {
                "bfloat16" : "bf16",
                "uint8"    : "i8",
                "uint16"   : "i16",
                "uint32"   : "i32",
                "uint64"   : "i64",
                "int8"     : "i8",
                "int16"    : "i16",
                "int32"    : "i32",
                "int64"    : "i64",
                "float32"  : "f32",
                "float64"  : "f64"
        }
        if not ub['dtype'] in typemap:
            raise RuntimeError(f"Unable to generate mlir string for np dtype {ub['dtype']}")
        return typemap[ub['dtype']]

    def _populate_ingress_egress(self)->None:
        """ populates the dict of ingress and egress userbuffers."""
        self._filter_ub()
        self._ingress_ub = {}
        self._egress_ub = {}
        for s in self._metadata["sequence"]:
            if s["srckernelname"] in self._userbuffers:
                if s["srckernelname"] not in self._ingress_ub:
                    c_ub = self._userbuffers[s['srckernelname']]
                    self._ingress_ub[s["srckernelname"]] = UBDataMovement(
                            ubname=s["srckernelname"],
                            symname=s["name"],
                            shape= c_ub['shape'],
                            dtype=self._get_ub_dtype_mlir_str(c_ub))

                self._ingress_ub[s["srckernelname"]].tilesizes.append(s["nbytes"])
                self._ingress_ub[s["srckernelname"]].srcoffsets.append(s["srcoffset"])
                self._ingress_ub[s["srckernelname"]].snkoffsets.append(s["snkoffset"])
                if len(s["srcslices"]) == 0:
                    self._ingress_ub[s["srckernelname"]].dim.append([0])
                else:
                    self._ingress_ub[s["srckernelname"]].dim.append(s["srcslices"])

            if s["snkkernelname"] in self._userbuffers:
                if s["snkkernelname"] not in self._egress_ub:
                    self._egress_ub[s["snkkernelname"]] = UBDataMovement(
                            ubname=s["snkkernelname"],
                            symname=s["name"],
                            shape=self._userbuffers[s['snkkernelname']]['shape'],
                            dtype=self._get_ub_dtype_mlir_str(c_ub))

                self._egress_ub[s["snkkernelname"]].tilesizes.append(s["nbytes"])
                self._egress_ub[s["snkkernelname"]].srcoffsets.append(s["srcoffset"])
                self._egress_ub[s["snkkernelname"]].snkoffsets.append(s["snkoffset"])
                if len(s["snkslices"]) == 0:
                    self._egress_ub[s["snkkernelname"]].dim.append([0])
                else:
                    self._egress_ub[s["snkkernelname"]].dim.append(s["snkslices"])
        self._ingress_egress_ub = {**self._ingress_ub, **self._egress_ub}

    def _check(self)->None:
        for _,ub in self._ingress_ub.items():
            self._check_ub(ub)

        for _,ub in self._egress_ub.items():
            self._check_ub(ub)


    def _check_ub(self, ub:UBDataMovement)->None:
        if not all(len(d) <= 1 for d in ub.dim):
            raise RuntimeError("Currently only support two dimensions (support coming soon)")
        if not len(set(ub.tilesizes)) == 1:
            raise RuntimeError("Currently we only support datamovement to and from a userbuffer where the tile size is static across the whole sequence (support coming soon)")
