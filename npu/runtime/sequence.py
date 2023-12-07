# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import ctypes
import numpy as np
import os
from typing import List, Dict, NamedTuple
from collections import namedtuple
from dataclasses import dataclass, field

opcode2op = {
    0: "Op0",
    1: "Op1",
    2: "SetRTP",
    3: "Op3",
    4: "Op4",
    5: "Op5",
    6: "Op6",
    7: "Op7",
    8: "Op8",
    9: "Op9",
    10: "Op10",
    11: "Op11",
    12: "Op12",
    13: "Op13"
}

op2opcode = {
   "Op0" : 0,
   "Op1" : 1,
   "SetRTP" : 2,
   "Op3" : 3,
   "Op4" : 4,
   "Op5" : 5,
   "Op6" : 6,
   "Op7" : 7,
   "Op8" : 8,
   "Op9" : 9,
    "Op10" : 10,
    "Op11" : 11,
    "Op12" : 12,
    "Op13": 13
}

op_num_words = {
    "Op0" : 1,
    "Op1" : 9,
    "SetRTP" : 3,
    "Op3" : 2,
    "Op4" : 11,
    "Op5" : 10,
    "Op6" : 10,
    "Op7" : 11,
    "Op8" : 11,
    "Op10" : 2,
    "Op11" : 9,
    "Op12" : 9,
    "Op13" : 3

}

def parse_word(s:str)->int:
    """ Parsed either an int string or hex string or raises an error. """
    try:
        return int(s,16)
    except:
        try:
            return int(s)
        except:
            raise ValueError(f"Error, unable to parse {s} as a hex value or a base-10 integer")


class Coord(NamedTuple):
    """ A coordinate of a location within the array (CT/MT/IT)."""
    row:int
    col:int

def createOpBin(opcode:int, coords:Coord, bdId:int):
    """ Takes the opcode/coordiates/BdId and constructs a 32-bit binary op word."""
    ret = 0
    ctypes.c_uint32(~ret).value
    ret = (opcode << 24) | (coords.col << 16) | (coords.row << 8) | bdId
    return ret

@dataclass
class Operation:
    """ A dataclass that defines an operation within a sequence.

    Attributes
    ----------
    opcode : int
        The 16bit binary opcode for this operation.
    words: int
        The number of 32-bit words this operation consumes from the sequence.
    bdId : int
        The bdId (if there is one) associated with this operation.
    coords: Coords
        The Coordinates that this operation applies to (CT/MT/IT)
    config: List[ctypes.c_uint32]
        A list of words that make up this entire sequence.
    """
    opcode:int = 0
    words:int = 1
    bdId:int = 0 
    coords:Coord = Coord(row=0, col=0) 
    config:List[ctypes.c_uint8] = field(default_factory=list)

    @property
    def bin(self)->List[ctypes.c_uint32]:
        """ Returns the binary form of this operation. """
        return self.config

    @property
    def str(self)->str:
        """ Renders a string to describe this operation. """
        s='Operation '
        s+= f"row={self.coords.row} col={self.coords.col} bdId={self.bdId}"
        return s

@dataclass
class RTPOp(Operation):
    """ RTP operation dataclass for setting an RTP value. Inherits from Operation.

    Attributes
    ----------
    addr : ctypes.c_uint32
        A 32-bit relative address for the location of this RTP.
    value: ctypes.c_uint32
        A 32-bit value for the RTP.
    rtpidx : int
        The index for this RTP in the kernel argument (associated on first parse of sequence)
    """
    addr:ctypes.c_uint32 = 0
    value:ctypes.c_uint32 = 0
    opcode:int = 2
    words:int = 3
    rtpidx:int = 0

    @property
    def bin(self)->List[ctypes.c_uint32]:
        """ Returns the binary form of this operation. """
        b = []
        b.append(createOpBin(self.opcode, self.coords, self.bdId))
        b.append(self.addr)
        b.append(self.value)
        return b

    @property
    def str(self)->str:
        """ Renders a string to describe this operation. """
        s='RTPOp '
        s+= f"row={self.coords.row} col={self.coords.col} bdId={self.bdId} addr={hex(self.addr)} value={self.value}"
        return s

def OperationFactory(words:List[ctypes.c_uint32])->(Operation, List[ctypes.c_uint32]):
    """ Peel off the next operation in the sequence words and produce an instance of it.
    Currently only exposes operations relevant to RTP writes."""
    opword = words[0]
    op = ParseOpCodeString(opword)
    coords = ParseTileCoords(opword)
    bdid = ParseBDId(opword)

    if op == "SetRTP" and isCT(coords):
        return (RTPOp(coords=coords, addr=words[1], value=words[2], rtpidx=words[2]), words[3:])
    else:
        if not op in op_num_words: 
            raise RuntimeError(f"Unable to determine number of words to peel off for Op {op=}")
        num_words = op_num_words[op]
        return (Operation(opcode=op2opcode[op], words=num_words, bdId=bdid, 
                          coords=coords, config=words[0:num_words]), words[num_words:])   

def isMT(coord)->bool:
    """ Accepts coords, returns true if location is a memory tile. """
    return coord.row == 1

def isIT(coord)->bool:
    """ Accepts coords, returns true if location is an interface tile. """
    return coord.row == 0

def isCT(coord)->bool:
    """ Accepts coords, returns true if location is a compute tile. """
    return coord.row > 1

def ParseOpCodeString(word:ctypes.c_uint32)->str:
    """ From a word containing an opcode extract the opecode string name. """
    op = (word & 0xFF000000) >> 24
    if op not in opcode2op:
       raise RuntimeError(f"Unknown opcode {op} attempting to be parsed from {word}")
    return opcode2op[op]

def ParseTileCoords(word:ctypes.c_uint32)->Coord:
    """ Gets the column coord from the instruction. """
    col = (word & 0x00FF0000) >> 16
    row = (word & 0x0000FF00) >> 8
    return Coord(row=row, col=col)

def ParseBDId(word:ctypes.c_uint32)->int:
    """ Parses the BD ident for the op-code word. """
    bd = (word & 0x000000FF)
    return bd

class Sequence:
    """ Performs a minimal parsing of the sequence to determine RTP writes and expose them.
        If this is the first time that the sequence is being parsed, set by an optional 
        parameter, then the value of any RTP parsed will be used to set it's rtpindex and 
        build the mlir_rtps dictionary.

    Attributes
    ----------
        _in_file: str
            The input sequence file.
        _str_contents: str
            The contents of the sequence file in string form.
        config_words : list[ctypes.c_uint32]
            A list of sequence words in binary form.
        operations : list[Operations]
            A list of operation dataclasses that have been parsed from the binary sequence.
        mlir_rtps : dict
            A dictionary of RTPs that have been parsed and can be set at runtime.
    """
    def __init__(self, binseq_file:str, first_parse:bool=True)->None:
        self._in_file = binseq_file
        with open(self._in_file) as f:
            self._str_contents = f.readlines()
            self._str_contents = [x.strip() for x in self._str_contents]
        self.config_words = self._get_config_words()
        self.operations = self._parse_config_words(self.config_words)
        
        # If first parse associate sequence location with RTPs
        if first_parse:
            self.mlir_rtps = {}
            for op in self.operations:
                if isinstance(op, RTPOp):
                    if not (op.coords[1], op.coords[0]) in self.mlir_rtps:
                        self.mlir_rtps[(op.coords[1], op.coords[0])] = {}
                    self.mlir_rtps[(op.coords[1], op.coords[0])][op.rtpidx] = op
        
    def _parse_config_words(self, config_words)->List[Operation]:
        """ For a binary list of configuration words peel of each operation in the
        sequence and construct an Operation dataclass for each. Return a completed 
        list of those Operations."""
        ops:List[Operation] = []
        c = config_words
        while(len(c) > 0):
            op, c = OperationFactory(c)
            ops.append(op)
        return ops

    def _get_config_words(self)->None:
        """ returns all the 32 bit words for the configuration instructions. """
        self._parse_unused_rtp_section()
        config_words = []
        for i in range(len(self._str_contents) - self.num_rtp_words):
            config_word = parse_word(self._str_contents[i+self.num_rtp_words])
            ctypes.c_uint32(~config_word).value
            config_words.append(config_word)
        return config_words

    def _parse_unused_rtp_section(self)->None:
        """ Parses the initial unused section of the sequence related to legacy RTPs"""
        self.num_rtp_words = parse_word(self._str_contents[0])
        ctypes.c_uint32(~self.num_rtp_words).value
        self.rtp = []
        for i in range(self.num_rtp_words - 1):
            word = parse_word(self._str_contents[i+1])
            ctypes.c_uint32(~word).value

            # extract the RTP values
            rtp1 = word & 0x0000FFFF
            rtp2 = (word & 0xFFFF0000) >> 16
            self.rtp.append(rtp1)
            self.rtp.append(rtp2)
        self.rtp_op_code_idx = 0

    @property
    def buffer(self)->np.array:
        """ Renders a new np.array for the sequence that can be passed into the device """
        return np.array(self.bin, dtype=np.uint32)

    @property
    def rtp_words(self)->List[int]:
        """ From the rtp array pack them into 32bit words """
        rtp_words = [self.num_rtp_words]
        for i in range(0, len(self.rtp), 2):
            rtp_word = self.rtp[i] | (self.rtp[i+1] << 16)
            ctypes.c_uint32(~rtp_word).value
            rtp_words.append(rtp_word)
        return rtp_words


    @property
    def bin(self)->List[ctypes.c_uint32]:
        """ Returns the complete binary for the sequence """
        b = self.rtp_words
        for op in self.operations:
            for word in op.bin:
                b.append(word)
        return b

    def txt(self, filename, annotated=False)->None:
        """ Dumps the instructions to a txt file. If annotated is set to True then
        print a descriptive string for each operation next to the operation."""
        with open(filename, "w") as fp:
            if not annotated:
                for word in self.bin:
                    hexval = '0x{:08x}'.format(word)
                    fp.write(f"{hexval}\n")
            else:
                for w in self.rtp_words:
                    fp.write(f"{int(w)} # RTP\n")

                for op in self.operations:
                    first = True
                    for word in op.bin:
                        hexval = '0x{:08x}'.format(word)
                        if first:
                            fp.write(f"{hexval} # OP {op.str}\n")
                            first = False
                        else:
                            fp.write(f"{hexval}\n")

