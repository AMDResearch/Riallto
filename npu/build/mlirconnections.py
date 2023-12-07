# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from typing import Tuple, List, Set, Dict

class MLIRConnect:
    """This class represents a connection in MLIR between kernels.  Broadcast connections are supported.

    Attributes
    ----------
    name : str
        MLIR variable name of this connection.
    id : int
        Incrementing unique id for this connection.  
    src : tuple
        The source kernel and port.
    dsts : list
        List of destination sink kernel and ports.
    nbytes : int
        Number of bytes transferred on this connection.
    offset : int
        Buffer offset for the source data transfer.
    config : tuple
        Tuple containing number of CT, MT, IT tiles that can be used to build the MLIR.
    """

    unique_id = -1

    @classmethod
    def next_id(cls):
        cls.unique_id += 1 
        return cls.unique_id
    
    @classmethod
    def reset_id(cls):
        cls.unique_id = 0    

    def __init__(self, name:str, src:Tuple[str,str], dsts:Set[Tuple[str,str]], nbytes:int, offset:int, apptiles:Dict, appkernels:Dict) -> None:
        """Return a new MLIRConnect object."""    
        self.name = name
        self.id = MLIRConnect.next_id()
        self.src = src 
        self.dsts = dsts
        self.nbytes = nbytes
        self.offset = offset

        self.srctile = self._init_tile_src(apptiles, appkernels)
        self.snktiles = self._init_tile_dsts(apptiles, appkernels)

        if self.nbytes % 4 != 0:
            raise ValueError(f'Cannot move non 4B array: {self.name} {self.nbytes}')

        self.memref = f'memref<{self.nbytes//4}xi32>'
        self.varname = f'@{self.name}'

    def _init_tile_src(self, apptiles, appkernels):
        srctile = apptiles[appkernels[self.src[0]]['tloc']]
        return srctile

    def _init_tile_dsts(self, apptiles, appkernels)->Set:
        snktiles = set() 
        for d in self.dsts:
            snktiles.add(apptiles[appkernels[d[0]]['tloc']])           
        return snktiles
  

class ObjectFIFO(MLIRConnect):
    """This class represents an object fifo connection in MLIR between kernels. 

    Attributes
    ----------
    elementname : str
        MLIR objectfifo element name.
    subviewname : str
        MLIR objectfifo subview name.
    """

    def __init__(self, name:str, src:Tuple[str,str], dsts:Set[Tuple[str,str]], nbytes:int, offset:int, apptiles:Dict, appkernels:Dict) -> None:   
        super().__init__(name, src, dsts, nbytes, offset, apptiles, appkernels)
        self.elementname = f'%elem{self.id}'
        self.subviewname = f'%subview{self.id}'

    def to_mlir_declare(self):
        """Return the MLIR text for declaring this object"""
        snktiles='{'
        for i,d in enumerate(self.snktiles):
            snktiles += f'{d.tvarname}'
            if i < len(self.snktiles) - 1:
                snktiles += ', '
        snktiles+='}'

        if len(self.snktiles) == 1:
            depths = '2 : i32'
        else:
            depths = '[' + '2,'*(len(self.snktiles)) + '2]' 
                
        return f'AIE.objectFifo {self.varname}({self.srctile.tvarname}, {snktiles}, {depths}) : !AIE.objectFifo<{self.memref}>'

    def to_mlir_acquire(self, io, indent):
        """Return the MLIR text for objectfifo acquire calls."""
        if io not in ["Consume", "Produce"]:
             raise ValueError(f'OjectFIFO - specify io as either "Consume" or "Produce" instead of {io}')

        s =  f'{indent}{self.subviewname} = AIE.objectFifo.acquire {self.varname}({io}, 1) : !AIE.objectFifoSubview<{self.memref}>\n'
        s += f'{indent}{self.elementname} = AIE.objectFifo.subview.access {self.subviewname}[0] : !AIE.objectFifoSubview<{self.memref}> -> {self.memref}'
        return s

    def to_mlir_release(self, io, indent):
        """Return the MLIR text for objectfifo release calls."""
        if io not in ["Consume", "Produce"]:
             raise ValueError(f'OjectFIFO - specify io as either "Consume" or "Produce" instead of {io}')

        return f'{indent}AIE.objectFifo.release {self.varname}({io}, 1)'   
