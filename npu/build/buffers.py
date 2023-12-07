# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from .port import BufferPort
from .kernelmeta import KernelMeta

  
class Buffer(KernelMeta):
    """This class is a superclass used for IT Buffers and MT Buffer objects.  The class
       contains a numpy array and any slices used to index into that array.  These slices
       need tracked as data moving to/from IT and MT buffers will result in DMA buffer descriptors
       based on those slices.

    Attributes
    ----------
    _array : ndarray
        The underlying buffer.
    slices : list
        TAn ordered list of slices to index into the underlying array.       
    """
    @classmethod
    def to_ndarray(self, a):
        if isinstance(a, np.ndarray):
            return a
        else:
            return a.array


    def __init__(self, array, slices, name, buffertiletype,disable_unique_name_id=False) -> None:
        """Return a new Buffer object."""     
        self._array = array
        self.slices = list() if slices is None else slices.copy()

        super().__init__(name, buffertiletype, "buffer", buffertiletype, disable_unique_name_id=disable_unique_name_id)

    def createports(self,num_inputs, num_outputs, port_slices=None):
        """Create a set of bufferports for this Buffer object."""
        ios = ['in'] * num_inputs +  ['out'] * num_outputs
        portslices = [None] * num_inputs + [None] * num_outputs if port_slices is None else port_slices

        for io, portslice_list in zip(ios,portslices):
            bp = BufferPort(f"{self.ttype}{io}", "uint8 *", self._array, portslice_list, io=io, parent=self)
            self.ports.append(bp)


    def _validate_inputbuffer(self,inputargs):
        if isinstance(inputargs, list):
            in_nbytes = sum([a.nbytes for a in inputargs])
        else:
            in_nbytes = inputargs.nbytes

        if in_nbytes != self.nbytes:
            raise ValueError(f'input number of bytes {in_nbytes} does not match mtbuffer size {self.nbytes}')


    def __getitem__(self, val, cls):
        raise LookupError(f'Shouldnt be calling getitem on a Buffer...instead use a BufferPort which can be sliced')

    def __setitem__(self, key, value):
        raise LookupError(f'This Buffer type does not support slicing ...')

    @property
    def array(self):
        """return the array after applying the tracked slices."""
        a = self._array
        for s in self.slices:
            a = a[s]
        return a

    @property
    def nbytes(self):
        return self.array.nbytes

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype    
    
    @array.setter
    def array(self, newarray):
        raise ValueError(f'should not be setting array value')

    def to_metadata(self):
        d = self._to_kbase_metadata()
        d['shape'] = self.shape
        d['dtype'] = self.dtype.name
        return d


class ITBuffer(Buffer):
    """An Interface Tile Buffer subclass from Buffer."""
    @classmethod
    def ndarray_to_itbuffer(self, a, setitem_itkernel=None):
        return ITBuffer(a, setitem_itkernel=setitem_itkernel) if isinstance(a, np.ndarray) else a

    @classmethod
    def ndarray_to_itb_port(self, a, setitem_itkernel=None):
        return ITBuffer(a, setitem_itkernel=setitem_itkernel).outputbufferports[0] if isinstance(a, np.ndarray) else a

    def __init__(self, array, slices=None, setitem_itkernel=None) -> None:
        super().__init__(array, slices, "itbuffer", "IT")
        self.createports(1,1)
        self.setitem_itkernel = setitem_itkernel
        
    def __setitem__(self, key, value):
        if self.setitem_itkernel is None:
            raise AttributeError(f'This ITBuffer had setitem called, but this object does not implement a setitem kernel for tracing.')

        _ = self.setitem_itkernel(value, bufref=self.outputbufferports[0][key])

            

class MTBuffer(Buffer):
    """A Memory Tile Buffer subclass from Buffer."""
    def __init__(self, array, slices=None, reuse_buffername = None, mtmode:str="passthrough") -> None:
        disable_unique_name_id = reuse_buffername is not None
        mtbname = reuse_buffername if disable_unique_name_id else "mtbuffer"
        self.mtmode = mtmode

        super().__init__(array, slices, mtbname, "MT", disable_unique_name_id=disable_unique_name_id)  

    def to_metadata(self):
        d = self._to_kbase_metadata()
        d['shape'] = self.shape
        d['dtype'] = self.dtype.name
        d['mtmode'] = self.mtmode
        return d

