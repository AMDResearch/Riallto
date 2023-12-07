# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np


class Port:
    """This class holds the attributes for a Kernel port in an AppBuilder application.  

    Attributes
    ----------
    name : str
        Name of this connection constructed from snk/src names.
    pdtype : str
        The port datatype.
    io : str
        The port direction, either 'in' or 'out'.
    ctype : str
        The connection type of this port.
    parent : Kernel
        The Kernel object that contains this Port.
    c_dtype : str
        Type information for the connection.

    """
    def __init__(self, name, pdtype=None, io=None, ctype=None, parent=None, c_dtype=None) -> None:
        """Return a new Port object.""" 
        self.name = name
        self.pdtype = pdtype
        self.io = io
        self.ctype = ctype
        self.parent = parent
        self.c_dtype = c_dtype

    def _to_pbase_metadata(self):
        d = {}
        d['c_dtype'] = self.c_dtype
        d['direction'] = self.io
        d['name'] = self.name
        d['ctype'] = None
        return d

    def to_metadata(self):
        """ produce a dict of metadata for the port """
        return self._to_pbase_metadata()

    @property
    def metadata(self):
        from npu import ReprDict
        return ReprDict(self.to_metadata(), rootname=self.name)

class RTPPort(Port):
    """This class represents a Run Time Parameter (RTP) port to a Kernel.
    
    Attributes
    ----------
    
    value : int
        the RTP's value that is updated by userspace writes.
    
    """

    def __init__(self, name, value=None, parent=None, c_dtype=None) -> None:
        """Return a new RTPPort object.""" 
        super().__init__(name, value, "in", "rtp", parent, c_dtype=c_dtype)
        self.value = value

    def copy(self, newparent=None):
        if newparent is None:
            newparent = self.parent

        return RTPPort(self.name, self.value, newparent, c_dtype=self.c_dtype)

    def to_metadata(self):
        d = self._to_pbase_metadata()
        d['ctype'] = 'rtp'
        d['value'] = self.value
        return d
    

class BufferPort(Port):
    """This class represents a kernel port that contains a buffer.  That buffer is held as a NumPy
    array and can be sliced to collect an offset into the array if DMA transfers are used to 
    move the buffer.  All IT and MT ports are bufferports, whereas CT kernels can have a 
    mix of RTP and BufferPorts.

    Attributes
    ----------
    _array : ndarray
        The underlying buffer.
    slices : list
        TAn ordered list of slices to index into the underlying array.       
    """

    def __init__(self, name, pdtype=None, array=None, slices=None, io=None, ctype=None, parent=None) -> None:
        """Return a new Buffer object.""" 
        super().__init__(name, pdtype, io, ctype, parent)
        self._array = array
        self.slices = list() if slices is None else slices.copy()

        self._validate_array()

    def _validate_array(self):
        if self._array is not None and not isinstance(self.array, np.ndarray):
            raise TypeError(f'BufferPort with slices {self.slices} against underlying array {self._array} is not a ndarray')

    def copy(self, newparent=None, newslices=None):
        if newparent is None:
            newparent = self.parent

        if newslices is None:
            newslices = self.slices

        return BufferPort(self.name, self.pdtype, self._array, newslices, self.io, self.ctype, parent=newparent)

    def __getitem__(self, val):
        newslices = self.slices.copy()

        if val != slice(None,None,None):
            '''ignore [:] slicing for sequence building'''
            newslices.append(val)

        return self.copy(newslices=newslices)

    def __setitem__(self, key, value):
        self.parent[key] = value

    @property
    def array(self):
        """Returns the array after all slices have been applied."""
        a = self._array
        for s in self.slices:
            a = a[s]
        return a

    @array.setter
    def array(self, newarray):
        """Kernel bufferport shapes are overwritable, Interface and MemTile buffers just take values.""" 
        if self.parent.ttype == 'CT':
            self._array = newarray
        elif self.parent.ttype == 'MT' or self.parent.ttype == 'IT':
            self.array[:] = newarray
        else:
            raise ValueError(f"Unable to set array contents for {self.name}'s Parent Type: {type(self.parent)} with ttype of {self.parent.ttype}")

    @property    
    def offset(self):
        """Return the byte offset into the original array after slices are applied."""
        if self.array is None:
            return None

        slicelow, _ = np.byte_bounds(self.array) 
        arraylow, _ = np.byte_bounds(self._array)
        return slicelow - arraylow

    @property
    def nbytes(self):
        return None if self.array is None else self.array.nbytes

    @property
    def shape(self):
        return None if self.array is None else self.array.shape

    def to_metadata(self):
        d = self._to_pbase_metadata()
        if self.array is not None:
            d['shape'] = self.array.shape 
            d['dtype'] = self.array.dtype.name 
        else:
            d['shape'] = None 
            d['dtype'] = None 
        return d
