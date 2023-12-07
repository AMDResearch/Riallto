# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from .port import BufferPort, RTPPort

class KernelMeta:
    """This class is a superclass for all Kernels in this package - including IT, MT, and CT kernels.  it keeps track of unique names for 
    kernels as they are instantiated and has helper functions to identify and iterate over kernel ports.

    Attributes
    ----------
    name : str
        The name of this kernel.
    shortname : str
        an abbreviated name for use in application metadata.
    ktype : str
        The kernel type - will be defined by the operation performed (E.g. 'mtpassthrough' for a MemoryTile passthrough operation).         
    ttype : str
        The kernel tile type - valid values are 'IT', 'MT' or 'CT'.  
    ports : list
        The kernel ports which are defined by the kernel operation. 
    tloc : tuple
        The requested Tile location for this kernel as an (x,y) pair.
    disable_unique_name_id : boolean
        Optional flag for disabling a unique name for this kernel.  If set, the name input argument will be used.
    """
    used_names = dict()

    @classmethod
    def unique_name(cls, kernelname):
        if kernelname not in cls.used_names:
            cls.used_names[kernelname] = 0 
        else:
            cls.used_names[kernelname] += 1
        return f'{kernelname}_{cls.used_names[kernelname]}' 
    
    @classmethod
    def reset_unique_names(cls):
        cls.used_names = dict()  

    def __init__(self, name, shortname, ktype, ttype, ports=None, tloc=None, disable_unique_name_id=False) -> None:
        """Return a new KernelMeta object.""" 
        self.name = name if disable_unique_name_id else self.unique_name(name)

        self.shortname = shortname
        self.ktype = ktype
        self.ttype = ttype
        self.tloc = tloc

        self.ports = ports if ports is not None else list()

        for p in self.ports:
            p.parent = self
            setattr(self, p.name, p)

    def _to_kbase_metadata(self):
        """ Produces the base metadata for the kernel, can be appended to in subclasses """
        from npu.build.port import RTPPort
        from npu.build.port import BufferPort
        d = {}
        d['name'] = self.name
        d['tloc'] = self.tloc
        d['ttype'] = self.ttype
        d['ktype'] = self.ktype
        d['type'] = self.ttype
        d['ports'] = {}
        for p in self.ports:
            d['ports'][p.name] = p.to_metadata()
        return d

    def to_metadata(self):
        """ Produces a dict of the metadata for the kernel """
        return self._to_kbase_metadata()

    @property
    def metadata(self):
        from npu import ReprDict
        return ReprDict(self.to_metadata(), rootname=self.ktype)

    @property
    def bufferports(self):
        return [p for p in self.ports if isinstance(p, BufferPort)]

    @property
    def inputbufferports(self):
        return [p for p in self.bufferports if p.io == "in"]
    
    @property
    def outputbufferports(self):
        return [p for p in self.bufferports if p.io == "out"]    

    @property    
    def rtpports(self):
        return [p for p in self.ports if isinstance(p, RTPPort)]

