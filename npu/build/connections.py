# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from .buffers import Buffer
from .port import Port
from .userspace import UserspaceRTP
from .buffers import BufferPort

class Connection():
    """This class holds the source and sink objects for a connection within
       an NPU application.  Sources and sinks can be IT, MT, or CTs and
       the name of a connection can be split up to obtain dictionary keys for the 
       src, src_port, snk and snk_port which is then used in AppBuilder tracing. 

    Notes
    -----
        Connections are point to point links.  Broadcast links are inferred by analyzing multiple Connection objects.
        Connection names are built as srcname__srcportname___snkname___snkportname.

    Attributes
    ----------
    srckernel : str
        The source kernel (IT, MT, or CT) for this connection.
    srcport : str
        The source port for this connection.
    sinkkernel : str
        The sink kernel (IT, MT, or CT) for this connection.
    sinkport : str
        The sink port for this connection.
    name : str
        Name of this connection constructed from snk/src names.
    dtype : str
        Type information for the connection.
    ctype : str
        The implementation type of this connection, E.g. 'rtp', 'objfifo,pingpong'.
    """

    def __init__(self, src_sink_tuple) -> None:
        """Return a new Buffer object.""" 
        self._srcobj, self._sinkobj = src_sink_tuple

        self.srckernel, self.srcport = self._parseport(self._srcobj, src_n_sink=True)
        self.sinkkernel, self.sinkport = self._parseport(self._sinkobj, src_n_sink=False)

        self.name = f'{self.srckernel.name}___{self.srcport.name}___{self.sinkkernel.name}___{self.sinkport.name}'

        self.kernels = [self.srckernel, self.sinkkernel]
        self.ports = [self.srcport, self.sinkport]

        self.dtype = self.srcport.pdtype
        self.ctype = None

        self._validate_connection()

    def _validate_connection(self):
            if  isinstance(self.srcport, BufferPort):
                if self.srcport.nbytes != self.sinkport.nbytes:
                    raise ValueError(f'Connection found with unequal number of src, snk bytes\nconnection: {self.name} \
                                     sending {self.srcport.nbytes} bytes, but expecting {self.sinkport.nbytes} bytes')

    def _parseport(self, obj, src_n_sink=True):
        """Based on object type, return the kernel, port pair."""
        if isinstance(obj, Buffer):
            if src_n_sink:
                return obj, obj.ports[-1]
            else:
                return obj, obj.ports[0]

        if isinstance(obj, Port):
            return obj.parent, obj

        if isinstance(obj, int):
            userspace_rtp = UserspaceRTP(obj)
            return userspace_rtp, userspace_rtp.ports[0]

        raise TypeError(f"obj of type {type(obj)} not support in Connections")
