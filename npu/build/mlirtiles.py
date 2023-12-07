# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

class Tile:
    """This class represents a Tile in MLIR.  IT, MT and CT Tiles are supported.

    Attributes
    ----------
    tloc : tuple
        The x,y coordinates of this tile.
    tvarname : str
        The MLIR variable name for this tile.
    """
    def __init__(self, x, y):
        """Return a new Buffer object.""" 
        self.tloc = (x, y)
        self.tvarname = f'%tile{self.tloc[0]}{self.tloc[1]}'  

    def to_mlir_tile_declare(self, indent=''):
          return f'{indent}{self.tvarname} = AIE.tile{self.tloc}'  


class CTTile(Tile):
    """This class represents a CT Tile in MLIR.

    Attributes
    ----------
    tloc : tuple
        The x,y coordinates of this tile.
    kernel : dict
        Dictionary describing the kernel placed on this tile.
    objfifos_produce : list
        list of objectfifos used as a producer by this tile.
    objfifos_consume : list
        list of objectfifos used as a consumer by this tile.        
    """

    def __init__(self, x, y, kernel=None) -> None:
        """Return a new CTTile object.""" 
        super().__init__(x,y)
        self.kernel = kernel
        self.objfifos_produce = list()
        self.objfifos_consume = list()

        self.cvarname = f'%core{self.tloc[0]}{self.tloc[1]}'

    @property
    def objfifos(self):
        return self.objfifos_produce + self.objfifos_consume

    def is_used(self):
        return self.kernel is not None

    def to_mlir_tile_declare(self, indent=''):
        s = f'{indent}{self.tvarname} = AIE.tile{self.tloc}' 
        has_rtp = any(e['ctype'] == 'rtp' for e in self.kernel["ports"].values())
        if has_rtp:
            s += f'\n{indent}%rtp_{self.tloc[0]}_{self.tloc[1]} = AIE.buffer({self.tvarname}) {{ sym_name = "rtp_{self.tloc[0]}_{self.tloc[1]}" }} : memref<{len(self.kernel["ports"])}xi32>'
        return s

    def to_mlir(self, indent="   "):
        """Return the MLIR string to represent this tile."""
        if self.kernel is None:
            return ""

        if len(self.objfifos_consume) == 0 or len(self.objfifos_produce) == 0:
            raise ValueError(f'AIE.core({self.tvarname}) either has no inputs or output objfifos')

        tilestr  = f'{indent}AIE.core({self.tvarname}) {{\n'
        tilestr += f'{indent*2}%c0 = arith.constant 0 : index\n'
        tilestr += f'{indent*2}%c1 = arith.constant 1 : index\n'
        tilestr += f'{indent*2}%intmax = arith.constant 0xFFFFFFFF : index\n'
        tilestr += f'{self._to_mlir_rtp_index_constants(indent*2)}'
        tilestr += f'{indent*2}scf.for %arg3 = %c0 to %intmax step %c1 {{\n'

        for obf in self.objfifos_consume:
            tilestr += f'{obf.to_mlir_acquire("Consume",indent*3)}\n'

        for obf in self.objfifos_produce:
            tilestr += f'{obf.to_mlir_acquire("Produce",indent*3)}\n'
        tilestr += '\n'

        tilestr += f'{self._to_mlir_rtp_load(indent*3)}'

        tilestr += f'{indent*3}{self._to_mlir_funccall()}\n'

        for obf in self.objfifos_consume:
            tilestr += f'{obf.to_mlir_release("Consume",indent*3)}\n'

        for obf in self.objfifos_produce:
            tilestr += f'{obf.to_mlir_release("Produce",indent*3)}\n'

        tilestr += f'      }}\n'
        tilestr += f'{indent*2}AIE.end\n'
        tilestr += f'{indent}}} {{ link_with="{self.kernel["ktype"]}.o" }}\n\n'

        return tilestr

    def _to_mlir_channel_vars_types(self):
        chvars = list()
        chtypes = list()
        for p in self.kernel['ports']:
            if self.kernel['ports'][p]["ctype"] == "rtp":
                chvars.append(f"%{p}")
                chtypes.append("i32") #TODO: might want non int RTP types?
            else:
                for obf in self.objfifos:
                    port_fullname = f'{self.kernel["name"]}___{p}'
                    if obf.name.endswith(port_fullname) or obf.name.startswith(port_fullname):
                        chvars.append(obf.elementname)
                        chtypes.append(obf.memref)
                        break
                    if (self.kernel['name'], p) in obf.dsts:
                        chvars.append(obf.elementname)
                        chtypes.append(obf.memref)
                        break
        return chvars, chtypes

    def to_mlir_kernel_declare(self):
        _, chtypes = self._to_mlir_channel_vars_types()
        return f'func.func private @{self.kernel["ktype"]}({", ".join(chtypes)}) -> ()'

    def _to_mlir_rtp_index_constants(self, indent=''):
        chvars, chtypes = self._to_mlir_channel_vars_types()
        s =''
        for i in range(len(chvars)):
            if chtypes[i] == "i32":
                s += f'{indent}%c_rtpidx_{i} = arith.constant {i} : index\n'     
        return s

    def _to_mlir_rtp_load(self, indent=''):
        chvars, chtypes = self._to_mlir_channel_vars_types()
        s =''
        for i in range(len(chvars)):
            if chtypes[i] == "i32":
                s += f'{indent}{chvars[i]} = memref.load %rtp_{self.tloc[0]}_{self.tloc[1]}[%c_rtpidx_{i}] : memref<{len(self.kernel["ports"])}xi32>\n'     
        return s

    def _to_mlir_funccall(self):
        chvars, chtypes = self._to_mlir_channel_vars_types()

        return f'func.call @{self.kernel["ktype"]}({", ".join(chvars)}) : ({", ".join(chtypes)}) -> ()\n'


class BufferTile(Tile):
    """Class to represent IT and MT Tiles in MLIR."""
    def __init__(self, x, y, buffers=None) -> None:
        super().__init__(x,y)    
        self.buffers = buffers if buffers is not None else dict()

    def is_used(self):
        return len(self.buffers) != 0

class MEMTile(BufferTile):
    """Class to represent MT Tiles in MLIR."""
    def __init__(self, x, y, buffers=None) -> None:
        super().__init__(x, y, buffers)     

class ITTile(BufferTile):
    """Class to represent IT Tiles in MLIR."""
    def __init__(self, x, y, buffers=None) -> None:
        super().__init__(x, y, buffers)      
