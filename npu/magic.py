# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from IPython.core.magic import cell_magic, Magics, magics_class
from IPython import get_ipython
from IPython.display import HTML
from IPython.display import display_javascript
from npu.build.kernel import Kernel
import os

@magics_class
class kernel_magic(Magics):
    def __init__(self, shell)->None:
        """ Keep track of all the kernel code """
        super(kernel_magic,self).__init__(shell)

    @cell_magic
    def kernel(self, _, cell:str):
        """
    Specify a compute tile C++ kernel and return an npu.build.Kernel object.

    This cell magic command allows users to input C++ kernel code within
    a Jupyter notebook cell. It then returns a corresponding Kernel object
    that can be compiled into an object file for use in a Riallto application.
    The Cpp source must return a void type, input and output buffers are specified
    as pointer types, as parameters are specified with non-pointer types.

    Header files included in the directory where the notebook is are permitted.

    Parameters
    ----------
    cell : str
        The string content of the cell, expected to be C++ code defining
        the kernel.

    Returns
    -------
    Kernel : object
        Returns a Kernel object that has the same name as the last function
        defined in the cell magic. See npu.build.Kernel.

    Examples
    --------
    In a Jupyter notebook %%kernel cell

    
    void passthrough(uint8_t *in_buffer, uint8_t* out_buffer, uint32_t nbytes) {
        for(int i=0; i<nbytes; i++) {
            out_buffer[i] = in_buffer[i];
            
        }
        
    }
    


    This will construct a passthrough npu.build.Kernel object that can be used within
    a callgraph to construct a complete application. 

    """
        try:
            # Parse magic code
            kern = Kernel(cell, requires_boilerplate=True)            
            kern.kb.srcpath = os.getcwd() 
            kern.kb.getheaders = True # grab headers from cwd
            self.shell.user_ns.update({kern.ktype : kern})
        except RuntimeError as r:
            return HTML("<pre>Kernel C/C++ FAILED\n" + r.args[0] + "</pre>")

instance = get_ipython()

if instance:
    get_ipython().register_magics(kernel_magic)
    import nest_asyncio
    nest_asyncio.apply()
