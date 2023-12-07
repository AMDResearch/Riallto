# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
from . import wslpath
from .wslbuilder import WSLBuilder
from .utils import wsl_prefix

class AppXclbinBuilder(WSLBuilder):
    """This class manages the WSL building of the application, resulting in
       the final xclbin binary that is run on the NPU.

    Attributes
    ----------
    xclbin : str
        Path to the NPU executable xclbin file.
    buildlog : str
        Path to the WSL build logfile.
    """

    def __init__(self) -> None:
        """Return a new AppXclbinBuilder object."""        
        super().__init__()
        self.xclbin = None
        self.buildlog = None

    def build(self, appname : str, mlir_file : str, kernels, debug=False):
        """Toplevel call to build the xclbin
        
        Parameters
        ----------
        appname : string
            The desired final name of the xclbin.
        mlir_file : string
            The filepath containing the MLIR to compile.
        kernels : list
            The list of kernels that need to be compiled first before building the xclbin.
        debug : boolean
            Optional setting to deliver extra logging and debug messages from the build.       

        Returns
        -------
        None
        
        """
        self._wslcall(f"{wsl_prefix()}bash", [f"{wslpath(self.build_path)}/check_license.sh"])
        
        self.build_kernels(kernels, debug)
        self.buildlog = os.path.join(self.build_path, "build.log")

        print("Building the xclbin...")
        self._wslcall(f"{wsl_prefix()}cp", [f"{wslpath(os.path.abspath(mlir_file))}", f"{wslpath(self.build_path)}/aie.mlir"], debug)
        self._wslcall(f"{wsl_prefix()}bash", [f"{wslpath(self.build_path)}/app_build.sh", f"{appname}"], debug)
        self._wslcall(f"{wsl_prefix()}cp", [f"{wslpath(self.build_path)}/final.xclbin", f"./{appname}.xclbin"], debug)
        self._wslcall(f"{wsl_prefix()}cp", [f"{wslpath(self.build_path)}/final.seq", f"./{appname}.seq"], debug)
        print(f"Successfully Building Application... {appname}.xclbin & {appname}.seq delivered")

        self.xclbin = f"{self.build_path}\final.xclbin.o"

    def build_kernels(self, kernels, debug):       
        """ Build the ComputeTile kernel object files if not done already."""

        # unique the list of kernels to be built
        kernel_types = {k.ktype : k for k in kernels if k.ttype == "CT"}
        for _, k in kernel_types.items():
            k.build(debug)
            objpath = wslpath(k.kb.buildobjpath)
            appobjpath =  wslpath(os.path.join(self.build_path, f'{k.ktype}.o'))
            self._wslcall(f"{wsl_prefix()}cp", [f"{objpath}", f"{appobjpath}"], debug)
