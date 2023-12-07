# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import shutil
import tempfile
import subprocess
from . import BUILD_TEMPLATE_PATH

class WSLBuilder:
    """This class contains methods to call into WSL to build NPU CT kernels and applications.
    
    Attributes
    ----------
    _tmp_dir : TemporaryDirectory
        The temporary directory where builds will occur.    
    
    """
    def __init__(self) -> None:
        """Return a new WSLBuilder object.""" 
        self._tmp_dir = tempfile.TemporaryDirectory(suffix=None, prefix="pnx", dir=None)
        if not os.path.exists(self._tmp_dir.name):
            raise RuntimeError(f"Unable to create temporary directory {self._tmp_dir.name} for build files")

        self.build_path = os.path.join(self._tmp_dir.name, "build_template")

        # Try and copy the build template folder to temp directory
        try:
            if not os.path.exists(self.build_path):
                shutil.copytree(BUILD_TEMPLATE_PATH, self.build_path)
        except:
            raise RuntimeError(f"Unable to copy build template from {BUILD_TEMPLATE_PATH} to {self.build_path}")

    def _wslcall(self, cmd, arglist=[], debug=False) -> None:
        """Call the requested cmd and print out any output text."""
        cmdlist = cmd.split(" ") + arglist
        try:
            output = subprocess.check_output(cmdlist, stderr=subprocess.STDOUT)
            if debug:
                print(f"{output.decode('UTF-8').rstrip()}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR! WSL failed \n\n{e.output.decode()}") 
            raise e
