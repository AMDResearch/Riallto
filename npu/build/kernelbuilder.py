# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
from . import wslpath
from .wslbuilder import WSLBuilder
from .utils import wsl_prefix
import hashlib
import glob
import shutil


KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib', 'kernels', 'cpp')

class KernelObjectBuilder(WSLBuilder):
    """This class builds ComputeTile kernel C/C++ into object files for linking into applications.
       There is also caching support so that a kernel is only built one-time.

    Attributes
    ----------
    name : str
        The name of the kernel.
    srccode : str
        The C/C++ source code of the ComputeTile kernel.
    srcfile : str
        The C/C++ source file path.

    Notes
    -----
    The cache of object files already built can be cleared by running KernelObjectBuilder.clear_cache().
    """

    prebuilt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib', 'cached')

    @classmethod
    def clear_cache(cls):
        if os.path.exists(cls.prebuilt_path) and glob.glob(os.path.join(cls.prebuilt_path, '*.o')):
            prebuilt_files = os.path.join(cls.prebuilt_path, "*.o")
            WSLBuilder()._wslcall(f"{wsl_prefix()}rm", ["-rf",f"{wslpath(prebuilt_files)}"])

    def __init__(self, name, srccode, srcfile) -> None:
        """Return a new KernelObjectBuilder object."""
        super().__init__()

        self.name = name
        self.srccode = srccode
        self.srcfile = srcfile
        self.getheaders = False

        self.srcpath = None if srcfile is None else os.path.dirname(os.path.abspath(srcfile))

        self.prebuilt_objpath = os.path.join(self.prebuilt_path, f'{self.name}.o')
        self.prebuilt_md5path = os.path.join(self.prebuilt_path, f'{self.name}.md5')

        self.buildlog = os.path.join(self.build_path, f'{self.name}.log')
        self.buildobjpath = os.path.join(self.build_path, f'{self.name}.o')

    def build(self, debug=False):
        """Build the kernel object file and copy it to self.prebuilt_objpath."""

        if self.cached_objfile_exists():
            print(f"Using cached {self.name} kernel object file...")

            self._wslcall(f"{wsl_prefix()}cp", [f"{wslpath(self.prebuilt_objpath)}", f"{wslpath(self.buildobjpath)}"], debug)
        else:
            print(f"Building the {self.name} kernel...")

            with open(os.path.join(self.build_path, f"{self.name}.cc"), "w") as fp:
                fp.write(self.srccode)
            shutil.copytree(KERNELS_DIR, self.build_path + '/kernels/')

            if self.srcfile is not None or self.getheaders:
                for extension in ['*.h', '*.hh', '*.hpp', '*.hxx', '*.h++']:
                    for hfile in glob.glob(os.path.join(self.srcpath, extension)):
                        self._wslcall(f"{wsl_prefix()}cp",
                                      [f"{wslpath(hfile)}",
                                       f"{wslpath(self.build_path)}"], debug)

            self._wslcall(f"{wsl_prefix()}bash", [f"{wslpath(self.build_path)}/kernel_build.sh", f"{self.name}"], debug)
            self._wslcall(f"{wsl_prefix()}cp", [f"{wslpath(self.buildobjpath)}", f"{wslpath(self.prebuilt_objpath)}"], debug)

        self.update_cache_md5()

    def cached_objfile_exists(self):
        """Check if cached object file exists built from identical source code."""

        if not os.path.exists(self.prebuilt_md5path):
            return False

        if not os.path.exists(self.prebuilt_objpath):
            return False

        srccode_md5 = hashlib.md5(self.srccode.encode('utf-8')).hexdigest()
        oldsrccode_md5 = open(self.prebuilt_md5path,'r').read()
        if srccode_md5 != oldsrccode_md5:
            return False
        else:
            return True

    def update_cache_md5(self):
        srccode_md5 = hashlib.md5(self.srccode.encode('utf-8')).hexdigest()
        with open(self.prebuilt_md5path,'w') as fh:
            fh.write(srccode_md5)
