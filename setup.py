# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""This setup.py file is used to build the IPU Python package for the
Riallto project. It installs Python APIs for developers to build and 
program the AMD IPU device. This package also includes a set of prebuilt
computer vision applications that can be used to run on the IPU.

Note
----
You can run prebuilt applications out of the box using this IPU Python
package, however in order to build your own applications you will need
to install Riallto using the full installer and additionally obtain
an AIEBuild license from https://www.xilinx.com/getlicense.
"""

from setuptools import find_packages, setup
import platform

# Windows and linux have different bindings version, so we need to add
# the appropriate constraint based on the platform.
required_python_version = ""
if platform.system() == 'Linux':
    required_python_version = "3.12.*"
elif platform.system() == 'Windows':
    required_python_version = "3.9.*"
else:
    raise OSError(f'Unknown Operating System: {platform.os.name} {platform.system()}')

setup(
    name="npu",
    version='1.0',
    package_data={
        '': ['*.py', '*.pyd', '*.so', '*.dll', 'Makefile', '.h', '.cpp',
            'tests/*',
	        'runtime/*.so',
            'runtime/*.dll',
            'build/*.txt',
            'utils/*',
            'lib/applications/*',
            'lib/kernels/*',
            'lib/graphs/*',
            'lib/kernels/cpp/*.cpp',
            'lib/kernels/cpp/*.h',
            'runtime/utils/*',
            'build_template/check_license.sh',
            'build_template/kernel_build.sh',
            'build_template/seq_build.sh',
            'build_template/app_build.sh',
            'lib/applications/binaries/*'],
    },
    packages=find_packages(),
    python_requires=f"=={required_python_version}",
    install_requires=[
        "numpy<2.0",
        "pytest",
        "pytest-cov",
        "opencv-python",
        "matplotlib",
        "CppHeaderParser",
        "jupyterlab",
        "ipywidgets",
        "pillow>=10.0.0",
        "ml_dtypes"
    ],
    description="Riallto is a simple framework for programming and interacting with the AMD NPU device.")
