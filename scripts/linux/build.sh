#!/bin/bash

# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

DRIVER_TARBALL=ubuntu24.04_npu_drivers.tar.gz

## Checks to make sure that all the required tarballs and license are in the directory 
if [ ! -f "./pynqMLIR-AIE.tar.gz" ]; then
	echo "Error! pynqMLIR-AIE.tar.gz is missing, downloading from opendownloads"
	wget -O pynqMLIR-AIE.tar.gz https://www.xilinx.com/bin/public/openDownload?filename=pynqMLIR_AIE_py310_v0.9.tar.gz 
fi

if [ ! -f "./xilinx_tools.tar.gz" ]; then
	echo "xilinx_tools.tar.gz is missing, downloading it from opendownloads"
	wget -O riallto_installer.zip https://www.xilinx.com/bin/public/openDownload?filename=Riallto-v1.0.zip
	unzip riallto_installer.zip
        mv Riallto_v1.0/Riallto/downloads/xilinx_tools_latest.tar.gz ./xilinx_tools.tar.gz	
fi

if [ ! -f "./xdna-driver-builder/${DRIVER_TARBALL}" ]; then
	echo "xdna-driver-builder/${DRIVER_TARBALL} is missing, building it from scratch"
	pushd xdna-driver-builder
	./build.sh
	popd
fi

if [ ! -f "./Xilinx.lic" ]; then
	echo "Error! Xilinx.lic is missing"
	exit 1
fi

tar -xzvf ./xdna-driver-builder/${DRIVER_TARBALL} -C ./

docker build \
	--build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
       	-t riallto:latest \
	./

