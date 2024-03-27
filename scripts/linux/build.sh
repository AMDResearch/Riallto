#!/bin/bash

# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT


## Checks to make sure that all the required tarballs and license are in the directory 
if [ ! -f "./pynqMLIR-AIE.tar.gz" ]; then
	echo "Error! pynqMLIR-AIE.tar.gz is missing"
	exit 1
fi

if [ ! -f "./xilinx_tools.tar.gz" ]; then
	echo "xilinx_tools.tar.gz is missing, downloading it from opendownloads"
	wget -O riallto_installer.zip https://www.xilinx.com/bin/public/openDownload?filename=Riallto-v1.0.zip
	unzip riallto_installer.zip
        mv Riallto_v1.0/Riallto/downloads/xilinx_tools_latest.tar.gz ./xilinx_tools.tar.gz	
fi

if [ ! -f "./xdna-driver-builder/ubuntu22.04_npu_drivers.tar.gz" ]; then
	echo "xdna-driver-builder/ubuntu22.04_npu_drivers.tar.gz is missing, building it from scratch"
	pushd xdna-driver-builder
	./build.sh
	popd
fi

if [ ! -f "./Xilinx.lic" ]; then
	echo "Error! Xilinx.lic is missing"
	exit 1
fi

tar -xzvf ./xdna-driver-builder/ubuntu22.04_npu_drivers.tar.gz -C ./

docker build \
	--build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
       	-t riallto:$(date -u +'%Y_%m_%d') \
	./
