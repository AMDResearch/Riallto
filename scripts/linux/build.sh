#!/bin/bash

## Checks to make sure that all the required tarballs and license are in the directory 
if [ ! -f "./pynqMLIR-AIE.tar.gz" ]; then
	echo "Error! pynqMLIR-AIE.tar.gz is missing"
	exit 1
fi

if [ ! -f "./xilinx_tools.tar.gz" ]; then
	echo "Error! xilinx_tools.tar.gz is missing"
	exit 1
fi

if [ ! -f "./ubuntu22.04_npu_drivers.tar.gz" ]; then
	echo "Error! ubuntu22.04_npu_drivers.tar.gz is missing"
	exit 1
fi

if [ ! -f "./Xilinx.lic" ]; then
	echo "Error! Xilinx.lic is missing"
	exit 1
fi

tar -xzvf ./xdna-ipu-driver-builder/ubuntu22.04_npu_drivers.tar.gz -C ./

docker build \
	--build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
       	-t riallto:$(date -u +'%Y_%m_%d') \
	./

