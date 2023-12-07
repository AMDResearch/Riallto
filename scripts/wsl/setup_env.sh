#!/bin/bash
#
# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

environ="WSL"
if [ "$#" -ne 2 ]; then
	echo "Please provide the xilinx_tools and pynqMLIR-AIE tarballs"
	echo "Usage: $0 xilinx_tools_latest.tar.gz mlir_aie_latest.tar.gz"
	exit 1
fi

# Extract and setup paths for Vitis/Vivado
echo "Extracting and setting up Xilinx tools..."
tar xf $1

# We will assume tools and platforms live in /opt to make example scripting easier
echo "moving tools to /opt"
sudo mkdir /opt/tools
sudo mv Vitis /opt/tools/

# Add dummy .so's required for xclbinutil and xchesscc to function
cp /opt/tools/Vitis/2023.1/lib/lnx64.o/libboost_chrono.so.1.72.0 /opt/tools/Vitis/2023.1/lib/lnx64.o/libgurobi90.so
cp /opt/tools/Vitis/2023.1/lib/lnx64.o/libboost_chrono.so.1.72.0 /opt/tools/Vitis/2023.1/aietools/lib/lnx64.o/libgurobi90.so

# Extract MLIR tools and move to /opt
echo "moving mlir-aie and llvm installations to /opt/mlir-aie"
mkdir -p /opt/mlir-aie && tar xf $2 -C $_
sed -i "s/\/vitis\//\$XILINX_VITIS\//g" /opt/mlir-aie/bin/xchesscc_wrapper

sudo cp mlir_settings.sh /opt/

# add mlir_settings to bashrc so you don't have to source every time
echo "source /opt/mlir_settings.sh" >> ~/.bashrc