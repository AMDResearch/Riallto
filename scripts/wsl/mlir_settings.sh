#!/bin/bash
#
# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

export MLIR_AIE_INSTALL_DIR=`realpath /opt/mlir-aie/`
export LLVM_INSTALL_DIR=`realpath /opt/mlir-aie`

export PATH=${MLIR_AIE_INSTALL_DIR}/bin:${LLVM_INSTALL_DIR}/bin:${PATH}
export PYTHONPATH=${MLIR_AIE_INSTALL_DIR}/python:${PYTHONPATH}
export LD_LIBRARY_PATH=${MLIR_AIE_INSTALL_DIR}/lib:${LLVM_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}

export XILINX_VITIS=/opt/tools/Vitis/2023.1
export VITIS_AIETOOLS_DIR=${XILINX_VITIS}/aietools
export VITIS_AIE2_INCLUDE_DIR=${XILINX_VITIS}/aietools/data/aie_ml/lib

export CHESSCC2_FLAGS="-f -p me -P ${VITIS_AIE2_INCLUDE_DIR} -I ${VITIS_AIETOOLS_DIR}/include -D__AIENGINE__=2 -D__AIEARCH__=20"
export CHESS_FLAGS="-P ${VITIS_AIE_INCLUDE_DIR}"

export PATH=$PATH:$XILINX_VITIS/aietools/bin:$XILINX_VITIS/bin
