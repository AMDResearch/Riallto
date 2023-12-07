#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

source /opt/mlir_settings.sh

# build the instructions
aie-opt --aie-dma-to-ipu -o gen_ipu_insts.mlir aiert_insts.mlir
aie-translate --aie-ipu-instgen -o final.seq gen_ipu_insts.mlir
