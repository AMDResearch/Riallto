#!/bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

source /opt/mlir_settings.sh

aiecc.py -v --no-aiesim --aie-generate-cdo --aie-generate-ipu --no-compile-host --xclbin-name=final.xclbin --kernel-name $1 --instance-name "Riallto" --ipu-insts-name=final.seq aie.mlir 2>&1 | tee $SCRIPT_DIR/build.log
