#!/bin/bash

set -o pipefail
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

source /opt/mlir_settings.sh

xchesscc $CHESSCC2_FLAGS -I kernels -c $1.cc -o $1.o #2>&1 | tee $1.log

echo "Successfully built $1.o"