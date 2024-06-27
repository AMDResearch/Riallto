#!/bin/bash
#
# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Check input args
if [ -z "$1" ]; then
	echo "Error: No notebooks directory supplied (usually Riallto/notebooks)"
	echo "Usage: $0 <notebooks directory>"
	exit 1
fi

DIR="$1"
ABS_DIR=$(cd "$DIR" && pwd)

# check to make sure the directory exists 
if [ ! -d "$ABS_DIR" ]; then
	echo "Error: $ABS_DIR is not a valid directory or does not exist."
	exit
fi

docker container stop riallto_jupyter > /dev/null 2>&1 || true 
docker container wait riallto_jupyter > /dev/null 2>&1 || true 

webcams=$(ls /dev/video*)
cmd="docker run -dit --rm --name riallto_jupyter"
cmd+=" --cap-add=NET_ADMIN --device=/dev/accel/accel0:/dev/accel/accel0"
cmd+=" -p 8888:8888 "
cmd+=" -v $ABS_DIR:/notebooks "
for cam in $webcams; do
        cmd+=" --device=$cam:$cam"
done
cmd+=" -w /notebooks riallto:latest /bin/bash"

echo " running $cmd"
eval $cmd

docker exec -it riallto_jupyter /bin/bash -c " (sudo chmod 666 /dev/video* || true) && source ~/.bashrc && cd /notebooks && python3 -m jupyterlab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=''"

docker container stop riallto_jupyter > /dev/null 2>&1 || true 
