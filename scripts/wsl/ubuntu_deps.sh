#!/bin/bash
#
# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Download ubuntu packages
if { sudo apt-get update 2>&1 || echo E: update failed; } | grep -q '^[WE]:'; then
	echo "apt-get update failed. Check your connection."
	exit 1
fi

apt-get install -y libgomp1 libc6-dev-i386 clang python3-pip