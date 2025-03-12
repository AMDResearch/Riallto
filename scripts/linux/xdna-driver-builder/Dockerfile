# Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y --install-recommends linux-generic-hwe-24.04
RUN apt-get install -y linux-headers-$(uname -r)
RUN apt-get install -y vim git flex bison make libelf1t64
RUN apt-get install -y dpkg-dev libelf-dev zstd
RUN apt-get install -y bc debhelper rsync kmod cpio libssl-dev:native

RUN git config --global http.postBuffer 157286400

#RUN apt-get install -y linux-base

RUN git config --global url."https://github.com/".insteadOf "git@github.com:"
RUN cd /root && git clone https://github.com/amd/xdna-driver.git --recursive
RUN cd /root/xdna-driver && git checkout 1e1c63f
RUN cd /root/xdna-driver && git submodule update --recursive

RUN apt-get install -y cmake jq pkg-config wget libdrm-dev
RUN apt-get install -y python3-pip ocl-icd-opencl-dev

RUN /root/xdna-driver/xrt/src/runtime_src/tools/scripts/xrtdeps.sh -docker

RUN cd /root/xdna-driver/build && ./build.sh -release
RUN cd /root/xdna-driver/build && ./build.sh -package

RUN cd /root/xdna-driver/xrt/build && ./build.sh -npu -opt

## Install xdna-driver
#RUN -E dpkg -i /root/xdna-driver/xrt/build/Release/xrt_*-amd64-npu.deb
#RUN -E dpkg -i /root/xdna-driver/build/Release/xrt_plugin*-amdxdna.deb

# Make a copy of the deb files
RUN mkdir /root/debs
RUN cp /root/xdna-driver/xrt/build/Release/xrt_*-amd64-npu.deb /root/debs/
RUN cp /root/xdna-driver/build/Release/xrt_plugin*.deb /root/debs/
