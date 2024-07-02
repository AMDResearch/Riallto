# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

FROM ubuntu:24.04 as base

ARG DEBIAN_FRONTEND=noninteractive
ARG KERNEL_HEADERS
ARG KERNEL_HEADERS_GENERIC
ARG KERNEL_MODULES
ARG KERNEL_IMAGE

RUN apt-get update
RUN apt-get install -y vim git
RUN apt-get install -y flex bison

RUN git config --global http.postBuffer 157286400
RUN apt-get install make

RUN apt-get install -y dpkg-dev
RUN apt-get install -y bc debhelper rsync kmod cpio libssl-dev:native
RUN apt-get install -y libelf-dev
RUN apt-get install -y zstd

COPY _work /root/

RUN apt-get install -y libelf1t64
RUN apt-get install -y kmod 
RUN apt-get install -y linux-base 

RUN dpkg -i /root/$KERNEL_HEADERS
RUN dpkg -i /root/$KERNEL_HEADERS_GENERIC
RUN dpkg -i /root/$KERNEL_MODULES
RUN dpkg -i /root/$KERNEL_IMAGE

RUN git config --global url."https://github.com/".insteadOf "git@github.com:"
RUN cd /root && git clone https://github.com/amd/xdna-driver.git --recursive
RUN cd /root/xdna-driver && git checkout 7682e0b4b6c435d6700faef3dc27cd709324e27f 

RUN apt-get install -y cmake jq pkg-config wget libdrm-dev 
RUN apt-get install -y python3-pip
RUN apt-get install -y ocl-icd-opencl-dev

RUN /root/xdna-driver/xrt/src/runtime_src/tools/scripts/xrtdeps.sh -docker

RUN mkdir /root/debs
RUN cd /root/xdna-driver/build && ./build.sh -release
RUN cd /root/xdna-driver/build && ./build.sh -package

RUN cd /root/xdna-driver/xrt/build && ./build.sh -noert 

RUN cp /root/xdna-driver/build/Release/xrt_plugin*.deb /root/debs/
RUN cp /root/xdna-driver/xrt/build/Release/xrt*xrt.deb /root/debs/
