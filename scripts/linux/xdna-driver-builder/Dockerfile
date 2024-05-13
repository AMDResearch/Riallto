# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

FROM ubuntu:24.04 as base

ARG DEBIAN_FRONTEND=noninteractive

COPY ./linux-config /root/
COPY ./linux /root/linux

RUN apt-get update
RUN apt-get install -y vim git
RUN apt-get install -y flex bison

RUN git config --global http.postBuffer 157286400
RUN apt-get install make

RUN apt-get install -y dpkg-dev
RUN apt-get install -y bc debhelper rsync kmod cpio libssl-dev:native
RUN apt-get install -y libelf-dev
RUN apt-get install -y zstd

RUN cd /root/linux && \ 
    mkdir -p build && \
    cp /root/linux-config ./build/.config && \
    make ARCH=x86 O=./build bindeb-pkg -j4

RUN dpkg -i /root/linux/linux-headers-*.deb
RUN dpkg -i /root/linux/linux-image-*.deb
RUN dpkg -i /root/linux/linux-libc-*.deb

RUN git config --global url."https://github.com/".insteadOf "git@github.com:"
RUN cd /root && git clone https://github.com/amd/xdna-driver.git --recursive
#RUN cd /root/xdna-driver && git checkout 036c6a49693f04c31e0bba2ab2cf8f7fb59b39e8

RUN apt-get install -y cmake jq pkg-config wget libdrm-dev 
RUN apt-get install -y python3-pip
RUN apt-get install -y ocl-icd-opencl-dev

RUN find /root/xdna-driver -type f -exec sed -i 's/\$(shell uname -r)/6.8.8+/g' {} +
RUN find /root/xdna-driver -type f -exec sed -i 's/\$(uname -r)/6.8.8+/g' {} +
RUN find /root/xdna-driver -type f -exec sed -i 's/\$(\/bin\/uname -r)/6.8.8+/g' {} +
RUN find /root/xdna-driver -type f -exec sed -i 's/`uname -r`/6.8.8+/g' {} +
RUN /root/xdna-driver/xrt/src/runtime_src/tools/scripts/xrtdeps.sh -docker

RUN cd /root/xdna-driver/build && ./build.sh -release
RUN cd /root/xdna-driver/build && ./build.sh -package

RUN cd /root/xdna-driver/xrt/build && ./build.sh -noert 

RUN mkdir /root/debs
RUN cp /root/linux/linux-headers-*.deb /root/debs/
RUN cp /root/linux/linux-image-*.deb /root/debs/
RUN cp /root/linux/linux-libc-*.deb /root/debs/
RUN cp /root/xdna-driver/build/Release/xrt_plugin*.deb /root/debs/
RUN cp /root/xdna-driver/xrt/build/Release/xrt*xrt.deb /root/debs/

