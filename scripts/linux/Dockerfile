# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

FROM ubuntu:24.04 as base

ARG LIC_MAC="00:15:5d:fe:46:53"
ENV LIC_MAC=${LIC_MAC}

SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y sudo adduser

RUN useradd -m riallto
RUN adduser riallto sudo
RUN echo 'riallto ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/riallto
USER riallto
ENV HOME /home/riallto
WORKDIR /home/riallto

RUN sudo chmod a+rwx /root

ARG DEBIAN_FRONTEND=noninteractive

# Generate locales
RUN sudo apt-get update
RUN sudo apt-get install -y git vim wget locales
RUN sudo locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC sudo -E apt-get install -y tzdata

RUN git clone https://github.com/amd/xdna-driver.git --recursive --depth=1 
RUN sudo chmod +x /home/riallto/xdna-driver/xrt/src/runtime_src/tools/scripts/xrtdeps.sh 
RUN sudo /home/riallto/xdna-driver/xrt/src/runtime_src/tools/scripts/xrtdeps.sh -docker

COPY /root/debs npu_ubuntu_debs

RUN sudo dpkg -i npu_ubuntu_debs/linux-headers*_amd64.deb
RUN sudo dpkg -i npu_ubuntu_debs/linux-image*_amd64.deb
RUN sudo dpkg -i npu_ubuntu_debs/linux-libc*_amd64.deb

RUN sudo apt-get install -y dkms udev python3
RUN sudo apt-get install -y python3-pip ocl-icd-opencl-dev uuid-dev

RUN sudo -E dpkg -i npu_ubuntu_debs/xrt_*-amd64-xrt.deb
RUN sudo -E dpkg -i npu_ubuntu_debs/xrt_plugin*-amdxdna.deb

RUN git clone https://github.com/STFleming/Riallto -b linux_bringup --depth=1 

SHELL ["/bin/bash", "-l", "-c"]

# Setup the bashrc environment
RUN echo "source /opt/xilinx/xrt/setup.sh" >> ~/.bashrc
RUN sed -i '6,9d' ~/.bashrc # remove interactive only mode in bashrc
RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/xilinx/xrt/lib"

# Setup Riallto
RUN sudo apt-get install -y python3-venv
RUN python3 -m venv ~/riallto_venv
RUN echo "source ~/riallto_venv/bin/activate" >> ~/.bashrc
RUN source ~/.bashrc && cd ~/Riallto/scripts/wsl/ && chmod +x ubuntu_deps.sh && sudo ./ubuntu_deps.sh
RUN source ~/.bashrc && cd ~/Riallto/scripts/wsl/ && python3 -m pip install -r requirements.txt
RUN python3 -m pip install ~/Riallto

# Install MLIR-AIE and Xilinx tools
COPY *.tar.gz /home/riallto/
RUN cd ~/Riallto/scripts/wsl/ && chmod +x setup_env.sh && sudo -E ./setup_env.sh /home/riallto/xilinx_tools.tar.gz /home/riallto/pynqMLIR-AIE.tar.gz && sudo chmod -R a+rwx /opt/

# Patch the MLIR-AIE xclbin metadata generation
RUN sudo sed -i '340 i\        "subtype":"DPU",' /opt/mlir-aie/python/aie/compiler/aiecc/main.py

# Setup license
COPY Xilinx.lic /opt/
RUN sudo apt-get install -y iproute2

# For external license
RUN sudo -E echo 'export XILINXD_LICENSE_FILE=/opt/Xilinx.lic' >> /opt/mlir_settings.sh
RUN sudo -E echo 'sudo ip link add vmnic0 type dummy || true' >> /opt/mlir_settings.sh
RUN sudo -E echo 'sudo ip link set vmnic0 addr ${LIC_MAC} || true' >> /opt/mlir_settings.sh

# Install Jupyterhub/notebooks
RUN source ~/.bashrc && python3 -m pip install jupyterhub
RUN source ~/.bashrc && python3 -m pip install notebook 

# Install the ONNX Runtime for 5_* notebooks
#COPY onnxruntime_linux /home/riallto/onnxruntime_linux
#
#RUN python3 -m pip install torch==2.1.0
#RUN python3 -m pip install torchvision==0.16.0
#RUN python3 -m pip install scikit-learn==1.3.2
#RUN python3 -m pip install seaborn==0.13.0
#RUN python3 -m pip install onnx==1.15.0
#RUN python3 -m pip install onnxruntime-extensions
#RUN python3 -m pip install tqdm
#
#RUN cd /home/riallto/onnxruntime_linux && \
#      python3 -m pip install wheels/onnxruntime*.whl
#
#RUN cd /home/riallto/onnxruntime_linux && \
#      python3 -m pip install wheels/voe*.whl
#
#RUN cd /home/riallto/onnxruntime_linux && \
#      python3 -m pip install wheels/vai_q_onnx*.whl --no-deps
#
#RUN cd /home/riallto/onnxruntime_linux && \
#      sudo tar -xzvf lib.tar.gz
#
#ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/mlir-aie/lib:/opt/xilinx/xrt/lib:/home/riallto/onnxruntime_linux/lib
#ENV XLNX_VART_FIRMWARE=/home/riallto/onnxruntime_linux/AMD_AIE2_4x4_Overlay.xclbin
#ENV XLNX_TARGET_NAME=AMD_AIE2_4x4_Overlay
#ENV NUM_OF_DPU_RUNNERS=1

#Cleanup a bit
RUN rm -rf /home/riallto/*.tar.gz

CMD ["/bin/bash", "-l", "-c", "jupyterhub-singleuser", "--allow-root"]

