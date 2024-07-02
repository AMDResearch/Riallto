.. _install-riallto-windows:

Install Riallto on Windows
===========================

The Riallto installer has two install options, 'Lite' and 'Full'. The 'Lite' version has the all essential packages required for the Riallto and ONNX runtimes on the NPU to get started with prebuilt vision applications. The 'Full' version enables developers to write their own custom applications, which requires additional tools that depend on a Ubuntu WSL instance. The diagram below shows the requirements and available notebooks for both 'Lite' and 'Full' install versions.

+------------------------+------------------+-------------------------------+
|                        | Lite             | Full                          |
+========================+==================+===============================+
| Installer requirements | IPU Driver       | IPU Driver                    |
|                        |                  |                               |
|                        |                  | WSL2                          |
|                        |                  |                               |
|                        |                  | AIE License                   |
+------------------------+------------------+-------------------------------+
| Supported Notebooks    | Sections 1,2,3,5 | All                           |
+------------------------+------------------+-------------------------------+


Prerequisites
-------------

You will need a laptop or computer with an AMD Ryzen AI processor. The Ryzen NPU and Riallto are currently only supported under Windows 11. 

The Ryzen AI NPU appears in the Windows Device Manager as an *IPU* (Inference Processing Unit) which is another term for an NPU. Both 'Lite' and 'Full' installation options require the installation of the IPU driver version 10.1009.8.100. You can install the Windows driver for the Ryzen AI NPU by following the instructions on the following page:

:ref:`prerequisites-driver`

If installing the 'Lite' version you can move directly to the next Install Riallto section below. If installing the 'Full' version, you require both the Windows Susbsystem for Linux (WSL2) and an AIE build license. Details on these can be found on the following pages:

:ref:`prerequisites-wsl`

:ref:`prerequisites-aie-license`

Download Riallto installer
--------------------------

Riallto consists of runtime software to load Ryzen AI applications, and a software toolchain to compile and build applications for Ryzen AI. Riallto also includes a series of Jupyter Notebook tutorials. 

`Download the latest v1.0 Riallto installer <https://www.xilinx.com/bin/public/openDownload?filename=Riallto-v1.0.zip>`_ and run it on your Ryzen AI laptop. This will install the Riallto software framework and a copy of the Riallto Jupyter notebooks that you can browse and run on your laptop. Make sure to select the correct installation option when prompted, the 'Full' version will not install without WSL2 or the AIE build license (see details in the Prerequisites section above). 

If you don't have a Ryzen AI laptop, you can browse the Riallto notebooks as webpages which make up the majority of the content on the webpages you are browsing now. See the *NPU Architecture Features* section to learn more about the NPU. The *Building Applications* section to learn how to build custom applications for the NPU. The last section, will show you how to run *Machine Learning with ONNX* on Ryzen AI.


Start Riallto
-------------

Once you have installed Riallto on your laptop, click on the desktop icon. This will start a JupyterLab server instance in your web browser.

In your browser, you should see the Jupyter home area where you can browse and open the Riallto tutorial notebooks. 

In the Jupyter home area, open `1_0_Introduction.ipynb` to start exploring Ryzen AI. 


Getting help
------------

For support, feel free to ask questions on the `Riallto GitHub Discussions page <https://github.com/AMDResearch/Riallto/discussions>`_.

If you have access to GitHub, you can browse the source code and post any bugs or software requests to the `Riallto GitHub issue tracker <https://github.com/AMDResearch/Riallto/issues>`_.

