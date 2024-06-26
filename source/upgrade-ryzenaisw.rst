.. _upgrade-ryzenaisw:

Upgrading RyzenAI-SW
====================

The Riallto Windows installer sets up RyzenAI-SW 1.0 packages for the section 5 notebooks. In order to upgrade to RyzenAI-SW v1.1 in the Riallto Python virtual environment you can run the utility script `upgrade_ryzenaisw.ps1`.

Upgrade steps
-------------

1. Download the latest `RyzenAI-SW package <https://account.amd.com/en/forms/downloads/ryzen-ai-software-platform-xef.html?filename=ryzen-ai-sw-1.1.zip>`_ from https://ryzenai.docs.amd.com.
2. Open a powershell terminal and and execute the `upgrade_ryzenaisw.ps1` script from the Riallto repo directory with the downloaded zipfile as an argument:
   ``./scripts/util/upgrade_ryzenaisw.ps1 C:/where/you/downloaded/ryzen-ai-sw-1.1.zip``

The script will:
1. Purge the previously installed onnxruntime, voe and vai_q_onnx python packages.
2. Install the wheels from `ryzen-ai-sw-1.1.zip`.
3. Replace the previous 1x4.xclbin and vaip.config files in the riallto_notebooks directory with the latest ones from the package you downloaded.

Refer to the `Ryzen AI docs <https://ryzenai.docs.amd.com/en/latest/inst.html>`_ for more detailed information on RyzenAI-SW and features.