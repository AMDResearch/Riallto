# Riallto Ubuntu 24.04 setup

Currently there is support for bringing up Riallto on Ubuntu 24.04 with docker.
To use Riallto on Linux requires the use of the [xdna-driver](https://github.com/amd/xdna-driver) which is installed as part of the setup.
Currently this driver has not been merged into the main Linux kernel, so a `6.8.8+` version of the kernel is built and installed as part of this installation.

## Install steps

On an NPU enabled laptop running Ubuntu 24.04.

1. __Setup Docker.__ 
You can follow the steps [here](https://docs.docker.com/desktop/install/ubuntu/).

2. __Add your user to the docker user group.__ (May require relogin)
```
sudo usermod -aG docker $USER
```

3. __Obtain a license file for Riallto.__
Please follow the [guide here](https://riallto.ai/prerequisites-aie-license.html#prerequisites-aie-license)

4. __Disable secure boot from your bios settings.__ Since we have to use an experimental kernel version it is required to disable secure boot before it can be used. There is a [guide](https://learn.microsoft.com/en-us/windows-hardware/manufacture/desktop/disabling-secure-boot?view=windows-11) from Microsoft here, but often the steps depend on your hardware manufacturer.

5. __Run `./setup_riallto_linux.sh <your license file>`.__
This command will check the kernel version is currently configured and if the xdna-driver has been installed. If not it will build the required kernel version and install it. This takes about 1 hour to run and after completing successfully the user will be asked to restart.

6. __Reboot the machine.__ 
To finish upgrading the kernel to 6.8.8+.

7. __Re run the `./setup_riallto_linux.sh <your license file>` script.__
This will build the Riallto docker.

## Testing the installation
Inside this directory there are a few scripts.

* `launch_jupyter.sh` - will launch a jupyterlab server from a docker container allowing you to use Riallto notebooks.
* `run_pytest.sh` - will run a suit of pytests to test the operation of your NPU device and the Riallto installation.

