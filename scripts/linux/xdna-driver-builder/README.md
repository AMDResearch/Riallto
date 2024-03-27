# XDNA Linux NPU driver builder 
A docker-based script that will recompile the kernel/xrt/xrt_plugin for the NPU device on linux.
It will produce a tarball containing all the debian files needed to setup the system.

### To build the drivers
Run the following command to build the tarball with the debs.
```bash
./build.sh
```

This will take a while.
The expected output is a `ubuntu22.04_npu_drivers.tar.gz` tarball.

### Setting up your system

First disable secure boot from the bios of your system.

Extract the tarball, and update the kernel.
```
tar -xzvf ubuntu22.04_npu_drivers.tar.gz
sudo dpkg -i ./root/debs/linux-headers-6.7.0-rc8+*_amd64.deb
sudo dpkg -i ./root/debs/linux-image-6.7.0-rc8+_6.7.0*_amd64.deb 
sudo dpkg -i ./root/debs/linux-libc-6.7.0-rc8+_6.7.0*_amd64.deb 
```

Once that has completed restart your machine.


Then install XRT and XRT plugin:
```
sudo dpkg -i ./root/debs/xrt_*xrt.deb
sudo dpkg -i ./root/debs/xrt_plugin.*-amdxdna.deb
```

### FAQ

* If you get the following error on boot:
```
error: bad shim signature
```
This means that secure boot has not been disabled from the machine and it cannot run the necessary kernel version.

* When I install the XRT debian package I get an error saying that xocl could not be installed?
That's okay and can be ignored for now, things should still work, this should be fixed in later driver versions.
