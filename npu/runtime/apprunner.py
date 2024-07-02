# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import gc
import keyword
import colorsys
from typing import Optional
from inspect import Parameter, Signature
import PIL
from io import BytesIO
import npu.runtime as ipr
from .kernelinstance import KernelInstance
from npu.repr_dict import ReprDict
from typing import List
import json
from npu.utils.xbutil import XBUtil
import ipywidgets as widget
from IPython.display import display
import subprocess
import platform
from warnings import warn

dtype_to_maxval = {
    "uint32_t" : 4294967296,
    "uint16_t" : 65536,
    "uint8_t" : 255
}
_colors = [colorsys.hsv_to_rgb(hue / 255, 1, 1) for hue in np.arange(0, 256)]
_color_matrix = np.array([_colors])


def _huebar_inrange(min: int, max: int):
    mat = np.copy(_color_matrix)
    mat[0,:min,:] = 1.
    mat[0,max:] = [1., 1., 1.]
    return np.uint8(mat*255)


def _updatebar(huebar: widget.Image , min:int, max: int):
    pil_image = PIL.Image.fromarray(_huebar_inrange(min, max))
    b = BytesIO()
    pil_image.save(b, format='jpeg')
    huebar.value = b.getvalue()


class IPUAppAlreadyLoaded(Exception):
    pass

class AppRunner:
    """This class abstracts the necessary setup steps of an NPU
    application and enables a simple interface with the accelerator
    using allocate() methods, treating buffers to the NPU as simple
    Numpy arrays.

    Attributes
    ----------
    xclbin_name : str
        Name of xclbin file
    fw_sequence : str
        Name of the firmware sequence, typically same name as the
        xclbin file
    handoff : str
        Name of the metadata handoff file, typically a .json file
        with the same name as the xclbin and firmware files

    Note
    ----
    This class is primarily built on top of the python bindings to
    XRT (Xilinx Runtime Library). You can read more about the runtime
    in the documentation at https://xilinx.github.io/XRT/.

    """

    def __init__(self, xclbin_name:str, fw_sequence:Optional[str]=None, handoff:Optional[str]=None):
        """Returns a new AppRunner object."""

        self._process_handoff_metadata(xclbin_name, handoff)

        # Extra stability checks for windows
        if platform.system() == "Windows":
            self.xbutil = XBUtil()
            self._stability_checks()

        # If sequence given, use it, otherwise look for same name as xclbin
        if fw_sequence:
            self.sequence = ipr.Sequence(fw_sequence, first_parse=True)
        else:
            self.sequence = ipr.Sequence(os.path.splitext(xclbin_name)[0] + '.seq', first_parse=True)

        xclbin = ipr.xclbin(xclbin_name)

        # Run the script to allow this unsigned firmware xclbin to run

        self.kernel_params = self._get_kernel_info(xclbin_name)
        self.device =  self._get_device()

        try:
            self.device.register_xclbin(xclbin)
        except RuntimeError as e:
            print(str(e))
            print("""Failed to register xclbin. Is another application running?
Try shutting down/restarting all other jupyter notebooks and try again.""")
            raise

        context = ipr.hw_context(self.device, xclbin.get_uuid())

        self.kernel = ipr.kernel(context, self.kernel_name)
        self.__mem_bank_indexes = self._get_mem_bank_indexes()

        self._load_sequence()
        self._apply_metadata()

        self._widgets =  {}
        self._allocated_arrays = []

    def _get_device(self):
        """ Checks to see if there is already an AppRunner that exists.
        if there is gets the device from previously allocated AppRunner or
        else creates a new device.
        """

        for obj in gc.get_objects():
            try:
                if isinstance(obj, type(self)) and (obj != self):
                    if getattr(obj, "device", None):
                        return obj.device
            except Exception as e:
                warn(f"Encountered an exception during isinstance check: {e}")
                continue
        return ipr.device(0)

    def _process_handoff_metadata(self, xclbin_name:str, handoff:Optional[str]=None)->None:
        """ Parses the handoff metadata if it exists and uses that to set the
        kernel name. If no metadata exists then parse the kernel name from
        the xclbin.
        """
        if handoff:
            with open(handoff, 'r') as f:
                self._metadata = json.load(f)
                self.kernel_name = self._metadata["application"]
        else:
            try:
                with open(os.path.splitext(xclbin_name)[0] + '.json', 'r') as f:
                    self._metadata = json.load(f)
                    self.kernel_name = self._metadata["application"]
            except:
                self._metadata = None
                self.kernel_name = self._infer_kernel_name(xclbin_name)
        return

    @property
    def metadata(self):
        return ReprDict(self._metadata)

    def _stability_checks(self)->None:
        """ Checks to ensure that the NPU is in a sensible state before
        trying to load the application
        """
        if self.xbutil.app_count >= 4:
            raise RuntimeError("There is currently no free space on the NPU "
                               "to run this application, have you tried closing"
                               " other applications that you are running or "
                               f"disabling WSE? \n\n{self.xbutil.app_table}")

    def _apply_metadata(self):
        """ Tries to associate metadata with RTP commands in the sequence."""
        self.rtps = {}
        if not self._metadata is None:
            ctkernels = [self._metadata["kernels"][k]
                                for k in self._metadata["kernels"]
                                if self._metadata["kernels"][k]["type"] == "CT"]
            for k in ctkernels:

                if k["tloc"] is None:
                    raise ValueError(f"A CT Kernel in the Metadata has not been placed to a tile, most likely the tloc attribute of a kernel was not set")

                d = KernelInstance()
                d._portlist = k["ports"]
                d._tloc = tuple(k["tloc"])
                self.rtps[k["name"]] = {}

                # Determine if there were RTPs picked up in the sequence for
                # this kernel instance
                if d._tloc in self.sequence.mlir_rtps:
                    idx = 0
                    for port in k["ports"].values():
                        if port["ctype"] == "rtp":
                            if idx in self.sequence.mlir_rtps[d._tloc]:
                                self.sequence.mlir_rtps[d._tloc][idx].value = port['value']
                                pdict = {
                                            "seq_ref" : self.sequence.mlir_rtps[d._tloc][idx],
                                            "dtype" : port["c_dtype"],
                                            "init_val" : port["value"]
                                        }
                                self.rtps[k["name"]][port["name"]] = pdict
                                setattr(d, port["name"], self.sequence.mlir_rtps[d._tloc][idx])
                        idx = idx + 1

                setattr(self, k["name"], d)

    def rtpupdate(self, rtpseq, val):
        rtpseq.value = val

    def _pairupdate(self, rtpseq:list, val:list):
        rtpseq[0].value = val[0]
        rtpseq[1].value = val[1]

    def _hueupdate(self, rtpseq:list, val: int, huebar: widget.Image):
        rtpseq[0].value = val[0]
        rtpseq[1].value = val[1]
        _updatebar(huebar, *val)

    def dropdownrtpupdate(self,rtpseq, options, val):
        rtpseq.value = options.index(val)

    def rtpwidgets(self, widgetmeta={}):
        """ This function automatically generates ipywidgets if this has been enabled in the metadata."""
        widgets = {}
        hboxes = []
        for instance_name, instance in self.rtps.items():
            widgets[instance_name] = []
            for rtpname, rtp in instance.items():
                if rtpname in widgetmeta:
                    wmeta = widgetmeta[rtpname]
                    if wmeta["type"] == "slider":
                        slider = widget.IntSlider(min=wmeta["min"],
                                                 max=wmeta["max"],
                                                 description=wmeta.get('name', rtpname),
                                                 value=rtp["init_val"])
                        slider.observe(
                            lambda change, v=self.rtps[instance_name][rtpname] : self.rtpupdate(v["seq_ref"], change.new)
                            , 'value')
                        slider.style.description_width = 'auto'
                        widgets[instance_name].append(slider)

                    elif wmeta["type"] == "dropdown":
                        dropdown = widget.Dropdown(
                            options=wmeta["options"],
                            description=wmeta.get('name', rtpname),
                            disabled=False,
                            value=wmeta["options"][rtp['init_val']])
                        dropdown.observe(
                            lambda change, v=self.rtps[instance_name][rtpname]:
                            self.dropdownrtpupdate(v["seq_ref"], wmeta["options"], change.new)
                            , 'value')
                        dropdown.style.description_width = 'auto'
                        widgets[instance_name].append(dropdown)

                    elif wmeta["type"] == "dropdownpair":
                        pair = instance.get(wmeta["pair"])
                        value = [rtp["init_val"], pair["init_val"]]
                        dropdownp = widget.Dropdown(
                            options=wmeta["options"],
                            description=wmeta.get('name', rtpname),
                            disabled=False,
                            value=value)
                        dropdownp.observe(
                            lambda change, v=self.rtps[instance_name][rtpname], h=pair["seq_ref"]:
                            self._pairupdate([v["seq_ref"], h], change.new)
                            , 'value')
                        dropdownp.style.description_width = 'auto'
                        widgets[instance_name].append(dropdownp)

                    elif wmeta["type"] == "rangeslider":
                        rhigh = instance.get(wmeta["rangehigh"])
                        value = [rtp["init_val"], rhigh["init_val"]]
                        rslider = widget.IntRangeSlider(
                                                value=value,
                                                min=wmeta["min"],
                                                max=wmeta["max"],
                                                step=wmeta.get('step', 1),
                                                description=wmeta.get('name', rtpname))
                        rslider.observe(
                            lambda change, v=self.rtps[instance_name][rtpname], h=rhigh["seq_ref"]:
                            self._pairupdate([v["seq_ref"], h], change.new)
                            , 'value')
                        rslider.style.description_width = 'auto'
                        widgets[instance_name].append(rslider)

                    elif wmeta["type"] == "hueslider":
                        rhigh = instance.get(wmeta["rangehigh"])
                        value = [rtp["init_val"], rhigh["init_val"]]
                        hueslider = widget.IntRangeSlider(
                                                value=value,
                                                min=wmeta["min"],
                                                max=wmeta["max"],
                                                step=wmeta.get('step', 1),
                                                description=wmeta.get('name', rtpname))
                        huebar = widget.Image(format='jpeg')
                        _updatebar(huebar, *value)
                        hueslider.metadata = {'key': instance_name}
                        hueslider.observe(
                            lambda change, v=self.rtps[instance_name][rtpname],
                            h=rhigh["seq_ref"], huebar=huebar:
                            self._hueupdate([v["seq_ref"], h], change.new, huebar)
                            , 'value')
                        hueslider.style.description_width = 'auto'

                        widgets[instance_name].append(hueslider)
                        widgets[instance_name].append(huebar)

            if len(widgets[instance_name]) > 0:
                hbox = widget.HBox(widgets[instance_name], description=instance_name)
                display(hbox)
                hboxes.append(hbox)
        self._widgets = widgets
        self._hboxes = hboxes

    def rtpsliders(self, filters:List[str] = [], radios={}):
        rtpsliders = {}
        for instance_name, instance in self.rtps.items():
            rtpsliders[instance_name] = []
            for rtpname, rtp in instance.items():
                if not rtpname in filters:
                    slider = widget.IntSlider(min = 0, max = dtype_to_maxval[rtp["dtype"]], description=f"{rtpname}", value=rtp['init_val'])
                    slider.observe(
                            lambda change, v=self.rtps[instance_name][rtpname] : self.rtpupdate(v["seq_ref"], change.new)
                            , 'value')
                    rtpsliders[instance_name].append(slider)
            if len(rtpsliders[instance_name]) > 0:
                hbox = widget.HBox(rtpsliders[instance_name], description=instance_name)
                display(hbox)

    def _load_sequence(self):
        """ Pre-allocate sequence, we assume it's always the 1st kernel argument."""
        try:
            self.instr = ipr.bo(self.device, self.sequence.buffer.nbytes, ipr.bo.flags.cacheable, self.__mem_bank_indexes[0])
            self.instr.write(self.sequence.buffer, 0)
            self.instr.sync(ipr.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        except MemoryError as e:
            print(str(e))
            print("Try resetting your device.")
            raise

    def display(self)->None:
        """ Display the graph of the loaded application."""
        from npu.utils.appviz import AppViz
        _viz = AppViz(self._metadata)
        _viz.show

    def save(self, filename:str=None)->None:
        """ Saves animation to a file."""
        from npu.utils.appviz import AppViz
        _viz = AppViz(self.metadata)
        _viz.save(filename)

    def _refresh_sequence(self):
        """ Reloads the sequence for the device."""
        try:
            self.instr.write(self.sequence.buffer, 0)
            self.instr.sync(ipr.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        except MemoryError as e:
            print(str(e))
            print("Try resetting your device.")
            raise

    def _get_mem_bank_indexes(self):
        """Since the instruction sequence is allocated and loaded onto the device
        on __init__ we don't have to worry about it. We are assuming the
        memory bank indexes will be the same for the remaining ports.
        """

        mem_bank_indexes = []

        for i in range(len(self.kernel_params['parameters'])):
            mem_bank_indexes.append(self.kernel.group_id(i))

        # app.allocate expects the indexes to be the same for user-defined, non-sequence buffers
        if len(set(mem_bank_indexes[2:])) != 1:
            warnings.warn("Host buffer memory bank indexes not the same, could cause allocation issues", UserWarning)

        return mem_bank_indexes

    def _infer_kernel_name(self, xclbin_name) -> str:
        """ From the xclbin infers the kernel name if one is note provided.
            Searched the xclbin metadata to find a kernel with instance name
            Riallto and uses the kernel name from that"""

        with open(xclbin_name, 'rb') as f:
            xclbin_data = f.read()

        kernel_name = ''
        found = False
        start = 0
        while 1:
            idx = xclbin_data[start:].find(b"<kernel ")
            if idx == -1:
                break
            start = start + idx + 1
            end = xclbin_data[start:].find(b"</kernel>")

            kernel_info = xclbin_data[start-8:start+end+9].decode()

            for line in kernel_info.split('\n'):
                if "kernel name" in line:
                    kernel_name = re.search(r'kernel name="(.*?)"', line).group(0).split('=')[1].strip('\'').strip('\"')
                if "instance name" in line:
                    instance_name = re.search(r'instance name="(.*?)"', line).group(0).split('=')[1].strip('\'').strip('\"')
                    if "Riallto" in instance_name:
                        found = True
                        break
        if not found:
            raise RuntimeError("Could not see any Riallto kernels in the xclbin")

        return kernel_name

    def _get_kernel_info(self, xclbin_name):
        """This function parses the xclbin contents for kernels and returns
        the parameters specific to the kernel that the app is being initialized
        with.

        TODO: make this work with axlf sections instead of byte parsing

        Parameters
        ----------
        xclbin_name : str

        Returns
        -------
        kernel_params : list
            List of kernel parameter metadata, i.e. types
        """

        kernel_params = {}

        with open(xclbin_name, 'rb') as f:
            xclbin_data = f.read()

        start = 0
        while 1:
            idx = xclbin_data[start:].find(b"<kernel ")
            if idx == -1:
                break
            start = start + idx + 1
            end = xclbin_data[start:].find(b"</kernel>")

            kernel_info = xclbin_data[start-8:start+end+9].decode()

            params = []
            for line in kernel_info.split('\n'):
                if "kernel name=" in line:
                    kernel_name = re.search(r'kernel name="(.*?)"', line).group(0).split('=')[1].strip('\'').strip('\"')

                if "arg name" in line:

                    name = re.search(r'name="(.*?)"', line)
                    name = name.group(0).split('=')[1].strip('\'').strip('\"')

                    types = re.search(r'type="(.*?)"', line)
                    types = types.group(0).split('=')[1].strip('\'').strip('\"')

                    if name in keyword.kwlist:
                        name = name + '_'

                    params.append(Parameter(name, kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=types))

            kernel_params[self.kernel_name] = {"parameters": params,
                                    "signature": Signature(params)}

        return kernel_params[self.kernel_name]

    def allocate(self, shape, dtype="u1", cacheable=False, param=None, **kwargs):
        """Allocate a new PynqBuffer object.

        This API mimics the numpy ndarray constructor.
        """

        # Default mem bank index for host buffers
        mem_bank_idx = self.__mem_bank_indexes[-1]

        elements = 1
        try:
            for s in shape:
                elements *= s
        except TypeError:
            elements = shape
        dtype = np.dtype(dtype)
        size = elements * dtype.itemsize

        # Instruction buffer should always be cacheable
        if cacheable:
            cache_flag = ipr.bo.flags.cacheable
        else:
            cache_flag = ipr.bo.flags.host_only

        # Create xrt buffer object for kernel port
        try:
            bo = ipr.bo(self.device, size, cache_flag, mem_bank_idx)
        except MemoryError as e:
            print(str(e))
            print("Try resetting your device.")
            raise

        # Expose buffer object memory as python buffer
        buf = bo.map()

        ar = PynqBuffer(
            shape=shape,
            dtype=dtype,
            bo=bo,
            buffer=buf)

        self._allocated_arrays.append(ar)

        return ar

    def call(self, *kwargs):
        """This function abstracts pyxrt.run(), passes pyxrt.bo objects to the
        pyxrt.run() function if they are recognized as PynqBuffer types, otherwise
        passes them as is.
        """

        self._refresh_sequence()
        # run_args = [self.instr, len(self.sequence.buffer)]
        run_args = []
        for arg in kwargs:
            if isinstance(arg, PynqBuffer):
                run_args.append(arg.bo)
            else:
                run_args.append(arg)

        run = self.kernel(self.instr, len(self.sequence.buffer), *run_args)
        ert_state = run.wait(5000) # 5 second timeout

        # Currently this check is only working with the windows bindings
        if (ert_state.value != 4) and (platform.system() == "Windows"):
            raise RuntimeError(f"Returned state is {ert_state}: {ert_state.value}, expected <ert_cmd_state.ERT_CMD_STATE_COMPLETED: 4>")

    def __delete__(self, instance):
        """Delete allocated instructions buffer to deallocate memory.
        Delete other pybind11 objects to make sure resources are freed.
        """
        self.__del__()

    def __del__(self):
        """Delete allocated instructions buffer to deallocate memory.
        Delete other pybind11 objects to make sure resources are freed.
        """
        if hasattr(self, "_allocated_arrays"):
            for ar in self._allocated_arrays:
                ar.free_memory()
        if hasattr(self, "instr"):
            del self.instr
        if hasattr(self, "kernel"):
            del self.kernel
        if hasattr(self, "device"):
            del self.device
        if hasattr(self, "_widgets"):
            for _, widget_list in self._widgets.items():
                for widget_item in widget_list:
                    widget_item.unobserve_all()
        if hasattr(self, "_hboxes"):
            for box in self._hboxes:
                box.unobserve_all()

    @property
    def Signature(self):
        return self.kernel_params['signature']

class PynqBuffer(np.ndarray):
    """This is a subclass of numpy.ndarray. This class is
    intended to be constructed using the AppRunner.allocate()
    method and should not be used as a standalone.

    Attributes
    ----------
    bo: pyxrt.bo
        A pyxrt buffer object.
    cacheable: bool
        Typically host buffers will not be cacheable, but instr
        buffers will always be.

    Note
    ----
    It's important to free the buffer memory after use -- this
    can be done with the `free_memory()` method. The AppRunner
    class tracks the allocated buffers and clears the buffers
    automatically when the object has been deleted.

    """

    def __new__(
        cls, *args, cacheable=False, bo=0, **kwargs
    ):
        self = super().__new__(cls, *args, **kwargs)
        self.bo = bo
        self.cacheable = cacheable
        return self

    def __array_finalize__(self, obj):
        if isinstance(obj, PynqBuffer):
            self.bo = obj.bo

    def sync_to_npu(self):
        self.bo.sync(ipr.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def sync_from_npu(self):
        self.bo.sync(ipr.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, self.bo.size(), 0)

    def free_memory(self):
        if self.bo:
            self.bo = None

    def __del__(self):
        self.free_memory()

    def __getitem__(self, index):
        if self.bo is None:
            raise RuntimeError("Cannot access data, device memory has been freed.")
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        if self.bo is None:
            raise RuntimeError("Cannot modify data, device memory has been freed.")
        super().__setitem__(index, value)

    def __repr__(self):
        if self.bo is None:
            return "<Freed buffer object>"
        return super().__repr__()

    def __str__(self):
        if self.bo is None:
            return "<Freed buffer object>"
        return super().__str__()
