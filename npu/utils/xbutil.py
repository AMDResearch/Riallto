# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
from io import FileIO
from pathlib import Path
from typing import List, Dict, Set
import subprocess
import tempfile
import json
import shutil
import sys
import time

XBUTIL_DIR = Path("C:\Windows\System32\AMD")

class XBUtil:

    def __init__(self):
        """ A class that wraps xbutil so that information can be 
        parsed about currently running applications on the NPU
        device. 

        Attributes
        ----------
        _xbutil : Path
            A Path or command to call the xbutil.exe application
        _devices : Set[str]
            A set of devices that are present on this machine.
        devid : str
            A unique string for the Phx device on this machine.
        """
        self._xbutil = Path(f"{XBUTIL_DIR}/xbutil.exe")
        self._check_xbutil_install()
        self._devices = self._get_devices()
        self._check_devices()
        self.devid = next(iter(self._devices))

    def app_exists(self, name:str)->bool:
        """ Returns true if an app with the given name is 
        present on the NPU device """
        riallto_pattern = "Riallto"
        for f in self._get_loaded_functions():
            if f.endswith(riallto_pattern):
                if f.startswith(name):
                    return True
        return False

    @property
    def app_table(self)->str:
        """ Returns a table of all the apps currently loaded onto the NPU device. """
        s = "Currently loaded apps:\n"
        for a in self.list_apps():
            s += f"\t{a}\n"
        return s

    def list_apps(self)->List[str]:
        """ Lists all the apps running on the
        NPU device """
        l = []
        wse_pattern = "IPUV1CNN" 
        riallto_pattern = "Riallto"
        for f in self._get_loaded_functions():
            if f.endswith(riallto_pattern):
                l.append(f)
            elif f.endswith(wse_pattern):
                l.append(f)
        return l 

    def _apps(self)->None:
        """ displays all apps that are running on the NPU
        device using an IPython widget """
        import ipywidgets as widgets
        from IPython.display import display
        output1 = widgets.Output()
        output2 = widgets.Output() # Multiple widgets to avoid flickering during update
        display(output1)
        while True:
            apps = self.list_apps()
            if len(apps) > 0:
                max_app_name = max(len(max(apps, key=len)), 10) 
            else:
                max_app_name = 10 

            s = f"Currently Running NPU apps: (update rate 1s)\n"
            s = f"| {'app name': <{max_app_name}} | {'Num columns': <14} | {'start column': <14} |\n"
            s += '-'*max_app_name + '-'*38 + '\n'

            if len(apps) == 0:
                s += ' '*( (int)((max_app_name+38)/2) - 12) + "No apps currently loaded\n"  
            else:

                for a in apps:
                    s += f"  {a: <{max_app_name}}   {1: <14}   {'?': <14}  \n"
            output2.append_stdout(s)
            output1.outputs = output2.outputs
            output2.outputs = ()
            time.sleep(1)

    def apps(self)->None:
        """ Starts a thread displaying all the apps in an ipython widget. """
        import threading
        thread = threading.Thread(target=self._apps)
        thread.start()


    @property
    def app_count(self)->int:
        """ The total number of running apps """
        return self.num_wse_streams + self.num_riallto_streams

    @property
    def num_wse_streams(self)->int:
        """ Returns the current number of active WSE streams """
        return self.loaded_functions.count('DPU_1x4:IPUV1CNN')

    @property
    def num_riallto_streams(self)->int:
        return sum(1 for app in self.loaded_functions if 'Riallto' in app)

    def _get_loaded_functions(self)->List[str]:
        """ Returns a list of the loaded functions on the NPU from xbutil. """
        l = []
        d = self._cmd(['examine', '-d', self.devid, '-r', 'dynamic-regions'])
        for f in d['devices'][0]['dynamic_regions'][0]['compute_units']:
            l.append(f['name'])
        return l

    @property
    def loaded_functions(self)->List[str]:
        """ Returns a list of loaded functions on the NPU. Applications can have multiple functions."""
        return self._get_loaded_functions()

    def _check_xbutil_install(self):
        """ Returns true if xbutil.exe is available. """
        _ = self._cmd(['examine', '-d'])

    def _cmd(self, cmdlist:List[str])->Dict:
        """ Runs an xbutil.exe command to produce a json report that is parsed into a dict and returned. """
        try:
            tmp_dir = tempfile.mkdtemp()
            _t = Path(tmp_dir) / "temp.json"
            output = subprocess.check_output([self._xbutil]  
                                             + cmdlist
                                             + ['-f', 'json', '-o', _t.absolute(), '--force']
                                             , stderr=subprocess.STDOUT)
            return json.load(FileIO(_t.absolute()))
        except subprocess.CalledProcessError as e:
            print(f"xbutil command failed.\n\n {e.output.decode()}")
            raise e

    def _get_devices(self)->Set[str]:
        """ Get the set of devices on this machine"""
        t = []
        examine_out = self._cmd(['examine', '-d'])
        devs = examine_out['system']['host']
        for d in devs['devices']:
            t.append(d['bdf'])
        return set(t)

    def _check_devices(self)->None:
        """Raises an error if a problem is detected with the device. """
        if len(self._devices) > 1:
            raise RuntimeError("Unable to determine which device is the NPU, "
                               f"devices found = {self._devices}")
        if len(self._devices) == 0:
            raise RuntimeError("Unable to find any NPU devices via XBUtil")
