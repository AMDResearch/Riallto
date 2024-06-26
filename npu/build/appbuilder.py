# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from .connections import Connection
from .appmetadata import AppMetada
from .mlirbuilder import MLIRBuilder
from .apptracer import AppTracer
from .sequence import SequenceList
from .appxclbinbuilder import AppXclbinBuilder
from .utils import check_wsl_install
from typing import Optional
import platform
import json

class AppBuilder:
    """This class is able to build NPU applications from a high level description
    that is specified in the callgraph() method using npu.build objects.

    Attributes
    ----------
    name : str
        The name of the application.
    ab : AppXclbinBuilder
        The xclbin builder class used to build the final xclbin binary.
    fxtracer : AppTracer
        The apptracer class used to build the application metadata.
    kernels : dict
        Dictionary storing all unique compute tile kernels in this application.
    connections : dict
        Dictionary storing all unique connections between kernels in this application.
    previous_build_args : list
        List containing the input arguments last used to build the application.

    Note
    ----
    This class is typically meant to be subclassed with a custom `callgraph()`.
    Many examples of this subclassing pattern are in tests\\test_applications.py.

    """


    def __init__(self, name=None) -> None:
        """Return a new AppBuilder object."""

        if platform.system() == 'Windows':
            check_wsl_install()

        self.name = type(self).__name__ if name is None else name
        self.ab = AppXclbinBuilder()

        self.fxtracer = AppTracer(self)
        self.kernels = None
        self.connections = None

        self.previous_build_args = None

    def __call__(self, *args):
        """ Calling the class will execute the callgraph directly."""
        self.previous_build_args = args
        if args:
            return self.callgraph(*args)
        else:
            return self.callgraph()

    def callgraph(self):
        """ This method should be overridden by a subclass. """
        raise NotImplementedError(f'Subclass needs to implement the callgraph function for use in tracing and behavioral execution')

    def to_metadata(self, *args):
        """ The application is converted into the AppMetadata after tracing the callgraph() call."""
        self.previous_build_args = args
        self.kernels, self.connections = self.fxtracer.to_trace(*args)

        return AppMetada(self.name,
                         self.unique_named(self.kernels),
                         self.unique_named(self.connections),
                         self.to_sequence())

    def to_handoff(self, *args, file=None):
        """ Converts the application into a serializable JSON file."""
        self.previous_build_args = args
        with open(file, 'w') as f:
            json.dump(self.to_json(*args), f, default = lambda o: '<not serialisable>')

    def to_json(self, *args):
        """ Converts the application into JSON."""
        self.previous_build_args = args
        return self.to_metadata(*args).to_json()

    @property
    def metadata(self, *args):
        """ Generates the application JSON and displays inside a IPython environment."""
        from npu import ReprDict
        self.validate_previous_build_args()
        return ReprDict(self.to_json(*self.previous_build_args), rootname=self.name)


    def to_mlir(self, *args, file=None):
        """ Generates the application mlir file from the metadata collected."""
        self.previous_build_args = args
        return MLIRBuilder(self.to_metadata(*args)).to_mlir(file)

    def to_sequence(self):
        """ Generates the application data movement sequence from the connections traced."""
        return SequenceList(self.connections)

    def display(self)->None:
        """ Generates the application SVG and displays inside a IPython environment."""
        from npu.utils.appviz import AppViz
        self.validate_previous_build_args()
        _viz = AppViz(self.to_json(*self.previous_build_args))
        _viz.show

    def save(self, filename:str=None)->None:
        """saves animation to a file."""
        from npu.utils.appviz import AppViz
        _viz = AppViz(self.to_json(*self.previous_build_args))
        _viz.save(filename)

    def displaymlir(self, *args, what:str="")->None:
        """ Displays part of the application in a IPython friendly way."""
        from IPython.display import display, Code
        self.validate_previous_build_args()
        _code = Code(self.to_mlir(*self.previous_build_args), language="llvm")
        display(_code)

    def build(self, *args, debug=False, mlir:Optional[str]=None):
        """ The application is built using the callgraph call with supplied arguments."""
        self.previous_build_args = args
        self.to_mlir(*args, file=f"{self.name}.mlir")
        self.to_handoff(*args, file=f"{self.name}.json")
        if mlir is None:
            self.ab.build(self.name, f"{self.name}.mlir", self.kernels, debug)
        else:
            self.ab.build(self.name, mlir, self.kernels, debug)

    def __add__(self, app_component):
        if isinstance(app_component, Connection):
            self.merge_applications(app_component.kernels, [app_component])
            return self

        if isinstance(app_component, AppBuilder):
            self.merge_applications(app_component.kernels, app_component.connections)
            return self

        raise TypeError(f"{app_component} of type {type(app_component)} is not supported")

    def validate_previous_build_args(self):
        if self.previous_build_args is None:
            raise ValueError(f'Before using this AppBuilder API, please first call the AppBuilder instance directly or call \
                             to_metadata(), to_json() or to_build() with callgraph args to complete the application graph')

    def merge_applications(self, newkernels, newconnections):
        self.connections.extend(newconnections)
        self.kernels.extend(newkernels)

    def unique_named(self, objs):
        unique_objs = list(set(objs))

        unique_objs_byname = {obj.name : obj for obj in unique_objs}
        unique_objs_byname_list = list(obj for _,obj in unique_objs_byname.items())

        unique_objs_byname_list.sort(key= lambda x : x.name)

        return unique_objs_byname_list
