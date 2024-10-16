# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import itertools
import numpy as np
from CppHeaderParser import CppHeader
from .kernelbuilder import KernelObjectBuilder
from .tracekernels import kerneltracer
from .kernelmeta import KernelMeta
from .buffers import Buffer
from .port import BufferPort, RTPPort
from typing import Optional, Callable, List, Dict
import re
import warnings


class Kernel(KernelMeta):
    """This class encapsulates a ComputeTile kernel C/C++ src code and methods to generate a compiled object - that compiled object
    is used within MLIR to build the final xclbin application.  Additionally, the kernel is parsed for input and output ports and can
    encapsulates a behavioral model to capture functional behavior and data shaping from input to output ports.  This metadata
    for the kernel enables behavioral execution to verify correctness in Python and also tracing to build the final AppBuilder xclbin.

    Attributes
    ----------
    srccode : str
        The C/C++ source code of the ComputeTile kernel.
    srcfile : str
        The C/C++ source file path.
    kname : str
        The name of this kernel instance.
    behavioralfx : function
        The behavioral function that emulates the C/C++ kernel's behavior.
    """
    def __init__(self, srccode: str, behavioralfx: Optional[Callable] = None,
                 top_function: Optional[str] = None,
                 requires_boilerplate: bool = False) -> None:
        """Return a new Kernel object."""
        if srccode.endswith('.cpp') or srccode.endswith('.cc'):
            with open(srccode, 'r') as file:
                self._srccode = file.read()
                self.srcfile = srccode
        else:
            self._srccode = srccode
            self.srcfile = None

        self._requires_boilerplate = requires_boilerplate
        self._top_function = top_function
        kname, _parsed_ports, self._main_function = self._parse_code()
        self._top_function = kname

        self.srccode = self.completed_srccode()
        super().__init__(kname, kname, kname, "CT", ports=_parsed_ports)

        if behavioralfx is None:
            self.behavioralfx = _default_behavioral_validate_bufferports
        else:
            self.behavioralfx = behavioralfx

        self.kb = KernelObjectBuilder(self.ktype, self.srccode, self.srcfile)
        self._asmlst = None
        self._main_function_sanity_check()
        self._extern_c_check()
        self._expose_ports()

    def _expose_ports(self) -> None:
        for p in self.ports:
            setattr(self, p.name, p)

    def _parse_code(self):
        """Using CppHeader package, C++ code is parsed to discover ports and their types."""
        parsedcpp = CppHeader(self._srccode, argType='string')
        allports = self._parsecpp_to_ports(parsedcpp)
        functions = self._parse_functions(parsedcpp.functions)
        if self._top_function is None:
            parsedname = parsedcpp.functions[-1]['name']
        else:
            parsedname = self._top_function
        return parsedname, allports, functions[parsedname]

    def _parse_functions(self, functions_l: List) -> Dict:
        """Parse the functions list into a dict."""
        f = {}
        for funcname in functions_l:
            e = {}
            e['name'] = funcname['name']
            e['last'] = (funcname == functions_l[-1])
            e['fullsig'] = funcname['debug']
            params = {}
            for p in funcname['parameters']:
                e_p = {}
                e_p['name'] = p['name']
                e_p['type'] = p['type']
                params[p['name']] = e_p
            e['params'] = params
            e['rtnType'] = funcname['rtnType']
            f[funcname['name']] = e
        return f

    def _main_function_sanity_check(self) -> None:
        if not self._main_function['rtnType'] == "void":
            raise RuntimeError(f"The return type of the top_level function should be void not {self._main_function['rtnType']}")

    def _extern_c_check(self):
        """Verify that extern C is used"""
        tight_code = self.srccode.replace(' ', '').replace('	', '')
        if 'extern"C"' not in tight_code:
            raise SyntaxError('extern "C" not found. Top level function '
                              'should be wrapped by extern "C"')

    def display(self) -> None:
        """Render the kernel code in a jupyter notebook."""
        from IPython.display import display, Code
        _code = Code(self.srccode, language="cpp")
        display(_code)

    def completed_srccode(self) -> str:
        """From the parsed information generate the source."""
        if self._requires_boilerplate:
            preamble = """
#define NOCPP
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define REL_WRITE 0
#define REL_READ 1
#include <aie_api/aie.hpp>

        """
            s = f"{preamble}{self._srccode}"
            sig = self._main_function['fullsig']
            pattern = r'\s*'.join(re.escape(char) for char in sig if char.strip())
            match = re.search(pattern, s)
            if match:
                start_index = match.start()
                s = s[:start_index] + '\nextern "C" {\n' + s[start_index:]
            else:
                raise RuntimeError(f"Unable to find the function {self._top_function} in the src")

            # Walk down the string until we get to the end of the function using a stack to track
            # nested { }
            match = re.search(pattern, s)
            end_index = self._find_matching_brackets(s, match.end()-1)
            s = s[:end_index] + '\n} // extern end\n' + s[end_index:]
        else:
            s = f"{self._srccode}"

        return s

    def _get_ptr_type_depth(self, arg) -> int:
        arg = arg.rstrip()
        count = 0
        for i in reversed(arg):
            if i == "*":
                count += 1
            else:
                break
        return count

    def _find_matching_brackets(self, s, start_index: int):
        stack = []
        for i in range(start_index, len(s)):
            if s[i] == '{':
                stack.append(i)
            elif s[i] == '}':
                stack.pop()

            if len(stack) == 0:
                return i+1

        raise RuntimeError("Unable to find closing brace for "
                           f"{self._main_function['fullsig']}")

    def to_cpp(self) -> None:
        """ output source code to a .cpp file"""
        with open(f'{self.name}.cpp', 'w') as file:
            file.write(self.completed_srccode())

    @kerneltracer
    def __call__(self, *args, behavioral_n_tracing=True):
        '''Tracer through a ComputeTile kernel call
        1. set port types to "in" or "out"
        2. set input port sizes based on input array sizes
        3. return tuple of output ports for use in subsequent tracing
        '''

        if self.behavioralfx is None:
            raise ValueError(f'Unable to trace Kernel I/O with no behavioral model for kernel {self.name}')

        bufferargs = [a for a in args if isinstance(a, BufferPort) or isinstance(a, Buffer) or isinstance(a, np.ndarray)]
        rtpargs = [a for a in args if isinstance(a, int)]

        self._validate_args(args, bufferargs, rtpargs, behavioral_n_tracing)
        self._set_arg_io_values(bufferargs, rtpargs)

        self.behavioralfx(self)

        output_buffers = self.create_outputs(behavioral_n_tracing)

        return output_buffers

    def _validate_args(self, args, bufferargs, rtpargs, behavioral_n_tracing):

        if len(args) != len(bufferargs + rtpargs):
            supported_argtypes = [BufferPort, Buffer, np.ndarray, int]
            raise TypeError(f'Unable to parse some datatype(s) found in args: {args} ... currently supporting {supported_argtypes} datatypes')

        if len(bufferargs) >= len(self.bufferports):
            raise IndexError(f'Kernel {self.name} called with {len(bufferargs)}, expecting less than {len(self.bufferports)}')

        if behavioral_n_tracing is True and not all([isinstance(a, np.ndarray) for a in bufferargs]):
            raise ValueError(f'Kernel {self.name} called with non-ndarray buffers while running a behavioral call')

        if len(self.inputbufferports) > 0 and len(self.inputbufferports) != len(bufferargs):
            raise ValueError(f'Kernel {self.name} called with {len(bufferargs)} buffer args, but previously called with {len(self.inputbufferports)} buffer args')

        if len(rtpargs) != len(self.rtpports):
            raise ValueError(f'Kernel {self.name} called with {len(rtpargs)} RTP args, but expecting {len(self.rtpports)} RTP args')

    def _set_arg_io_values(self, bufferargs, rtpargs):
        '''Map C/C++ arguments to BufferPorts and RTP ports.'''
        mapped_buffers = itertools.zip_longest(self.bufferports, bufferargs)

        for bufferport, input_arg in mapped_buffers:
            if input_arg is not None:
                bufferport.io = 'in'
                bufferport.array = Buffer.to_ndarray(input_arg)
            else:
                bufferport.io = 'out'
                bufferport.slices = list()

        for rtpport, call_value in zip(self.rtpports, rtpargs):
            rtpport.value = call_value

    def create_outputs(self, behavioral_n_tracing):
        """From kernel call, produce the output value or tuple."""
        if behavioral_n_tracing is True:
            outputs = [Buffer.to_ndarray(op) for op in self.outputbufferports]
        else:
            outputs = self.outputbufferports

        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

    def build(self, debug=False):
        """Build the kernel object file for linking into the complete application."""
        if not os.path.exists(self.kb.buildobjpath):
            self.kb.build(debug)
            with open(self.kb.buildobjpath + '.lst', 'r', encoding='utf-8') as file:
                vliwasm = file.read()
            self._asmlst = '\n'.join(vliwasm.split('\n')[6:])

    @property
    def objfile(self):
        self.build()
        return self.kb.buildobjpath

    @property
    def asm(self):
        """Returns string of VLIW Assembly instructions"""
        if self._asmlst is None:
            raise RuntimeError(f'Kernel: {self.name} is not built (compiled). '
                               'Build kernel to check assembly')
        return self._asmlst

    def asmdisplay(self) -> None:
        """Render the VLIW Assembly instructions in a jupyter notebook"""
        from IPython.display import display, Code
        display(Code(self.asm, language="c-objdump"))

    def _parsecpp_to_ports(self, parsedcpp):
        bufferports = [BufferPort(param['name'], param['type'])
                       for param in parsedcpp.functions[-1]["parameters"]
                       if '*' in param['type']]

        rtpports = [RTPPort(param['name'], param['type'], c_dtype=param['type'])
                    for param in parsedcpp.functions[-1]["parameters"]
                    if '*' not in param['type']]

        return bufferports + rtpports


def _default_behavioral_validate_bufferports(invobj):
    """ A behavioural model that gives users guidance to build a behavorial model or
        set array sizes before calling the kernel."""
    for p in invobj.ports:
        if not isinstance(p, RTPPort):
            if p.array is None:
                raise RuntimeError(f"Default behavioral model is being used but cannot determine shape for port {p.name} - \
                                   please specify a behavioral function for this kernel or set the array sizes before \
                                   using the kernel.  E.g. {p.name}.array = np.ndarray(...)")
