# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import copy


class AsmVLIWInstructions:
    """This class encapsulates logic to analyze a Kernel VLIW instructions"""

    def __init__(self, buildobjpath):
        with open(buildobjpath + '.lst', 'r', encoding='utf-8') as file:
            asmcode = file.read()
        self._vliwasm = '\n'.join(asmcode.split('\n')[6:])

    @property
    def asm(self):
        """Returns assembly instructions of the compiled kernel"""
        return self._vliwasm

    @property
    def loops(self):
        """Returns a dict with the VLIW instructions for each loop"""
        asm = copy.deepcopy(self._vliwasm)
        inloop = False
        loop_nesting = 0
        loop_dict = {}
        loop_count = 0
        for line in asm.split('\n'):
            if '.loop_nesting' in line:
                loop_nesting = int(line.split(' ')[1])
                continue

            if '.begin_of_loop' in line:
                inloop = True
                loop_dict[loop_count] = {
                    'loop_nesting': loop_nesting,
                    'asm': []
                }
                continue

            if '.end_of_loop' in line:
                inloop = False
                loop_count += 1

            if '.noswbrkpt' in line or 'nohwbrkpt' in line:
                continue

            if inloop and '.label' not in line:
                vliwops = []
                for i in range(92, len(line), 30):
                    vliwops.append(line[i:i+30].replace('          ','').split(';')[0])
                loop_dict[loop_count]['asm'].append(vliwops)

        return loop_dict
