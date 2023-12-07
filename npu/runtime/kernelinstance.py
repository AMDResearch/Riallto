# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

class KernelInstance():
    """ A Kernel instance object that is used when associating RTPs at runtime. 

    Attributes
    -------------------------
    _portlist : list
        A list of ports associated with the kernel.
    _tloc : tuple[int, int]
        The mapped location of the kernel.
    """

    def __init__(self):
        self._portlist = []
        self._tloc = (0, 0)
        pass
