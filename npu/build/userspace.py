# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from .kernelmeta import KernelMeta
from .port import RTPPort


class UserspaceRTP(KernelMeta):
    """This class contains a userspace port for acessing NPU runtime paraemters (RTP)."""
    def __init__(self, value) -> None:
        """Return a new UserspaceRTP object."""         
        rtpport = RTPPort(f"write", self)
        rtpport.value = value

        super().__init__("_", "_", "RTP", "RTP", ports=[rtpport])
        self.name = "user"
        self.ttype = "RTP"
