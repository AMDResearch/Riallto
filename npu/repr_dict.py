# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import json


def _default_repr(obj):
    return repr(obj)


class ReprDict(dict):
    """Subclass of the built-in dict that will display using the JupyterLab
    JSON repr.

    The class is recursive in that any entries that are also dictionaries
    will be converted to ReprDict objects when returned.

    """

    def __init__(self, *args, rootname="root", expanded=False, **kwargs):
        """Dictionary constructor

        Parameters
        ----------
        rootname : str
            The value to display at the root of the tree
        expanded : bool
            Whether the view of the tree should start expanded

        """
        self._rootname = rootname
        self._expanded = expanded

        super().__init__(*args, **kwargs)

    def _repr_json_(self):
        return json.loads(json.dumps(self, default=_default_repr)), {
            "expanded": self._expanded,
            "root": self._rootname,
        }

    def __getitem__(self, key):
        obj = super().__getitem__(key)
        if type(obj) is dict:
            return ReprDict(obj, expanded=self._expanded, rootname=key)
        else:
            return obj
