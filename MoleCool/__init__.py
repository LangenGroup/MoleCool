# -*- coding: utf-8 -*-
try:
    # Preferred: get version from installed metadata
    from importlib.metadata import version as _get_version
    __version__ = _get_version("MoleCool")
except Exception:
    # Fallback: use setuptools_scm-generated _version.py
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "0.0.0"  # last-resort fallback

from .System import *