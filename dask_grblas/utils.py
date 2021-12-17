import numpy as np


def np_dtype(dtype):
    return np.dtype(dtype.numba_type.name)


def get_meta(val):
    return getattr(val, "_meta", val)


def get_grblas_type(val):
    return _grblas_types[type(val)]


def get_inner_type(val):
    return _inner_types[type(val)]


def get_return_type(val):
    return _return_types[type(val)]


def wrap_inner(val):
    return _inner_types[type(val)](val)


# These will be finished constructed in __init__
_grblas_types = {}
_inner_types = {}
_return_types = {}
