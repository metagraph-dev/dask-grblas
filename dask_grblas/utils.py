import dask.array as da
import numpy as np
from dask.base import tokenize


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


def build_chunk_offsets_dask_array(x, axis, name):
    """
    Calculate offsets at which each chunk of x starts along axis `axis`
    e.g. chunks=(..., (5, 3, 4), ...) -> x_offset=[0, 5, 8]
    """
    offset = np.roll(np.cumsum(x.chunks[axis]), 1)
    offset[0] = 0
    # it is vital to give a unique name to this dask array
    name = name + tokenize(offset, axis)
    offset = da.core.from_array(offset, chunks=1, name=name)
    # Tamper with the declared chunks of offset to make blockwise align it with
    # x[axis]
    return da.core.Array(offset.dask, offset.name, (x.chunks[axis],), offset.dtype, meta=x._meta)


def build_chunk_ranges_dask_array(x, axis, name):
    """
    Calculate ranges at which each chunk of starts along axis `axis`
    e.g. chunks=(..., (5, 3, 4), ...) -> x_range=[slice(0, 5), slice(5, 8), slice(8, 12)]
    """
    offset = np.roll(np.cumsum(x.chunks[axis]), 1)
    offset[0] = 0
    ranges = np.array([slice(start, start + len) for (start, len) in zip(offset, x.chunks[axis])])
    # it is vital to give a unique name to this dask array
    name = name + tokenize(ranges, axis)
    ranges = da.core.from_array(ranges, chunks=1, name=name)
    # Tamper with the declared chunks of offset to make blockwise align it with
    # x[axis]
    return da.core.Array(
        ranges.dask, ranges.name, (x.chunks[axis],), ranges.dtype, meta=ranges._meta
    )


# These will be finished constructed in __init__
_grblas_types = {}
_inner_types = {}
_return_types = {}
