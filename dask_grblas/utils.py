import os
import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
from dask.base import tokenize
from dask.delayed import delayed
from .io import MMFile


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


def build_block_index_dask_array(x, axis, name):
    """
    Calculate block-index for each chunk of x along axis `axis`
    e.g. chunks=(..., (5, 3, 4), ...) -> x_axis_indices=[0, 1, 2]
    """
    # it is vital to give a unique name to this dask array
    name = name + tokenize(x, axis, x.numblocks[axis])
    indices = da.arange(x.numblocks[axis], chunks=1, name=name)
    # Tamper with the declared chunks of `indices` to make blockwise align it with
    # x[axis]
    return da.core.Array(indices.dask, indices.name, (x.chunks[axis],), indices.dtype, meta=x._meta)


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
    Calculate ranges at which each chunk of `x` starts along axis `axis`
    e.g. chunks=(..., (5, 3, 4), ...) -> x_range=[slice(0, 5), slice(5, 8), slice(8, 12)]
    """
    return build_ranges_dask_array_from_chunks(x.chunks[axis], name, axis=0)


def build_ranges_dask_array_from_chunks(chunks, name, axis=0):
    """
    Calculate ranges at which each chunk starts,
    e.g. chunks=(..., (5, 3, 4), ...) -> x_range=[slice(0, 5), slice(5, 8), slice(8, 12)]
    """
    offset = np.roll(np.cumsum(chunks), 1)
    offset[0] = 0
    ranges = np.array([slice(start, start + len) for (start, len) in zip(offset, chunks)])
    # it is vital to give a unique name to this dask array
    name = name + tokenize(ranges, axis)
    ranges = da.core.from_array(ranges, chunks=1, name=name)
    # Tamper with the declared chunks of offset to make blockwise align it with
    # x[axis]
    return da.core.Array(ranges.dask, ranges.name, (chunks,), ranges.dtype, meta=ranges._meta)


def build_slice_dask_array_from_chunks(s, size, chunks="auto"):
    start, stop, step = s.indices(size)
    indx = slice(start, stop, step)
    (chunk_sizes,) = da.core.normalize_chunks(chunks, shape=(size,), dtype=int)
    chunk_offsets = np.roll(np.cumsum(chunk_sizes), 1)
    chunk_offsets[0] = 0
    starts = start + step * chunk_offsets
    stops = start + step * (chunk_offsets + chunk_sizes)
    indx = np.array([slice(start, stop, step) for start, stop in zip(starts, stops)])
    name = "slice_index-" + tokenize(indx, 0)
    indx = da.core.from_array(indx, chunks=1, name=name)
    return da.Array(indx.dask, indx.name, (chunk_sizes,), indx.dtype, meta=indx._meta)


def rcd_df(r, c, d):
    return pd.DataFrame({"r": r, "c": c, "d": d})


def _read_MMFile_part(filename, line_start=None, line_stop=None, read_begin=None, read_end=None):
    r, c, d = MMFile().read_part(
        filename,
        line_start=line_start,
        line_stop=line_stop,
        read_begin=read_begin,
        read_end=read_end,
    )
    return rcd_df(r, c, d)


def wrap_dataframe(filename, nreaders):
    from scipy.io import mminfo
    from math import ceil

    if nreaders is None:
        raise ValueError("nreaders (the number of parallel readers) must be specified.")

    rows, cols, entries, format, field, symmetry = mminfo(filename)

    if format == "coordinate":
        import os

        read_begin = MMFile().get_data_begin(filename)
        read_end = os.path.getsize(filename)

        chunksize = int(ceil((read_end - read_begin) / nreaders))
        locations = list(range(read_begin, read_end, chunksize)) + [read_end]

        dfs = [
            delayed(_read_MMFile_part)(filename, read_begin=start, read_end=stop)
            for (start, stop) in zip(locations[:-1], locations[1:])
        ]

    elif format == "array":
        if symmetry == "general":
            nnz = rows * cols
        elif symmetry == "skew":
            nnz = (rows * cols - rows) // 2
        else:
            nnz = (rows * cols + rows) // 2

        chunksize = int(ceil(nnz / nreaders))
        locations = list(range(0, nnz, chunksize)) + [nnz]

        dfs = [
            delayed(_read_MMFile_part)(filename, line_start=start, line_stop=stop)
            for (start, stop) in zip(locations[:-1], locations[1:])
        ]

    if field == "complex":
        raise NotImplementedError()
    elif field == "real":
        field_dtype = np.float64
    else:
        field_dtype = np.int64

    meta = pd.DataFrame(
        {
            "r": pd.Series(dtype=np.int64),
            "c": pd.Series(dtype=np.int64),
            "d": pd.Series(dtype=field_dtype),
        }
    )
    df = dd.from_delayed(dfs, meta)
    return df, rows, cols


# These will be finished constructed in __init__
_grblas_types = {}
_inner_types = {}
_return_types = {}
