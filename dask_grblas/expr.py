from functools import partial, reduce
from numbers import Number
import dask.array as da
import grblas as gb
import numpy as np


from .base import BaseType, InnerBaseType
from .mask import Mask
from .utils import (
    get_grblas_type,
    get_meta,
    get_return_type,
    np_dtype,
    wrap_inner,
    build_chunk_offsets_dask_array,
    build_chunk_ranges_dask_array,
)
from grblas.exceptions import DimensionMismatch
from dask.base import tokenize


class GbDelayed:
    def __init__(self, parent, method_name, *args, meta, **kwargs):
        self.parent = parent
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
        self._meta = meta

    def _matmul(self, meta, mask=None):
        left_operand = self.parent
        right_operand = self.args[0]
        op = self.args[1]

        if (len(left_operand.shape) == 2) and left_operand._is_transposed:
            a = left_operand._matrix._delayed
            at = True
            lhs_ind = "ji"
        else:
            a = left_operand._delayed
            at = False
            lhs_ind = "ij" if (a.ndim == 2) else "j"

        if (len(right_operand.shape) == 2) and right_operand._is_transposed:
            b = right_operand._matrix._delayed
            bt = True
            rhs_ind = "kj"
        else:
            b = right_operand._delayed
            bt = False
            rhs_ind = "jk" if (b.ndim == 2) else "j"

        if len(lhs_ind) == 2:
            out_ind = "ik" if (len(rhs_ind) == 2) else "i"
        else:
            out_ind = "k" if (len(rhs_ind) == 2) else ""

        no_mask = mask is None
        grblas_mask_type = None
        if no_mask:
            args = [a, lhs_ind, b, rhs_ind]
        else:
            grblas_mask_type = get_grblas_type(mask)
            args = [mask.mask._delayed, out_ind, a, lhs_ind, b, rhs_ind]

        return da.core.blockwise(
            partial(_matmul, op, at, bt, meta.dtype, no_mask, grblas_mask_type),
            out_ind,
            *args,
            meta=wrap_inner(meta),
        )

    def _matmul2(self, meta, mask=None):
        left_operand = self.parent
        right_operand = self.args[0]
        # out_ind includes all dimensions to prevent full contraction
        # in the blockwise below

        a_ndim = len(left_operand.shape)
        if a_ndim == 2:
            out_ind = (0, 1, 2)
            compress_axis = 1
            if left_operand._is_transposed:
                a = left_operand._matrix._delayed
                at = True
                lhs_ind = (1, 0)
            else:
                a = left_operand._delayed
                at = False
                lhs_ind = (0, 1)
        else:
            compress_axis = 0
            out_ind = (0, 1)
            a = left_operand._delayed
            lhs_ind = (0,)
            at = False

        if len(right_operand.shape) == 2:
            if right_operand._is_transposed:
                b = right_operand._matrix._delayed
                bt = True
                rhs_ind = (a_ndim, a_ndim - 1)
            else:
                b = right_operand._delayed
                bt = False
                rhs_ind = (a_ndim - 1, a_ndim)
        else:
            out_ind = out_ind[:-1]
            b = right_operand._delayed
            rhs_ind = (a_ndim - 1,)
            bt = False

        op = self.args[1]
        sum_meta = wrap_inner(meta)
        if mask is None:
            out = da.core.blockwise(
                partial(_matmul2, op, meta.dtype, at, bt),
                out_ind,
                a,
                lhs_ind,
                b,
                rhs_ind,
                adjust_chunks={compress_axis: 1},
                dtype=np.result_type(a, b),
                concatenate=False,
                meta=FakeInnerTensor(meta, compress_axis),
            )
        else:
            m = mask.mask._delayed
            grblas_mask_type = get_grblas_type(mask)
            mask_ind = list(out_ind)
            mask_ind.remove(compress_axis)
            mask_ind = tuple(mask_ind)
            out = da.core.blockwise(
                partial(_matmul2_masked, op, meta.dtype, at, bt, grblas_mask_type),
                out_ind,
                m,
                mask_ind,
                a,
                lhs_ind,
                b,
                rhs_ind,
                adjust_chunks={compress_axis: 1},
                dtype=np.result_type(a, b),
                concatenate=False,
                meta=FakeInnerTensor(meta, compress_axis),
            )

        # out has an extra dimension (a slab or a bar), and now reduce along it
        out = sum_by_monoid(op.monoid, out, axis=compress_axis, meta=sum_meta)
        return out

    def _reduce_along_axis(self, axis, dtype):
        assert not self.kwargs
        op = self.args[0]
        delayed = da.reduction(
            self.parent._delayed,
            partial(_reduce_axis, op, dtype),
            partial(_reduce_axis_combine, op),
            concatenate=False,
            dtype=np_dtype(dtype),
            axis=axis,
        )
        return delayed

    def _reduce_scalar(self, dtype):
        assert not self.kwargs
        op = self.args[0]
        delayed = da.reduction(
            self.parent._delayed,
            partial(_reduce_scalar, op, dtype),
            partial(_reduce_combine, op),
            concatenate=False,
            dtype=np_dtype(dtype),
        )
        return delayed

    def _reduce(self, dtype):
        assert not self.kwargs
        op = self.args[0]
        delayed = da.reduction(
            self.parent._delayed,
            partial(_reduce, op, dtype),
            partial(_reduce_combine, op),
            concatenate=False,
            dtype=np_dtype(dtype),
        )
        return delayed

    def new(self, *, dtype=None, mask=None):
        if mask is not None:
            assert isinstance(mask, Mask)
            meta = self._meta.new(dtype=dtype, mask=mask._meta)
            delayed_mask = mask.mask._delayed
            grblas_mask_type = get_grblas_type(mask)
        else:
            meta = self._meta.new(dtype=dtype)
            delayed_mask = None
            grblas_mask_type = None

        if self.method_name == "reduce":
            delayed = self._reduce(meta.dtype)
        elif self.method_name == "reduce_scalar":
            delayed = self._reduce_scalar(meta.dtype)
        elif self.method_name == "reduce_rowwise":
            delayed = self._reduce_along_axis(1, meta.dtype)
        elif self.method_name == "reduce_columnwise":
            delayed = self._reduce_along_axis(0, meta.dtype)
        elif self.method_name in {"apply", "ewise_add", "ewise_mult"}:
            self_kwargs = {
                key: (
                    self.kwargs[key]._delayed
                    if isinstance(self.kwargs[key], BaseType)
                    else self.kwargs[key]
                )
                for key in self.kwargs
            }
            delayed = da.core.elemwise(
                _expr_new,
                self.method_name,
                dtype,
                grblas_mask_type,
                self_kwargs,
                self.parent._delayed,
                delayed_mask,
                *[x._delayed if isinstance(x, BaseType) else x for x in self.args],
                dtype=np_dtype(meta.dtype),
            )
        elif self.method_name in {"vxm", "mxv", "mxm"}:
            # TODO: handle dtype and mask
            delayed = self._matmul2(meta, mask=mask)
        else:
            raise ValueError(self.method_name)
        return get_return_type(meta)(delayed)

    def _update(self, updating, *, mask=None, accum=None, replace=None):
        updating._meta.update(self._meta)
        assert updating._meta._is_scalar or updating._meta.nvals == 0
        meta = updating._meta

        if self.method_name == "reduce":
            meta.clear()
            delayed = self._reduce(meta.dtype)
            # Is it important to call `update` on the scalar?
            # For now, let's say no.
            # Also, is it possible for dtypes to be different?  Should we check?
            if accum is not None:
                delayed = da.core.elemwise(
                    _reduce_accum,
                    updating._delayed,
                    delayed,
                    accum,
                    dtype=np_dtype(updating.dtype),
                )
        elif self.method_name == "reduce_scalar":
            meta.clear()
            delayed = self._reduce_scalar(meta.dtype)
            if accum is not None:
                delayed = da.core.elemwise(
                    _reduce_accum,
                    updating._delayed,
                    delayed,
                    accum,
                    dtype=np_dtype(updating.dtype),
                )
        elif self.method_name == "reduce_rowwise":
            meta.clear()
            delayed = self._reduce_along_axis(1, meta.dtype)
            if accum is not None:
                delayed = da.core.elemwise(
                    _reduce_axis_accum,
                    updating._delayed,
                    delayed,
                    accum,
                    dtype=np_dtype(updating.dtype),
                )
        elif self.method_name == "reduce_columnwise":
            meta.clear()
            delayed = self._reduce_along_axis(0, meta.dtype)
            if accum is not None:
                delayed = da.core.elemwise(
                    _reduce_axis_accum,
                    updating._delayed,
                    delayed,
                    accum,
                    dtype=np_dtype(updating.dtype),
                )
        elif self.method_name in {"apply", "ewise_add", "ewise_mult"}:
            delayed = updating._optional_dup()
            self_kwargs = {
                key: (
                    self.kwargs[key]._delayed
                    if isinstance(self.kwargs[key], BaseType)
                    else self.kwargs[key]
                )
                for key in self.kwargs
            }
            if mask is None and accum is None:
                delayed = da.core.elemwise(
                    _update_expr,
                    self.method_name,
                    delayed,
                    self.parent._delayed,
                    self_kwargs,
                    *[x._delayed if isinstance(x, BaseType) else x for x in self.args],
                    dtype=np_dtype(meta.dtype),
                )
            else:
                if mask is not None:
                    delayed_mask = mask.mask._delayed
                    grblas_mask_type = get_grblas_type(mask)
                else:
                    delayed_mask = None
                    grblas_mask_type = None
                delayed = da.core.elemwise(
                    _update_expr_full,
                    self.method_name,
                    delayed,
                    accum,
                    delayed_mask,
                    grblas_mask_type,
                    replace,
                    self.parent._delayed,
                    self_kwargs,
                    *[x._delayed if isinstance(x, BaseType) else x for x in self.args],
                    dtype=np_dtype(meta.dtype),
                )
        elif self.method_name in {"vxm", "mxv", "mxm"}:
            delayed = self._matmul2(meta, mask=mask)
            updating(mask=mask, accum=accum, replace=replace) << get_return_type(meta)(delayed)
            return
        else:
            raise ValueError(self.method_name)
        updating._delayed = delayed

    @property
    def value(self):
        self._meta.value
        return self.new().value


class Updater:
    def __init__(self, parent, *, mask=None, accum=None, replace=False, input_mask=None):
        self.parent = parent
        self.mask = mask
        self.accum = accum
        if mask is None:
            self.replace = None
        else:
            self.replace = replace
        self._meta = parent._meta(mask=get_meta(mask), accum=accum, replace=replace)

    def __getitem__(self, keys):
        return Assigner(self, keys)

    def __setitem__(self, keys, obj):
        Assigner(self, keys).update(obj)

    def __lshift__(self, delayed):
        # Occurs when user calls C(params) << delayed
        self.update(delayed)

    def update(self, delayed):
        # Occurs when user calls C(params).update(delayed)
        # self.parent._update(delayed, **self.kwargs)
        if self.mask is None and self.accum is None:
            return self.parent.update(delayed)
        self.parent._meta._update(
            get_meta(delayed),
            mask=get_meta(self.mask),
            accum=self.accum,
            replace=self.replace,
        )
        if self.parent._meta._is_scalar:
            self.parent._update(delayed, accum=self.accum)
        else:
            self.parent._update(delayed, accum=self.accum, mask=self.mask, replace=self.replace)


class Fragmenter:
    """
    stores only that part of the data-chunk selected by the index
    """

    def __init__(self, ndim=None, index=None, mask=None, obj=None):
        self.ndim = ndim
        self.index = index
        self.mask = mask
        self.obj = obj


def _ceildiv(a, b):
    return -(a // -b)


def _squeeze(tupl):
    if len(tupl) == 1:
        return tupl[0]
    return tupl


def _shape(x, indices):
    shape = ()
    for axis, index in enumerate(indices):
        if isinstance(index, Number):
            shape += ()
        elif type(index) is slice:
            start, stop, step = index.indices(x.shape[axis])
            shape += (_ceildiv(stop - start, step),)
        elif type(index) in {list, np.ndarray}:
            shape += (len(index),)
        elif type(index) is da.Array:
            shape += (index.shape[0],)
    return shape


def fuse_slice_pair(slice0, slice1, length):
    """computes slice `s` such that array[s] = array[s0][s1] where array has length `length`"""
    a0, o0, e0 = slice0.indices(length)
    a1, o1, e1 = slice1.indices(_ceildiv(abs(o0 - a0), abs(e0)))
    o01 = a0 + o1 * e0
    return slice(a0 + a1 * e0, None if o01 < 0 else o01, e0 * e1)


def fuse_index_pair(i, j, length=None):
    """computes indices `s` such that array[s] = array[i][j] where array has length `length`"""
    if type(i) in {list, np.ndarray}:
        return i[j]
    if length is None:
        raise ValueError("Length argument is missing")
    if type(i) is slice and isinstance(j, Number):
        a0, _, e0 = i.indices(length)
        return a0 + j * e0
    if type(i) is slice and type(j) in {list, np.ndarray}:
        a0, _, e0 = i.indices(length)
        f = lambda x: a0 + x * e0
        return [f(x) for x in j] if type(j) is list else list(f(j))
    elif type(i) is slice and type(j) is slice:
        return fuse_slice_pair(i, j, length=length)
    else:
        raise NotImplementedError()


def _chunk_in_slice(chunk_begin, chunk_end, slice_start, slice_stop, slice_step):
    """returns the part of the chunk that lies within the slice,
    also returning True if it exists, otherwise False"""

    if slice_step > 0:
        cb = chunk_begin
        ce = chunk_end
        start, stop, step = slice_start, slice_stop, slice_step
    else:
        cb = -(chunk_end - 1)
        ce = -(chunk_begin - 1)
        start, stop, step = -slice_start, -slice_stop, -slice_step
    if start < ce and stop > cb:
        if start < cb:
            rem = (cb - start) % step
            start = cb + (step - rem) if rem > 0 else cb
        stop = min(ce, stop)
        idx_within = _ceildiv(stop - start, step) > 0
        if idx_within:
            idx = slice(start, stop, step) if slice_step > 0 else slice(-start, -stop, -step)
        else:
            idx = None
        return idx_within, idx
    else:
        return False, np.array([], dtype=int)


def _data_x_index_meshpoint_4assign(*args, x_ndim, subassign, obj_offset_axes):
    x_ranges = args[0:x_ndim]
    indices = args[x_ndim : 2 * x_ndim]

    mask = args[2 * x_ndim]
    obj = args[2 * x_ndim + 1]
    if obj_offset_axes:
        obj_axis_offsets = args[2 * x_ndim + 2 :]
        obj_offsets = [None] * x_ndim
        for axis, offsets in zip(obj_offset_axes, obj_axis_offsets):
            obj_offsets[axis] = offsets[0]

    mask_ = mask
    if mask is not None:
        mask_ = mask.value
        y = mask_

    obj_ = obj
    if isinstance(obj, InnerBaseType):
        obj_ = obj.value
        if obj.ndim > 0:
            y = obj_

    obj_is_scalar = not (isinstance(obj, InnerBaseType) and obj.ndim > 0)
    index_tuple = ()
    obj_index_tuple = ()
    obj_axis = 0
    for axis in range(x_ndim):
        index = indices[axis]
        axis_len = x_ranges[axis][0].stop - x_ranges[axis][0].start
        idx_offset = x_ranges[axis][0].start
        obj_idx = None
        if type(index) is np.ndarray:
            # CASE: array
            idx = index - idx_offset
            idx_filter = (idx >= 0) & (idx < axis_len)
            idx_within = np.any(idx_filter)
            idx = idx[idx_filter]
            if (subassign and mask is not None) or not obj_is_scalar:
                # Here mask (if it exists) and index are already aligned
                obj_idx = np.argwhere(idx_filter)[:, 0]
                obj_index_tuple += (obj_idx,)
        elif type(index) is slice:
            # CASE: slice
            s = index
            data_begin, data_end = 0, axis_len
            if (subassign and mask is not None) or not obj_is_scalar:
                # Here mask and obj are already aligned if both exist
                obj_begin = s.start + s.step * obj_offsets[axis]
                obj_end = s.start + s.step * (obj_offsets[axis] + y.shape[obj_axis])
                obj_begin = obj_begin - idx_offset
                obj_end = obj_end - idx_offset
                idx_within, idx = _chunk_in_slice(data_begin, data_end, obj_begin, obj_end, s.step)
                if idx_within:
                    stop = _ceildiv(idx.stop - obj_begin, s.step)
                    obj_idx = slice(
                        _ceildiv(idx.start - obj_begin, s.step),
                        stop if stop >= 0 else None,
                        idx.step // s.step,
                    )
                    obj_index_tuple += (obj_idx,)
            else:
                # Here mask (if it exists) and data chunk are already aligned
                start = s.start - idx_offset
                stop = s.stop - idx_offset
                idx_within, idx = _chunk_in_slice(data_begin, data_end, start, stop, s.step)
        elif isinstance(indices[axis], Number):
            # CASE: number
            idx = indices[axis] - idx_offset
            idx_within = (idx >= 0) & (idx < axis_len)
            obj_axis -= 1
        else:
            raise NotImplementedError()

        obj_axis += 1

        if not idx_within:
            break
        index_tuple += (idx,)

    if idx_within:
        if not obj_is_scalar:
            obj_ = obj_[_squeeze(obj_index_tuple)].new()
        if subassign and mask is not None:
            mask_ = mask_[_squeeze(obj_index_tuple)].new()
            index_tuple, obj_, mask_ = _uniquify(x_ndim, index_tuple, obj_, mask_)
        else:
            mask_ = None
            index_tuple, obj_, _ = _uniquify(x_ndim, index_tuple, obj_)
        return Fragmenter(x_ndim, index_tuple, mask_, obj_)
    else:
        return Fragmenter(x_ndim)


def _uniquify_merged(x, axis=None, keepdims=None, computing_meta=None):
    if x.index is None:
        return x
    x.index, x.obj, x.mask = _uniquify(x.ndim, x.index, x.obj, x.mask)
    return x


def _assign(
    old_data,
    mask,
    new_data,
    band_offset,
    band_axis,
    band_selection,
    subassign,
    mask_type,
    replace,
    accum,
):
    x = old_data.value

    mask = new_data.mask if subassign else mask
    mask = mask.value if isinstance(mask, InnerBaseType) else mask
    if mask is not None:
        mask = mask_type(mask)

    if new_data.index is None:
        if not subassign and replace:
            if band_selection:
                chunk_band = band_selection[band_axis] - band_offset[0]
                if 0 <= chunk_band and chunk_band < x.shape[band_axis]:
                    band_selection[band_axis] = chunk_band
                    band_selection = tuple(band_selection)
                    x(mask=mask, replace=replace)[band_selection] << x[band_selection].new()
            else:
                x(mask=mask, replace=replace) << x
        return wrap_inner(x)

    normalized_index = ()
    for i in new_data.index:
        if type(i) is slice and i.start >= 0 and i.stop < 0 and i.step < 0:
            i = slice(i.start, None, i.step)
        normalized_index += (i,)
    index = _squeeze(normalized_index)

    obj = new_data.obj

    if subassign:
        x[index](mask=mask, replace=replace, accum=accum) << obj
    else:
        x(mask=mask, replace=replace, accum=accum)[index] << obj

    return wrap_inner(x)


def _data_x_index_meshpoint_4extract(*args, mask_type, index_is_a_number, gb_dtype, gb_meta):
    """returns only that part of the data-chunk selected by the index"""
    x = args[0]
    indices = args[1 : x.ndim + 1]
    mask = args[x.ndim + 1]
    x_offsets = args[x.ndim + 2 :]
    index_tuple = ()
    mask_index_tuple = ()
    idx_within = True
    for axis in range(x.ndim):
        index = indices[axis]
        offset = x_offsets[axis][0]
        if type(index) is np.ndarray:
            if type(index[0]) is slice:
                # Note: slice is already aligned with mask if it exists
                s = index[0]
                idx_within, idx = _chunk_in_slice(
                    offset, offset + x.shape[axis], s.start, s.stop, s.step
                )
                if idx_within:
                    mask_begin = _ceildiv(idx.start - s.start, s.step)
                    mask_end = _ceildiv(idx.stop - s.start, s.step)
                    # beware of negative indices in slice specification!
                    stop = idx.stop - offset
                    idx = slice(idx.start - offset, stop if stop >= 0 else None, idx.step)
                    stop = _ceildiv(stop - s.start, s.step)
                    mask_idx = slice(mask_begin, mask_end, 1)
                else:
                    break
            else:
                idx = np.array(index) - offset
                idx_filter = (idx >= 0) & (idx < x.shape[axis])
                idx_within = np.any(idx_filter)
                if idx_within:
                    idx = idx[idx_filter]
                    mask_idx = np.argwhere(idx_filter)[:, 0]
                else:
                    break
        elif isinstance(index, Number):
            idx = index - offset
            idx_within = (idx >= 0) and (idx < x.shape[axis])
            idx = np.array([idx], dtype=np.int64)
            mask_idx = None
            if not idx_within:
                break
        else:
            raise NotImplementedError()

        index_tuple += (idx,)
        mask_index_tuple += (mask_idx,)
    x = x.value
    if idx_within:
        index_tuple = _squeeze(index_tuple)
        mask_index_tuple = _squeeze(mask_index_tuple)
        mask = mask_type(mask.value[mask_index_tuple].new()) if mask is not None else None
        return wrap_inner(x[index_tuple].new(gb_dtype, mask=mask))
    else:
        return wrap_inner(gb_meta.new(gb_dtype))


def _defrag_to_index_chunk(*args, x_chunks, cat_axes, dtype=None):
    """reassembles chunk-fragments, sorting them by the indices of the index chunk"""
    ndim = len(x_chunks)
    indices = args[0:ndim]
    fused_fragments = args[ndim].value
    index_tuple = ()
    for axis, idx in enumerate(indices):
        if axis not in cat_axes:
            index_tuple += (0,)
        else:
            if type(idx) is np.ndarray and type(idx[0]) is slice:
                s = idx[0]
                # this branch needs more efficient-handling: (e.g. we should avoid use of np.arange)
                idx = np.arange(s.start, s.stop, s.step, dtype=np.int64)

            # Needed when idx is unsigned
            idx = idx.astype(np.int64)

            # Normalize negative indices
            idx = np.where(idx < 0, idx + sum(x_chunks[axis]), idx)

            x_chunk_offset = 0
            chunk_output_offset = 0

            # Assemble the final index that picks from the output of the previous
            # kernel by adding together one layer per chunk of x
            idx_final = np.zeros_like(idx)
            for x_chunk in x_chunks[axis]:
                idx_filter = (idx >= x_chunk_offset) & (idx < x_chunk_offset + x_chunk)
                idx_cum = np.cumsum(idx_filter)
                idx_final += np.where(idx_filter, idx_cum - 1 + chunk_output_offset, 0)
                x_chunk_offset += x_chunk
                if idx_cum.size > 0:
                    chunk_output_offset += idx_cum[-1]

            index_tuple += (idx_final,)
    index_tuple = _squeeze(index_tuple)
    return wrap_inner(fused_fragments[index_tuple].new())


class AmbiguousAssignOrExtract:
    def __init__(self, parent, index):
        self.parent = parent
        self.index = index
        self._meta = parent._meta[index]

    def new(self, *, dtype=None, mask=None):
        if mask is not None:
            assert isinstance(mask, Mask)
            assert self.parent._meta.nvals == 0
            meta = self._meta.new(dtype=dtype, mask=mask._meta)
            delayed_mask = mask.mask._delayed
            grblas_mask_type = get_grblas_type(mask)
        else:
            meta = self._meta.new(dtype=dtype)
            delayed_mask = None
            grblas_mask_type = None

        ndim = len(self.parent.shape)
        dtype = np_dtype(meta.dtype)
        if ndim == 1 and type(self.index) is not tuple:
            indices = (self.index,)
        else:
            indices = self.index
        if ndim in [1, 2]:
            x = self.parent._delayed
            # prepare arguments for blockwise:
            indices_args = []
            offset_args = []
            cat_axes = []
            new_axes = ()
            for axis in range(ndim):
                indx = indices[axis]
                if type(indx) is da.Array:
                    indices_args += [indx, (ndim + axis,)]
                    new_axes += (ndim + axis,)
                    cat_axes += [axis]
                elif type(indx) in {list, np.ndarray}:
                    # convert list or numpy array to dask array
                    indx = np.array(indx)
                    name = "list_index-" + tokenize(indx, axis)
                    indx = da.core.from_array(indx, chunks="auto", name=name)
                    indices_args += [indx, (ndim + axis,)]
                    new_axes += (ndim + axis,)
                    cat_axes += [axis]
                elif type(indx) is slice:
                    # convert slice to dask array
                    start, stop, step = indx.indices(x.shape[axis])
                    indx = slice(start, stop, step)
                    if mask is None:
                        slice_len = _ceildiv(abs(stop - start), abs(step))
                        chunk_sizes = (slice_len,)
                        indx = np.array([indx])
                    else:
                        chunk_sizes = delayed_mask.chunks[axis]
                        chunk_offsets = np.roll(np.cumsum(chunk_sizes), 1)
                        chunk_offsets[0] = 0
                        starts = start + step * chunk_offsets
                        stops = start + step * (chunk_offsets + chunk_sizes)
                        indx = np.array(
                            [slice(start, stop, step) for start, stop in zip(starts, stops)]
                        )
                    name = "slice_index-" + tokenize(indx, axis)
                    indx = da.core.from_array(indx, chunks=1, name=name)
                    indx = da.Array(
                        indx.dask, indx.name, (chunk_sizes,), indx.dtype, meta=indx._meta
                    )
                    indices_args += [indx, (ndim + axis,)]
                    new_axes += (ndim + axis,)
                    cat_axes += [axis]
                else:
                    indices_args += [indx, None]

                offset = build_chunk_offsets_dask_array(x, axis, "offset-")
                offset_args += [offset, (axis,)]

            x_ind = tuple(range(ndim))
            fragments_ind = x_ind + new_axes
            mask_args = [delayed_mask, new_axes if mask is not None else None]
            index_is_a_number = [isinstance(i, Number) for i in indices]

            # this blockwise is essentially a cartesian product of data chunks and index chunks
            # both index and data chunks are fragmented in the process
            fragments = da.core.blockwise(
                _data_x_index_meshpoint_4extract,
                fragments_ind,
                x,
                x_ind,
                *indices_args,
                *mask_args,
                *offset_args,
                mask_type=grblas_mask_type,
                index_is_a_number=index_is_a_number,
                gb_dtype=dtype,
                gb_meta=meta,
                dtype=dtype,
                meta=wrap_inner(meta),
            )

            # this blockwise is essentially an aggregation over the data chunk axes
            extracts_ind = tuple((i + ndim) for i in x_ind if i in cat_axes)
            delayed = da.core.blockwise(
                _defrag_to_index_chunk,
                extracts_ind,
                *indices_args,
                fragments,
                fragments_ind,
                x_chunks=x.chunks,
                cat_axes=cat_axes,
                concatenate=True,
                dtype=dtype,
                meta=wrap_inner(meta),
            )

            return get_return_type(meta)(delayed)
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return Assigner(self.parent(*args, **kwargs), self.index, subassign=True)

    def update(self, obj):
        Assigner(Updater(self.parent), self.index).update(obj)

    def __lshift__(self, rhs):
        self.update(rhs)

    @property
    def value(self):
        self._meta.value
        return self.new().value


def _uniquify(ndim, index, obj, mask=None):
    # here we follow the SuiteSparse:GraphBLAS specification for
    # duplicate index entries: ignore all but the last unique entry
    def extract(obj, indices, axis):
        indices = fuse_index_pair(slice(None, None, -1), indices, obj.shape[axis])
        n = len(obj.shape)
        obj_index = [slice(None)] * n
        axis = 0 if n < ndim else axis
        obj_index[axis] = indices
        indices = _squeeze(tuple(obj_index))
        return obj[indices].new()

    index = index if type(index) is tuple else (index,)
    unique_indices_tuple = ()
    obj_axis = 0
    for axis in range(ndim):
        indx = index[axis]
        if type(indx) is not slice and not isinstance(indx, Number):
            if type(indx) is da.Array:
                reverse_indx = indx[::-1]
                unique_indx, obj_indx = da.routines.unique(reverse_indx, return_index=True)
            else:
                reverse_indx = np.array(indx)[::-1]
                unique_indx, obj_indx = np.unique(reverse_indx, return_index=True)
            if unique_indx.size < reverse_indx.size:
                indx = list(unique_indx)
                if (isinstance(obj, BaseType) or isinstance(obj, gb.base.BaseType)) and len(
                    obj.shape
                ) > 0:
                    obj = extract(obj, obj_indx, obj_axis)
                if mask is not None:
                    mask = extract(mask, obj_indx, obj_axis)
            obj_axis += 1
        unique_indices_tuple += (indx,)
    return unique_indices_tuple, obj, mask


def _identity_func(x, axis, keepdims):
    return x


class Assigner:
    def __init__(self, updater, index, subassign=False):
        self.updater = updater
        self.parent = updater.parent
        self.index = index
        self._meta = updater.parent._meta[index]
        self.subassign = subassign

    def update(self, obj):
        parent = self.parent
        mask = self.updater.mask
        replace = self.updater.replace
        subassign = self.subassign
        indices = self.index
        obj_is_scalar = not (
            isinstance(obj, BaseType) and type(obj._meta) in {gb.Vector, gb.Matrix}
        )
        if mask is not None:
            assert isinstance(mask, Mask)
            assert parent._meta.nvals == 0
            meta = self._meta.new(mask=mask._meta)
            delayed_mask = mask.mask._delayed
            grblas_mask_type = get_grblas_type(mask)
        else:
            meta = self._meta.new()
            delayed_mask = None
            grblas_mask_type = None

        ndim = len(parent.shape)
        dtype = np_dtype(meta.dtype)
        if ndim == 1 and type(self.index) is not tuple:
            indices = (self.index,)
        else:
            indices = self.index
        if not obj_is_scalar and _shape(parent, indices) != obj.shape:
            raise DimensionMismatch()
        if ndim in [1, 2]:
            # prepare arguments for blockwise:
            x = parent._optional_dup()
            x_ind = tuple(range(ndim))
            mask_ind = None
            # pre-align chunks where necessary:
            if subassign:
                if mask is not None and not obj_is_scalar:
                    # align mask and obj in order to fix obj_offsets
                    # before calculating them
                    ind = tuple(range(obj._delayed.ndim))
                    _, (obj._delayed, delayed_mask) = da.core.unify_chunks(
                        obj._delayed, ind, delayed_mask, ind
                    )
            elif mask is not None:
                # align `x` and its mask in order to fix offsets
                # of `x` before calculating them
                mask_ind = tuple(i for i in x_ind if not isinstance(indices[i], Number))
                _, (x, delayed_mask) = da.core.unify_chunks(x, x_ind, delayed_mask, mask_ind)

            indices_args = []
            x_ranges_args = []
            obj_offset_args = []
            obj_offset_axes = ()
            index_axes = ()
            # Note: the blockwise kwarg `new_axes` is non-empty only in one case:
            # where index is a slice and obj is a scalar and there is no mask
            new_axes = dict()
            obj_axis = 0
            for axis in range(ndim):
                indx = indices[axis]
                if type(indx) is da.Array:
                    indices_args += [indx, (ndim + axis,)]
                    index_axes += (ndim + axis,)
                elif type(indx) is slice:
                    s = indx
                    start, stop, step = s.indices(x.shape[axis])
                    indices_args += [slice(start, stop, step), None]
                    index_axes += (ndim + axis,)
                    if not obj_is_scalar:
                        obj_offset = build_chunk_offsets_dask_array(
                            obj._delayed, obj_axis, "obj_offset-"
                        )
                        obj_offset_args += [obj_offset, (ndim + axis,)]
                        obj_offset_axes += (axis,)
                    elif subassign and mask is not None:
                        obj_offset = build_chunk_offsets_dask_array(
                            delayed_mask, obj_axis, "obj_offset-"
                        )
                        obj_offset_args += [obj_offset, (ndim + axis,)]
                        obj_offset_axes += (axis,)
                    else:
                        new_axes[ndim + axis] = 1
                elif type(indx) in {list, np.ndarray}:
                    indx = np.array(indx)
                    name = "list_index-" + tokenize(indx, axis)
                    indx = da.core.from_array(indx, chunks="auto", name=name)
                    indices_args += [indx, (ndim + axis,)]
                    index_axes += (ndim + axis,)
                elif isinstance(indx, Number):
                    indices_args += [indx, None]
                    obj_axis -= 1
                else:
                    raise NotImplementedError()

                x_ranges = build_chunk_ranges_dask_array(x, axis, "x_ranges-")
                x_ranges_args += [x_ranges, (axis,)]
                obj_axis += 1

            if subassign:
                mask_args = [delayed_mask, (index_axes if mask is not None else None)]
            else:
                mask_args = [None, None]

            if not obj_is_scalar:
                obj_args = [obj._delayed, index_axes]
            elif isinstance(obj, BaseType):  # dask_grblas.Scalar
                obj_args = [obj._delayed, None]
            else:  # Number: int, float, etc.
                obj_args = [obj, None]

            if not obj_offset_args:
                obj_offset_args = [None, None]

            # this blockwise is essentially a cartesian product of data chunks and index chunks
            # for assign: index and obj chunks are fragmented in the process;
            # for subassign: index, obj, and mask chunks are fragmented in the process
            fragments_ind = x_ind + index_axes
            fragments = da.core.blockwise(
                _data_x_index_meshpoint_4assign,
                fragments_ind,
                *x_ranges_args,
                *indices_args,
                *mask_args,
                *obj_args,
                *obj_offset_args,
                x_ndim=ndim,
                subassign=subassign,
                obj_offset_axes=obj_offset_axes,
                new_axes=new_axes if new_axes else None,
                dtype=dtype,
                meta=wrap_inner(meta),
            )

            # gather all information for the assignment to each chunk of the old data.
            # (Alas, a for-loop is necessary as dask reduction with multiple axes at
            # the same time works in undesirable ways.)
            red_axes = tuple(k for k, m in enumerate(fragments_ind) if m not in x_ind)
            uniquifed = fragments
            for axis in red_axes[::-1]:
                if uniquifed.numblocks[axis] > 1:
                    aggregate_func = _uniquify_merged
                else:
                    aggregate_func = _identity_func
                uniquifed = da.reduction(
                    uniquifed,
                    _identity_func,
                    aggregate_func,
                    axis=axis,
                    dtype=dtype,
                    meta=wrap_inner(meta),
                )

            # perform the assignment for each chunk of the old data
            assign_ind = x_ind
            if subassign:
                mask_args = [None, None]
            else:
                mask_args = [delayed_mask, mask_ind]
            # for row or column assign:
            band_axis = [axis for axis, i in enumerate(indices) if isinstance(i, Number)]
            is_row_or_col_assign = len(band_axis) == 1
            if is_row_or_col_assign:
                band_selection = [i if isinstance(i, Number) else slice(None) for i in indices]
                band_axis = band_axis[0]
                band_offset_args = x_ranges_args[2 * band_axis : 2 * band_axis + 2]
            else:
                band_selection = None
                band_axis = None
                band_offset_args = [None, None]

            delayed = da.core.blockwise(
                *(_assign, assign_ind),
                *(x, x_ind),
                *mask_args,
                *(uniquifed, x_ind),
                *band_offset_args,
                band_axis=band_axis,
                band_selection=band_selection,
                subassign=subassign,
                mask_type=grblas_mask_type,
                replace=replace,
                accum=self.updater.accum,
                dtype=dtype,
                meta=wrap_inner(parent._meta),
            )
            parent._delayed = delayed
            return
        raise NotImplementedError()

    def __lshift__(self, obj):
        self.update(obj)


class FakeInnerTensor(InnerBaseType):
    # Class to help in efficient dask computation of mxv, vxm and mxm methods.

    def __init__(self, value, compress_axis):
        assert type(value) in {gb.Matrix, gb.Vector}
        self.dtype = np_dtype(value.dtype)
        self.value = value
        self.shape = value.shape[:compress_axis] + (1,) + value.shape[compress_axis:]
        self.ndim = len(value.shape) + 1
        self.compress_axis = compress_axis


def _expr_new(method_name, dtype, grblas_mask_type, kwargs, x, mask, *args):
    # expr.new(...)
    args = [x.value if isinstance(x, InnerBaseType) else x for x in args]
    kwargs = {
        key: (kwargs[key].value if isinstance(kwargs[key], InnerBaseType) else kwargs[key])
        for key in kwargs
    }
    expr = getattr(x.value, method_name)(*args, **kwargs)
    if mask is not None:
        mask = grblas_mask_type(mask.value)
    return wrap_inner(expr.new(dtype=dtype, mask=mask))


# This mutates the value in `updating`
def _update_expr(method_name, updating, x, kwargs, *args):
    # v << left.ewise_mult(right)
    args = [x.value if isinstance(x, InnerBaseType) else x for x in args]
    kwargs = {
        key: (kwargs[key].value if isinstance(kwargs[key], InnerBaseType) else kwargs[key])
        for key in kwargs
    }
    expr = getattr(x.value, method_name)(*args, **kwargs)
    updating.value << expr
    return updating


# This mutates the value in `updating`
def _update_expr_full(method_name, updating, accum, mask, mask_type, replace, x, kwargs, *args):
    # v(mask=mask) << left.ewise_mult(right)
    args = [x.value if isinstance(x, InnerBaseType) else x for x in args]
    kwargs = {
        key: (kwargs[key].value if isinstance(kwargs[key], InnerBaseType) else kwargs[key])
        for key in kwargs
    }
    expr = getattr(x.value, method_name)(*args, **kwargs)
    if mask is not None:
        mask = mask_type(mask.value)
    updating.value(accum=accum, mask=mask, replace=replace) << expr
    return updating


def _reduce_axis(op, gb_dtype, x, axis=None, keepdims=None, computing_meta=None, dtype=None):
    """Call reduce_rowwise or reduce_columnwise on each chunk"""
    if computing_meta:
        return np.empty(0, dtype=dtype)
    if axis == (1,):
        return wrap_inner(x.value.reduce_rowwise(op).new(dtype=gb_dtype))
    if axis == (0,):
        return wrap_inner(x.value.reduce_columnwise(op).new(dtype=gb_dtype))


def _reduce_axis_combine(op, x, axis=None, keepdims=None, computing_meta=None, dtype=None):
    """Combine results from _reduce_axis on each chunk"""
    if computing_meta:
        return np.empty(0, dtype=dtype)
    if type(x) is list:
        vals = [val.value for val in x]
        return wrap_inner(reduce(lambda x, y: x.ewise_add(y, op).new(), vals))
    return x


def _reduce_scalar(op, gb_dtype, x, axis=None, keepdims=None, computing_meta=None, dtype=None):
    """Call reduce_scalar on each chunk"""
    if computing_meta:
        return np.empty(0, dtype=dtype)
    return wrap_inner(x.value.reduce_scalar(op).new(dtype=gb_dtype))


def _reduce(op, gb_dtype, x, axis=None, keepdims=None, computing_meta=None, dtype=None):
    """Call reduce on each chunk"""
    if computing_meta:
        return np.empty(0, dtype=dtype)
    return wrap_inner(x.value.reduce(op).new(dtype=gb_dtype))


def _reduce_combine(op, x, axis=None, keepdims=None, computing_meta=None, dtype=None):
    """Combine results from reduce or reduce_scalar on each chunk"""
    if computing_meta:
        return np.empty(0, dtype=dtype)
    if type(x) is list:
        # do we need `gb_dtype` instead of `np_dtype` below?
        if type(x[0]) is list:
            vals = [val.value.value for sublist in x for val in sublist]
        else:
            vals = [val.value.value for val in x]
        values = gb.Vector.from_values(list(range(len(vals))), vals, size=len(vals), dtype=dtype)
        return wrap_inner(values.reduce(op).new())
    return x


def _reduce_accum(output, reduced, accum):
    """Accumulate the results of reduce with a scalar"""
    # This is pretty ugly.  If only we could call binary operators on scalars...
    dtype = output.value.dtype
    if output.value.is_empty:
        left = gb.Vector.new(dtype, 1)
    else:
        left = gb.Vector.from_values([0], [output.value.value], dtype=dtype)
    if reduced.value.is_empty:
        right = gb.Vector.new(reduced.value.dtype, 1)
    else:
        right = gb.Vector.from_values([0], [reduced.value.value], dtype=reduced.value.dtype)
    result = left.ewise_add(right, op=accum, require_monoid=False).new(dtype=dtype)
    result = result[0].new()
    return wrap_inner(result)


def _reduce_axis_accum(output, reduced, accum):
    """Accumulate the results of reduce_axis with a vector"""
    if isinstance(reduced, np.ndarray) and (reduced.size == 0):
        return wrap_inner(gb.Vector.new())
    dtype = output.value.dtype
    if output.value.shape == 0:
        left = gb.Vector.new(dtype, 1)
    else:
        left = output.value
    if reduced.value.shape == 0:
        right = gb.Vector.new(reduced.value.dtype, 1)
    else:
        right = reduced.value
    result = left.ewise_add(right, op=accum, require_monoid=False).new(dtype=dtype)
    return wrap_inner(result)


def _transpose_if(x, xt):
    if xt:
        return x.value.T
    return x.value


def _matmul(op, at, bt, dtype, no_mask, mask_type, *args, computing_meta=None):
    if computing_meta:
        return np.empty(0, dtype=dtype)

    if no_mask:
        a_blocks, b_blocks = args
        mask = None
    else:
        mask, a_blocks, b_blocks = args
        mask = mask_type(mask.value)

    vals = [
        op(_transpose_if(a, at) @ _transpose_if(b, bt)).new(mask=mask, dtype=dtype)
        for a, b in zip(a_blocks, b_blocks)
    ]
    gb_obj = reduce(lambda x, y: x.ewise_add(y, op.monoid).new(), vals)

    return wrap_inner(gb_obj)


def _matmul2(op, dtype, at, bt, a, b, computing_meta=None):
    left = _transpose_if(a, at)
    right = _transpose_if(b, bt)
    return op(left @ right).new(dtype=dtype)


def _matmul2_masked(op, dtype, at, bt, mask_type, mask, a, b, computing_meta=None):
    left = _transpose_if(a, at)
    right = _transpose_if(b, bt)
    mask = mask_type(mask.value)
    return op(left @ right).new(dtype=dtype, mask=mask)


def _sum_by_monoid(monoid, a, axis=None, keepdims=None):
    if type(a) is not list:
        out = a
    else:
        out = reduce(lambda x, y: x.ewise_add(y, monoid).new(), a)
    if not keepdims:
        out = wrap_inner(out)
    return out


def sum_by_monoid(
    monoid,
    a,
    axis=None,
    dtype=None,
    keepdims=False,
    split_every=None,
    out=None,
    meta=None,
):
    if dtype is None:
        dtype = getattr(np.zeros(1, dtype=a.dtype).sum(), "dtype", object)
    result = da.reduction(
        a,
        partial(_sum_by_monoid, monoid),
        partial(_sum_by_monoid, monoid),
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
        split_every=split_every,
        out=out,
        meta=meta,
        concatenate=False,
    )
    return result


@da.core.concatenate_lookup.register(Fragmenter)
def _concat_fragmenter(seq, axis=0):
    """
    Concatenates Fragmenter objects in the sequence `seq` along the axis `axis`.
    Fragmenter objects must have the same value for attribute `.ndim` which is the
    number of dimensions of the array-like object being indexed by the index-tuple
    stored in the `.index` attribute of the Fragmenter.  Hence, the attribute
    `.index` should be a tuple `t` of length `.ndim`, each of whose elements can
    be a Number, array, or slice representing an index.  `t[i]` is the index along
    axis `i`.  However, `axis - .ndim` represents the
    axis along the array-like object obtained AFTER applying the index `.index`
    on an array with number of dimensions `.ndim`.
    Fragmenter attributes of value `None` are ignored.
    Fragmenter attributes (`.obj` and `.mask`) of type `Vector` or `Matrix`
    should be able to be concatenated along the given axis according to the usual
    rules otherwise an error will be raised.
    If the attribute `.obj` of any Fragmenter in `seq` is of scalar value then it
    should be the same over all Fragmenter objects, and the final result will
    also have the same value for this attribute.
    """

    def concatenate_fragments(frag1, frag2, axis=0, base_axis=0):
        out = Fragmenter(frag1.ndim)
        out_index = list(frag1.index)
        if type(frag1.index[base_axis]) is slice:
            s1, s2 = frag1.index[base_axis], frag2.index[base_axis]
            if type(s1) is not type(s2):
                raise TypeError(
                    f"Can only concatenate contiguous slices of the same step.  "
                    f"Got a slice and something else: {type(s1)} and {type(s2)}."
                )
            if s1.step != s2.step:
                raise ValueError(
                    f"Can only concatenate contiguous slices of the same step.  "
                    f"Got unequal steps: {s1.step} and {s2.step}."
                )

            s1_last = range(s1.start, s1.stop, s1.step)[-1]
            s1_next = s1_last + s1.step
            if s1_next != s2.start:
                raise ValueError(
                    f"Can only concatenate contiguous slices of the same step.  "
                    f"Got non-contiguous slices: first slice (with step {s1.step})"
                    f" stops at {s1_last} while the second starts from {s2.start}."
                )
            out_index[base_axis] = slice(s1.start, s2.stop, s1.step)
        else:
            out_index[base_axis] = np.concatenate([frag1.index[base_axis], frag2.index[base_axis]])
        out.index = tuple(out_index)

        obj = frag1.obj
        if isinstance(obj, gb.base.BaseType) and type(obj) in {gb.Vector, gb.Matrix}:
            concat = da.core.concatenate_lookup.dispatch(type(wrap_inner(obj)))
            obj = concat([wrap_inner(frag1.obj), wrap_inner(frag2.obj)], axis=axis).value
        out.obj = obj

        mask = frag1.mask
        if mask is not None:
            concat = da.core.concatenate_lookup.dispatch(type(wrap_inner(mask)))
            mask = concat([wrap_inner(frag1.mask), wrap_inner(frag2.mask)], axis=axis).value
        out.mask = mask

        return out

    # seq is assumed to be a non-empty list
    ndim = seq[0].ndim
    axis -= ndim
    seq_ = [fragment for fragment in seq if fragment.index]
    if seq_:
        if axis not in set(range(ndim)):
            raise ValueError(
                f"Can only concatenate for axis with values in {set(range(ndim))}.  " f"Got {axis}."
            )

        non_singleton_axes = [
            axis for axis, index in enumerate(seq_[0].index) if not isinstance(index, Number)
        ]
        base_axis = non_singleton_axes[axis]
        return reduce(partial(concatenate_fragments, axis=axis, base_axis=base_axis), seq_)
    else:
        return seq[0]
