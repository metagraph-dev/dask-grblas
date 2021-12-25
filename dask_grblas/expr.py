from functools import partial, reduce

import dask.array as da
import grblas as gb
import numpy as np

from dask.distributed import Lock

from .base import BaseType, InnerBaseType
from .mask import Mask
from .utils import get_grblas_type, get_meta, get_return_type, np_dtype, wrap_inner


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
            updating(mask=mask, accum=accum, replace=replace) << get_return_type(meta)(
                delayed
            )
            return
        else:
            raise ValueError(self.method_name)
        updating._delayed = delayed

    @property
    def value(self):
        self._meta.value
        return self.new().value


class Updater:
    def __init__(self, parent, *, mask=None, accum=None, replace=False):
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
            self.parent._update(
                delayed, accum=self.accum, mask=self.mask, replace=self.replace
            )


def _resolve_indices(grblas_obj, indices):
    out = ()
    ndim = len(grblas_obj.shape)
    for axis in range(ndim):
        resolved_indices = np.arange(grblas_obj.shape[axis])
        for index in indices[::-1]:
            if type(index) is tuple:
                ind = index[axis]
                resolved_indices = resolved_indices[ind]
        out += (resolved_indices,)
    return out if ndim > 1 else out[0]


def _expand_mask_to_fit(value, mask_loc, mask, mask_type):
    grblas_type = get_grblas_type(value)
    expanded_mask = grblas_type.new(dtype=mask.mask.dtype, size=value.size)
    expanded_mask[mask_loc] << mask.mask
    return mask_type(expanded_mask)


def _extractor_new(x, dtype, mask, mask_type):
    indices = []
    inner = x
    while inner.index is not None:
        indices.append(inner.index)
        inner = inner.inner
    inner = inner.inner
    if mask is not None:
        mask = mask_type(mask.value)
    if len(indices) == 0:
        # Is there some way we can avoid this dup here?
        # This likely comes from a slice such as v[:] or v[:10]
        # Ideally, we would use `_optional_dup` in the DAG
        value = inner.value.dup(dtype=dtype, mask=mask)
    elif len(indices) > 0:
        value = inner.value
        resolved_indices = _resolve_indices(value, indices)
        value = value[resolved_indices].new(dtype=dtype, mask=mask)
    else:
        raise NotImplementedError(f"indices: {indices}")
    return wrap_inner(value)


def _assigner_update(x, dtype, mask_type, accum, obj, subassign, *args):
    # `inner` is of Extractor type
    indices = []
    inner = x
    while inner.level > 0:
        indices.append(inner.index)
        inner = inner.inner
    # Extractor level 0 has the full InnerVector chunk and mask (if given):
    mask = inner.mask
    inner = inner.inner
    chunk = inner.value
    resolved_indices = _resolve_indices(chunk, indices)
    if isinstance(obj, InnerBaseType):
        obj_value = obj.value
    else:
        obj_value = obj
    if subassign:
        mask, replace = args
        if mask is not None:
            mask = mask_type(mask.value)
        chunk[resolved_indices](mask=mask, accum=accum, replace=replace) << obj_value
    else:
        if mask is not None:
            mask = mask_type(mask.value)
        chunk(mask=mask, accum=accum)[resolved_indices] << obj_value
    if type(chunk) == gb.Vector:
        return wrap_inner(gb.Vector.new(dtype, size=resolved_indices.size))
    else:
        nrows = resolved_indices[0].size
        ncols = resolved_indices[1].size
        return wrap_inner(gb.Matrix.new(dtype, nrows=nrows, ncols=ncols))


def _get_expanded_mask_indices(mask, mask_range):
    def complement_if(mask_indices, mask):
        if mask.complement:
            return list(set(range(mask.mask.size)) - set(mask_indices))
        return mask_indices

    structure, values = mask.mask.to_values()
    if mask.value:
        mask_indices = [s for s, v in zip(structure, values) if v]
    elif mask.structure:
        mask_indices = structure
    else:
        TypeError("Mask must be a ValueMask or StructuralMask, " "complemented or not.")
    mask_indices = complement_if(mask_indices, mask)
    return list(set(mask_range).intersection(set(mask_indices)))


class Extractor:
    """
    This stores the extraction history of an InnerVector.
    """

    def __init__(self, inner, mask=None, mask_type=None, replace=None, index=None):
        self.inner = inner
        self.mask = mask
        self.index = index
        self.dtype = inner.dtype
        self.ndim = inner.ndim
        if type(inner) is Extractor:
            self.level = inner.level + 1
        else:
            self.level = 0
        if replace and (type(mask) is not np.ndarray):
            # This branch is necessary when replace is True:
            # Here, we remove all chunk elements that lie in
            # the complement of the mask
            chunk = inner.value
            chunk_copy = chunk.dup()
            mask = mask_type(mask.value)
            chunk(mask=mask, replace=True) << chunk_copy

    def __getitem__(self, index):
        return Extractor(self, index=index)


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

        # indices = gb.base.IndexerResolver(self.parent._meta, self.index, raw=False).indices
        ndim = len(self.parent.shape)
        if ndim == 1 and type(self.index) is not tuple:
            indices = (self.index,)
        else:
            indices = self.index
        if ndim in [1, 2]:
            delayed = self.parent._delayed.map_blocks(
                Extractor,
                dtype=np_dtype(meta.dtype),
            )
            if ndim == 1:
                delayed = delayed[indices]
            else:
                delayed = delayed[indices[0], :][:, indices[1]]
            delayed = da.core.elemwise(
                _extractor_new,
                delayed,
                dtype,
                delayed_mask,
                grblas_mask_type,
                dtype=np_dtype(meta.dtype),
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


def _get_Extractor_meta(w):
    while type(w) is Extractor:
        w = w.inner
    return w.value


def _clear_mask(a, computing_meta=None, axis=None, keepdims=None):
    if computing_meta:
        return np.array()
    if type(a) is list:
        if type(a[0]) is list:
            meta = _get_Extractor_meta(a[0][0])
        else:
            meta = _get_Extractor_meta(a[0])
    else:
        meta = _get_Extractor_meta(a)

    dtype = np_dtype(meta.dtype)
    if len(meta.shape) == 1:
        o = type(meta).new(dtype=dtype, size=1)
    else:
        o = type(meta).new(dtype=dtype, nrows=1, ncols=1)

    if keepdims:
        return wrap_inner(o)
    else:
        return wrap_inner(gb.Scalar.from_value(0, dtype=dtype))


def _clear_mask_final_step(a, axis=None, keepdims=None):
    if type(a) is list:
        if type(a[0]) is list:
            meta = _get_Extractor_meta(a[0][0])
        else:
            meta = _get_Extractor_meta(a[0])
    else:
        meta = _get_Extractor_meta(a)

    dtype = np_dtype(meta.dtype)
    if len(meta.shape) == 1:
        o = type(meta).new(dtype=dtype, size=1)
    else:
        o = type(meta).new(dtype=dtype, nrows=1, ncols=1)

    if keepdims:
        return wrap_inner(o)
    else:
        return wrap_inner(gb.Scalar.from_value(0, dtype=dtype))


def _uniquify(index, obj, mask=None):
    # here we follow the SuiteSparse:GraphBLAS specification for
    # duplicate index entries: ignore all but the last unique entry
    rev_index = np.array(index)[::-1]
    unique_indices, obj_indices = np.unique(rev_index, return_index=True)
    if unique_indices.size < rev_index.size:
        if isinstance(obj, BaseType):
            obj = obj[::-1].new()
            obj = obj[obj_indices].new()
        if mask:
            mask = mask[::-1].new()
            mask = mask[obj_indices].new()
        index = unique_indices
    return index, obj, mask


def _uniquify2D(index, obj, mask=None):
    # here we follow the SuiteSparse:GraphBLAS specification for
    # duplicate index entries: ignore all but the last unique entry
    is_not_unique = False
    unique_indices_tup = ()
    obj_indices_tup = ()
    pm = ()
    for axis in range(2):
        if type(index[axis]) is slice:
            unique_indices_tup += (index[axis],)
            obj_indices_tup += (slice(None),)
            pm += (1,)
            continue
        rev_index = np.array(index[axis])[::-1]
        unique_indices, obj_indices = np.unique(
            rev_index, return_index=True)
        unique_indices_tup += (unique_indices,)
        obj_indices_tup += (obj_indices,)
        pm += (-1,)
        if unique_indices.size < rev_index.size:
            is_not_unique = True

    if is_not_unique:
        if isinstance(obj, BaseType):
            obj = obj[::pm[0], ::pm[1]].new()
            obj = obj[obj_indices_tup].new()
        if mask:
            mask = mask[::pm[0], ::pm[1]].new()
            mask = mask[obj_indices_tup].new()
        index = unique_indices_tup
    return index, obj, mask


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

        dtype = np_dtype(meta.dtype)
        ndim = len(self.parent.shape)
        if ndim == 1 and type(self.index) is not tuple:
            if type(self.index) in {list, np.ndarray}:
                if subassign and mask:
                    self.index, obj, submask = _uniquify(self.index, obj, mask=mask.mask)
                    mask.mask = submask
                    delayed_mask = submask._delayed
                else:
                    self.index, obj, _ = _uniquify(self.index, obj)
            indices = (self.index,)
        else:
            if subassign and mask:
                self.index, obj, submask = _uniquify2D(
                    self.index, obj, mask=mask.mask)
                mask.mask = submask
                delayed_mask = submask._delayed
            else:
                self.index, obj, _ = _uniquify2D(self.index, obj)
            indices = self.index
        if ndim in [1, 2]:
            delayed_dup = parent._optional_dup()
            ind = "i" if ndim == 1 else "ij"
            if (not subassign) and mask:
                # We use blockwise here to enable chunk alignment between
                # `delayed_mask` and `delayed_dup`
                args = [delayed_dup, ind, delayed_mask, ind]
                delayed_dup = da.core.blockwise(
                    lambda x, y: x,
                    ind,
                    *args,
                    dtype=dtype,
                )
                # Extractor level 0:
                # clear data chunks using corresponding mask
                delayed = da.core.blockwise(
                    Extractor,
                    ind,
                    *args,
                    dtype=dtype,
                    replace=replace,
                    mask_type=grblas_mask_type,
                )
                if replace:
                    # when replace=True we use this branch to touch ALL
                    # chunks to ensure all data chunks are cleared
                    replaced = da.reduction(
                        delayed,
                        _clear_mask,
                        _clear_mask_final_step,
                        dtype=dtype,
                        concatenate=False,
                    )
                    clear_mask = get_return_type(gb.Scalar.new(dtype))(replaced)
            else:
                args = [delayed_dup, ind]
                delayed = da.core.blockwise(
                    Extractor,
                    ind,
                    *args,
                    dtype=dtype,
                    replace=False,
                    mask_type=grblas_mask_type,
                )
            
            if ndim == 1:
                delayed = delayed[indices]
            else:
                delayed = delayed[indices[0], :][:, indices[1]]
            # the size/shape of `delayed` is now in general
            # different from `delayed_dup` by virtue of `indices`.
            # The following updates the indexed elements of
            # `delayed_dup` but returns a dask.array of zeros
            # since it is difficult to return the updated
            # `delayed_dup` directly:
            args = []
            if subassign:
                args = [delayed_mask, replace]
            delayed = da.core.elemwise(
                _assigner_update,
                delayed,
                dtype,
                grblas_mask_type,
                self.updater.accum,
                obj._delayed if isinstance(obj, BaseType) else obj,
                subassign,
                *args,
                dtype=dtype,
            )
            parent._delayed = delayed_dup
            # The following last few lines are only meant to trigger the
            # above updates whenever parent.compute() is called
            if (not subassign) and mask and replace:
                parent << parent.apply(
                    gb.binary.plus, right=clear_mask
                )
            if ndim == 1:
                zero = get_return_type(meta)(delayed).reduce().new()
            else:
                zero = get_return_type(meta)(delayed).reduce_scalar().new()
            parent << parent.apply(gb.binary.plus, right=zero)
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
