import dask.array as da
import grblas as gb
import numpy as np
from functools import partial, reduce
from .base import BaseType, InnerBaseType
from .mask import Mask
from .utils import np_dtype, get_meta, get_return_type, get_grblas_type, get_inner_type, wrap_inner


class GbDelayed:
    def __init__(self, parent, method_name, *args, meta, **kwargs):
        self.parent = parent
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
        self._meta = meta

    def _matmult(self, meta, updating=None, mask=None, accum=None):
        a = self.parent._delayed
        b = self.args[0]._delayed
        op = self.args[1]
        lhs_ind = 'ij' if (a.ndim == 2) else 'j'
        rhs_ind = 'jk' if (b.ndim == 2) else 'j'
        if lhs_ind == 'ij':
            out_ind = 'ik' if (rhs_ind == 'jk') else 'i'
        else:
            out_ind = 'k' if (rhs_ind == 'jk') else ''

        not_updating = updating is None
        no_mask = mask is None
        grblas_mask_type = None
        if not_updating:
            args = [a, lhs_ind, b, rhs_ind]
        elif no_mask:
            args = [updating, out_ind,
                    a, lhs_ind, b, rhs_ind]
        else:
            grblas_mask_type = get_grblas_type(mask)
            args = [updating, out_ind,
                    mask.mask._delayed, out_ind,
                    a, lhs_ind, b, rhs_ind]

        return da.core.blockwise(
            partial(_matmul, op, meta.dtype,
                    not_updating,
                    no_mask, grblas_mask_type,
                    accum),
            out_ind,
            *args,
            meta=wrap_inner(meta)
        )

    def _matmul2(self, meta, updating=None, mask=None, accum=None):
        a = self.parent._delayed
        b = self.args[0]._delayed
        op = self.args[1]
        a_is_1d = False
        if a.ndim == 1:
            a_is_1d = True
            a = a.map_blocks(asOneRowMatrix, new_axis=0, meta=asOneRowMatrix(wrap_inner(meta)))
            sum_meta = asOneRowMatrix(wrap_inner(meta))
            out_meta = FakeInnerTensor(sum_meta.value, face=(0, 2))

        b_is_1d = False
        if b.ndim == 1:
            b_is_1d = True
            b = b.map_blocks(asOneColMatrix, new_axis=1, meta=asOneColMatrix(wrap_inner(meta)))
            sum_meta = asOneColMatrix(wrap_inner(meta))
            out_meta = FakeInnerTensor(sum_meta.value, face=(0, 2))

        # out_ind includes all dimensions to prevent contraction
        # in the blockwise below
        out_ind = (0, 1, 2)
        # lhs_ind includes `a`/LHS dimensions
        lhs_ind = (0, 1)
        # on `b`/RHS -2 dimension is "contracted" with the last dimension
        # of `a`, last dimension of `b` is `b` specific
        rhs_ind = (1, 2)
    
        out = da.core.blockwise(
            partial(_matmul2, op, meta.dtype),
            out_ind,
            a,
            lhs_ind,
            b,
            rhs_ind,
            adjust_chunks={lhs_ind[-1]: 1},
            dtype=np.result_type(a, b),
            concatenate=False,
            meta=out_meta
        )
        
        # out is 3D (a slab or a bar)
        out = sum_by_monoid(op.monoid, out, axis=-2, meta=sum_meta) # 2D
        if a_is_1d:
            out = out[..., 0, :]    # 1D
        if b_is_1d:
            out = out[..., 0]   #1D
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
            axis=axis
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

        if self.method_name == 'reduce':
            delayed = self._reduce(meta.dtype)
        elif self.method_name == 'reduce_scalar':
            delayed = self._reduce_scalar(meta.dtype)
        elif self.method_name == 'reduce_rowwise':
            delayed = self._reduce_along_axis(1, meta.dtype)
        elif self.method_name == 'reduce_columnwise':
            delayed = self._reduce_along_axis(0, meta.dtype)
        elif self.method_name in {'apply', 'ewise_add', 'ewise_mult'}:
            delayed = da.core.elemwise(
                _expr_new,
                self.method_name,
                dtype,
                grblas_mask_type,
                self.kwargs,
                self.parent._delayed,
                delayed_mask,
                *[x._delayed if isinstance(x, BaseType) else x for x in self.args],
                dtype=np_dtype(meta.dtype),
            )
        elif self.method_name in {'vxm', 'mxv'}:
            delayed = self._matmul2(meta)
        else:
            raise ValueError(self.method_name)
        return get_return_type(meta)(delayed)

    def _update(self, updating, *, mask=None, accum=None, replace=None):
        updating._meta.update(self._meta)
        assert updating._meta._is_scalar or updating._meta.nvals == 0
        meta = updating._meta

        if self.method_name == 'reduce':
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
        elif self.method_name == 'reduce_scalar':
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
        elif self.method_name == 'reduce_rowwise':
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
        elif self.method_name == 'reduce_columnwise':
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
        elif self.method_name in {'apply', 'ewise_add', 'ewise_mult'}:
            delayed = updating._optional_dup()
            if mask is None and accum is None:
                delayed = da.core.elemwise(
                    _update_expr,
                    self.method_name,
                    delayed,
                    self.parent._delayed,
                    self.kwargs,
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
                    self.kwargs,
                    *[x._delayed if isinstance(x, BaseType) else x for x in self.args],
                    dtype=np_dtype(meta.dtype),
                )
        elif self.method_name in {'vxm', 'mxv'}:
            delayed = self._matmul2(meta)
            updating(mask=mask, accum=accum) << get_return_type(meta)(delayed)
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
        self._meta[keys]
        return AmbiguousAssignOrExtract(self, keys)

    def __setitem__(self, keys, obj):
        raise NotImplementedError()
        self._meta[keys] = obj
        self.parent._meta.clear()  # XXX: test to see if this is necessary

    def __lshift__(self, delayed):
        # Occurs when user calls C(params) << delayed
        self.update(delayed)

    def update(self, delayed):
        # Occurs when user calls C(params).update(delayed)
        # self.parent._update(delayed, **self.kwargs)
        if self.mask is None and self.accum is None:
            return self.parent.update(delayed)
        self.parent._meta._update(get_meta(delayed), mask=get_meta(self.mask), accum=self.accum, replace=self.replace)
        if self.parent._meta._is_scalar:
            self.parent._update(delayed, accum=self.accum)
        else:
            self.parent._update(delayed, accum=self.accum, mask=self.mask, replace=self.replace)


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
        # This likely comes from a slice sich as v[:] or v[:10]
        # Ideally, we would use `_optional_dup` in the DAG
        value = inner.value.dup(dtype=dtype, mask=mask)
    elif len(indices) == 1:
        index = indices[0]
        if type(index) is tuple and len(index) == 1:
            index = index[0]
        value = inner.value[index].new(dtype=dtype, mask=mask)
    else:
        raise NotImplementedError(f'indices: {indices}')
    return wrap_inner(value)


class Extractor:
    def __init__(self, inner, index=None):
        self.inner = inner
        self.index = index
        self.dtype = inner.dtype
        self.ndim = inner.dtype

    def __getitem__(self, index):
        return Extractor(self, index)


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
        if len(self.parent.shape) == 1 and type(self.index) is not tuple:
            indices = (self.index,)
        else:
            indices = self.index
        if len(indices) == 1 or len(indices) == 2:
            delayed = self.parent._delayed.map_blocks(
                Extractor,
                dtype=np_dtype(meta.dtype),
            )
            delayed = delayed[indices]
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
        updater = self.parent(*args, **kwargs)
        return updater[self.index]

    def update(self, obj):
        self.parent[self.index] = obj

    def __lshift__(self, obj):
        self.update(obj)

    @property
    def value(self):
        self._meta.value
        return self.new().value


class FakeInnerTensor(InnerBaseType):
    # Class to help in efficient dask computation of mxv, vxm
    # and mxm methods.
    # The underlying data-structure of FakeInnerTensor is always
    # a grblas matrix whose two dimensions coincide with two of
    # the dimensions of FakeInnerTensor.  These two dimensions
    # are stored as a tuple in the attribute 'face'. face[0] is
    # always 0.  The rest of the dimensions of FakeInnerTensor
    # have a maximum length of 1 each.  The total number ndim of
    # dimensions of FakeInnerTensor is always greater than 2.

    def __init__(self, grblas_matrix, shape=None, face=(0, 1)):
        assert type(grblas_matrix) is gb.Matrix
        self.value = grblas_matrix
        self.face = face
        if shape is None:
            shape = [1]*3
            shape[self.face[0]] = self.value.shape[0]
            shape[self.face[1]] = self.value.shape[1]
            self.shape = tuple(shape)
            self.ndim = 3
        else:
            self.shape = shape
            self.ndim = len(shape) 
        self.dtype = np_dtype(grblas_matrix.dtype)

    def __getitem__(self, index):
        # Fix index, handling ellipsis and incomplete slices.
        if not isinstance(index, tuple):
            index = (index,)
        fixed = []
        new_axes = [k for k, i in enumerate(index) if i is None]
        n_newdims = len(new_axes)
        length, dims = len(index), self.ndim + n_newdims
        for slice_ in index:
            if slice_ is Ellipsis:
                fixed.extend([slice(None)]*(dims - length + 1))
                length = len(fixed)
            elif isinstance(slice_, int):
                fixed.append(slice(slice_, slice_ + 1, 1))
            else:
                fixed.append(slice_)
        index = tuple(fixed)
        if len(index) < dims:
            index += (slice(None),) * (dims - len(index))
        if n_newdims > 0:
            # temporarily remove the new axes
            new_axes = [k for k, i in enumerate(index) if i is None]
            index = list(index)
            for k in new_axes[::-1]:
                index.pop(k)
            index = tuple(index)
        # Take the slice.  This always copies!
        assert (type(index) is tuple) and (len(index) == self.ndim)
        i, j = index[self.face[0]], index[self.face[1]]
        value = self.value[i, j].new()
        shape = [0 if k in self.face else len([object][x])
                 for k, x in enumerate(index)]
        shape[self.face[0]] = value.shape[0]
        shape[self.face[1]] = value.shape[1]
        out = FakeInnerTensor(value, shape=tuple(shape), face=self.face)
        if n_newdims == 0:
            return out
        else:
            # restore the new axes that were previously removed
            shape = list(out.shape)
            face = out.face[1]
            for k in new_axes:
                shape.insert(k, 1)
                if k <= face:
                    face += 1
            return out.reshape(tuple(shape), face=(0, face))
    
    def reshape(self, *args, face=None):
        # New shape is assumed to preserve shape of underlying grblas matrix
        shape, = args
        prod = lambda x, y: x*y
        volume = reduce(prod, self.value.shape)
        no_len_axes = [i for i, sz in enumerate(shape) if sz == -1]
        if len(no_len_axes) == 1:
            shape[no_len_axes[0]] = volume//(-reduce(prod, shape))
        new_volume = reduce(prod, shape)
        if volume == new_volume:
            new_ndim = len(shape)
            if new_ndim == 2:
                if shape == self.value.shape:
                    return wrap_inner(self.value)
            elif new_ndim > 2:
                if face is None:
                    face = [k for k, x in enumerate(shape)
                            if (x > 1) and (k > 0)]
                    n = len(face)
                    if n == 0:
                        face = self.face
                    elif n == 1:
                        face = (self.face[0], face[0])
                return FakeInnerTensor(self.value, shape=shape, face=face)
        raise NotImplementedError()


# Dask task functions


def _expr_new(method_name, dtype, grblas_mask_type, kwargs, x, mask, *args):
    # expr.new(...)
    args = [x.value if isinstance(x, InnerBaseType) else x for x in args]
    expr = getattr(x.value, method_name)(*args, **kwargs)
    if mask is not None:
        mask = grblas_mask_type(mask.value)
    return wrap_inner(expr.new(dtype=dtype, mask=mask))


def _reduce_axis(op, gb_dtype, x, axis=None, keepdims=None, computing_meta=None, dtype=None):
    """ Call reduce_rowwise or reduce_columnwise on each chunk"""
    if computing_meta:
        return np.empty(0, dtype=dtype)
    if axis == (1,):
        return wrap_inner(x.value.reduce_rowwise(op).new(dtype=gb_dtype))
    if axis == (0,):
        return wrap_inner(x.value.reduce_columnwise(op).new(dtype=gb_dtype))
    

def _reduce_axis_combine(op, x, axis=None, keepdims=None, computing_meta=None, dtype=None):
    """ Combine results from _reduce_axis on each chunk"""
    if computing_meta:
        return np.empty(0, dtype=dtype)
    if type(x) is list:
        vals = [val.value for val in x]
        return wrap_inner(reduce(lambda x, y: x.ewise_add(y, op).new(), vals))
    return x


def _reduce_scalar(op, gb_dtype, x, axis=None, keepdims=None, computing_meta=None, dtype=None):
    """ Call reduce_scalar on each chunk"""
    if computing_meta:
        return np.empty(0, dtype=dtype)
    return wrap_inner(x.value.reduce_scalar(op).new(dtype=gb_dtype))


def _reduce(op, gb_dtype, x, axis=None, keepdims=None, computing_meta=None, dtype=None):
    """ Call reduce on each chunk"""
    if computing_meta:
        return np.empty(0, dtype=dtype)
    return wrap_inner(x.value.reduce(op).new(dtype=gb_dtype))


def _reduce_combine(op, x, axis=None, keepdims=None, computing_meta=None, dtype=None):
    """ Combine results from reduce or reduce_scalar on each chunk"""
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
    """ Accumulate the results of reduce with a scalar"""
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
    """ Accumulate the results of reduce_axis with a vector"""
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


# This mutates the value in `updating`
def _update_expr(method_name, updating, x, kwargs, *args):
    # v << left.ewise_mult(right)
    args = [x.value if isinstance(x, InnerBaseType) else x for x in args]
    expr = getattr(x.value, method_name)(*args, **kwargs)
    updating.value << expr
    return updating


# This mutates the value in `updating`
def _update_expr_full(method_name, updating, accum, mask, mask_type, replace, x, kwargs, *args):
    # v(mask=mask) << left.ewise_mult(right)
    args = [x.value if isinstance(x, InnerBaseType) else x for x in args]
    expr = getattr(x.value, method_name)(*args, **kwargs)
    if mask is not None:
        mask = mask_type(mask.value)
    updating.value(accum=accum, mask=mask, replace=replace) << expr
    return updating


def _matmul(op, dtype, not_updating, no_mask, mask_type, accum, *args, computing_meta=None):
    if computing_meta:
        return np.empty(0, dtype=dtype)

    if not_updating:
        a_blocks, b_blocks = args
        vals = [op(a.value @ b.value).new(dtype=dtype)
                for a, b in zip(a_blocks, b_blocks)]
        return wrap_inner(reduce(lambda x, y: x.ewise_add(y, op.monoid).new(),
                                 vals))
    else:
        if no_mask:
            u, a_blocks, b_blocks = args
            mask = None
        else:
            u, mask, a_blocks, b_blocks = args
            mask = mask_type(mask.value)

        vals = [op(a.value @ b.value).new(dtype=dtype)
                for a, b in zip(a_blocks, b_blocks)]
        gb_obj = reduce(lambda x, y: x.ewise_add(y, op.monoid).new(),
                        vals)
        u.value(mask=mask, accum=accum) << gb_obj
        return u


def asOneColMatrix(inner_vector):
    vector = inner_vector.value
    contents = vector.ss.export(format='sparse', raw=True, give_ownership=False)
    return wrap_inner(gb.Matrix.ss.import_csc(
        nrows=contents['size'], ncols=1,
        indptr=np.array([0, contents['nvals']], dtype=contents['indices'].dtype),
        values=contents['values'],
        row_indices=contents['indices'],
        take_ownership=False
    ))


def asOneRowMatrix(inner_vector):
    vector = inner_vector.value
    contents = vector.ss.export(format='sparse', raw=True, give_ownership=False)
    return wrap_inner(gb.Matrix.ss.import_csr(
        ncols=contents['size'], nrows=1,
        indptr=np.array([0, contents['nvals']], dtype=contents['indices'].dtype),
        values=contents['values'],
        col_indices=contents['indices'],
        take_ownership=False
    ))


def asVector(matrix):
    if matrix.nrows == 1:
        contents = matrix.ss.unpack(format='csr', raw=True)
        return gb.Vector.ss.import_sparse(
            size=contents['ncols'],
            indices=contents['col_indices'],
            values=contents['values'],
            take_ownership=True
        )
    elif matrix.ncols == 1:
        contents = matrix.ss.unpack(format='csc', raw=True)
        return gb.Vector.ss.import_sparse(
            size=contents['nrows'],
            indices=contents['row_indices'],
            values=contents['values'],
            take_ownership=True
        )
    else:
        raise NotImplementedError()


def _matmul2(op, dtype, a, b, computing_meta=None):
    if computing_meta:
        return np.empty(0, dtype=dtype)

    return FakeInnerTensor(op(a.value @ b.value).new(dtype=dtype),
                           face=(0, 2))


def _sum_by_monoid(monoid, a, axis=None, keepdims=None):
    axis, = axis
    demote = lambda x: x if x < axis else (x - 1)
    if type(a) is not list:
        if keepdims:
            return a
        else:
            if a.shape[axis] == 1:
                # remove axis, relocating face
                face0 = demote(a.face[0])
                face1 = demote(a.face[1])
                before = a.shape[:axis]
                beyond = () if axis == a.ndim - 1 else a.shape[axis + 1:]
                return a.reshape(before + beyond, face=(face0, face1))
            else:
                raise NotImplementedError()
    vals = [x.value for x in a]
    out = FakeInnerTensor(
        reduce(lambda x, y: x.ewise_add(y, monoid).new(), vals),
        shape=a[0].shape, face=a[0].face)
    if keepdims:
        return out
    # remove axis
    if a[0].ndim == 3:
        return wrap_inner(out.value)    # now 2D
    if a[0].ndim == 4:
        # remember to relocate face
        face0 = demote(a[0].face[0])
        face1 = demote(a[0].face[1])
        before = a[0].shape[:axis]
        beyond = () if axis == a[0].ndim - 1 else a[0].shape[axis + 1:]
        return out.reshape(before + beyond, face=(face0, face1)) # now 3D
    else:
        raise NotImplementedError()


def sum_by_monoid(monoid, a, axis=None, dtype=None, keepdims=False, split_every=None, out=None, meta=None):
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
