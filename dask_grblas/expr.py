from collections import namedtuple
from functools import partial, reduce
from numbers import Number, Integral

import dask.array as da
import numpy as np
import grblas as gb

from grblas.exceptions import DimensionMismatch
from dask.base import tokenize

from .base import BaseType, InnerBaseType, _check_mask, DOnion, is_DOnion, like_DOnion
from .mask import Mask
from .utils import (
    get_grblas_type,
    get_meta,
    get_return_type,
    np_dtype,
    wrap_inner,
    build_chunk_offsets_dask_array,
    build_chunk_ranges_dask_array,
    build_slice_dask_array_from_chunks,
)


class GbDelayed:
    def __init__(self, parent, method_name, *args, meta=None, **kwargs):
        self.has_dOnion = np.any([getattr(x, "is_dOnion", False) for x in (parent,) + args])
        self.parent = parent
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
        self._meta = meta
        # InfixExpression and Aggregator requirements:
        self.dtype = meta.dtype
        self.output_type = meta.output_type
        self.ndim = len(meta.shape)
        if self.ndim == 1:
            self._size = meta.size
        elif self.ndim == 2:
            self._nrows = meta.nrows
            self._ncols = meta.ncols

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
        at = self.parent._is_transposed
        T = (1, 0) if at else (0, 1)
        delayed = self.parent._matrix._delayed if at else self.parent._delayed
        delayed = da.reduction(
            delayed,
            partial(_reduce_axis, at, op, dtype),
            partial(_reduce_axis_combine, op),
            concatenate=False,
            dtype=np_dtype(dtype),
            axis=T[axis],
        )
        return delayed

    def _reduce_scalar(self, dtype):
        assert not self.kwargs
        op = self.args[0]
        at = self.parent._is_transposed
        delayed = self.parent._matrix._delayed if at else self.parent._delayed
        delayed = da.reduction(
            delayed,
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

    def _aggregate(
        self, op, updating=None, dtype=None, mask=None, accum=None, replace=None, name=None
    ):
        """Handover to the Aggregator to compute the reduction"""

        if updating is None:
            output = self.construct_output(dtype, name=name)
            updater = output(mask=mask)
        else:
            output = updating
            updater = output(mask=mask, accum=accum, replace=replace)

        self.args = (self.parent,) + self.args
        if self.parent.ndim == 1:
            self.cfunc_name = "GrB_Vector_reduce_Aggregator"
        elif self.parent.ndim == 2:
            if self.method_name == "reduce_scalar":
                self.cfunc_name = "GrB_Matrix_reduce_scalar_Aggregator"
            else:
                self.cfunc_name = "GrB_Matrix_reduce_Aggregator"
        op._new(updater, self)
        return output

    def new(self, dtype=None, *, mask=None, name=None):
        if mask is not None:
            _check_mask(mask)

        if self.has_dOnion or mask is not None and mask.is_dOnion:

            def GbDelayed_new(p, pt, m, t, *args, dtype=None, mask=None, **kwargs):
                p = p.T if pt else p
                args = tuple(a.T if xt else a for (xt, a) in zip(t, args))
                gbd = getattr(p, m)(*args, **kwargs)
                return gbd.new(dtype=dtype, mask=mask)

            gbd_args = tuple(getattr(x, "dOnion_if", x) for x in self.args)
            is_T = tuple(
                getattr(x, "is_dOnion", False) and getattr(x, "_is_transposed", False)
                for x in self.args
            )
            gbd_kwargs = {k: getattr(v, "dOnion_if", v) for k, v in self.kwargs.items()}
            meta_kwargs = {k: getattr(v, "_meta", v) for k, v in self.kwargs.items()}

            if self.method_name.startswith(("reduce", "apply")):
                # unary operations
                a = self.parent
                op = self.args[0]
                args = self.args[1:]
                if self.method_name == "apply":
                    # grblas does not like empty Scalars!
                    if "left" in meta_kwargs and type(meta_kwargs["left"]) is gb.Scalar:
                        meta_kwargs["left"] = gb.Scalar.from_value(
                            1, dtype=meta_kwargs["left"].dtype
                        )
                    if "right" in meta_kwargs and type(meta_kwargs["right"]) is gb.Scalar:
                        meta_kwargs["right"] = gb.Scalar.from_value(
                            1, dtype=meta_kwargs["right"].dtype
                        )
                elif self.method_name.startswith("reduce"):
                    # grblas bug occurs when shape is (0, 0)
                    if a._meta.shape == (0,) * a.ndim:
                        a._meta.resize(*((1,) * a.ndim))
                meta = getattr(a._meta, self.method_name)(op, *args, **meta_kwargs).new(dtype=dtype)
                meta.clear()
            else:
                # binary operations
                a = self.parent
                b = self.args[0]
                op = self.args[1]

                try:
                    meta = getattr(a._meta, self.method_name)(b._meta, op=op, **meta_kwargs).new(
                        dtype=dtype
                    )
                except DimensionMismatch:
                    if self.method_name == "mxm":
                        b_meta = gb.Matrix.new(
                            dtype=b._meta.dtype, nrows=a._meta.ncols, ncols=b._meta.ncols
                        )
                    elif self.method_name == "vxm":
                        b_meta = gb.Matrix.new(
                            dtype=b._meta.dtype, nrows=a._meta.size, ncols=b._meta.ncols
                        )
                    elif self.method_name == "mxv":
                        b_meta = gb.Vector.new(dtype=b._meta.dtype, size=a._meta.ncols)

                    elif self.method_name in ("ewise_add", "ewise_mult"):
                        b_meta = a._meta.dup(dtype=b._meta.dtype)

                    meta = getattr(a._meta, self.method_name)(b_meta, op=op, **meta_kwargs).new(
                        dtype=dtype
                    )

            donion = DOnion.multiple_access(
                meta,
                GbDelayed_new,
                a.dOnion_if,
                a.is_dOnion and getattr(a, "_is_transposed", False),
                self.method_name,
                is_T,
                *gbd_args,
                dtype=dtype,
                mask=None if mask is None else mask.dOnion_if,
                **gbd_kwargs,
            )
            return get_return_type(meta)(donion, meta=meta)

        if mask is not None:
            meta = self._meta.new(dtype=dtype, mask=mask._meta)
            delayed_mask = mask.mask._delayed
            grblas_mask_type = get_grblas_type(mask)
        else:
            meta = self._meta.new(dtype=dtype)
            delayed_mask = None
            grblas_mask_type = None

        if self.method_name.startswith("reduce"):
            op = self._meta.op
            if op is not None and op.opclass == "Aggregator":
                return self._aggregate(op, dtype=dtype, mask=mask, name=name)

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
            delayed = self._matmul2(meta, mask=mask)
        else:
            raise ValueError(self.method_name)
        return get_return_type(meta)(delayed)

    def _update(self, updating, *, mask=None, accum=None, replace=None):
        updating._meta.update(self._meta)
        assert updating._meta._is_scalar or updating._meta.nvals == 0
        meta = updating._meta

        if self.method_name.startswith("reduce"):
            op = self._meta.op
            if op is not None and op.opclass == "Aggregator":
                self._aggregate(op, updating, mask=mask, accum=accum, replace=replace)
                return

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
        updating.__init__(delayed)

    def construct_output(self, dtype=None, *, name=None):
        if dtype is None:
            dtype = self.dtype
        return get_return_type(self._meta.output_type.new(dtype)).new(
            dtype, *self._meta.shape, name=name
        )

    @property
    def value(self):
        self._meta.value
        return self.new().value

    def _new_scalar(self, dtype, *, name=None):
        """Create a new empty Scalar."""
        from .scalar import Scalar

        return Scalar.new(dtype, name=name)

    def _new_vector(self, dtype, size=0, *, name=None):
        """Create a new empty Vector."""
        from .vector import Vector

        return Vector.new(dtype, size, name=name)

    def _new_matrix(self, dtype, nrows=0, ncols=0, *, name=None):
        """Create a new empty Matrix."""
        from .matrix import Matrix

        return Matrix.new(dtype, nrows, ncols, name=name)


AxisIndex = namedtuple("AxisIndex", ["size", "index"])


class IndexerResolver:
    def __init__(self, obj, indices, check_shape=True):
        self.obj = obj
        if indices is Ellipsis:
            from .vector import Vector

            if type(obj) in {Vector, gb.Vector}:
                normalized = slice(None).indices(obj._size)
                self.indices = [AxisIndex(obj._size, slice(*normalized))]
            else:
                normalized0 = slice(None).indices(obj._nrows)
                normalized1 = slice(None).indices(obj._ncols)
                self.indices = [
                    AxisIndex(obj._nrows, slice(*normalized0)),
                    AxisIndex(obj._ncols, slice(*normalized1)),
                ]
        else:
            if not check_shape and hasattr(obj, "_meta"):
                shape = obj._meta.shape
            else:
                shape = obj.shape
            self.indices = self.parse_indices(indices, shape, check_shape)

    @property
    def is_single_element(self):
        for idx in self.indices:
            if idx.size is not None:
                return False
        return True

    def parse_indices(self, indices, shape, check_shape=True):
        """
        Returns
            [(rows, rowsize), (cols, colsize)] for Matrix
            [(idx, idx_size)] for Vector

        Within each tuple, if the index is of type int, the size will be None
        """
        if len(shape) == 1:
            if type(indices) is tuple:
                raise TypeError(f"Index for {type(self.obj).__name__} cannot be a tuple")
            # Convert to tuple for consistent processing
            indices = (indices,)
        else:  # len(shape) == 2
            if type(indices) is not tuple or len(indices) != 2:
                raise TypeError(f"Index for {type(self.obj).__name__} must be a 2-tuple")

        out = []
        for i, idx in enumerate(indices):
            typ = type(idx)
            if typ is tuple:
                raise TypeError(
                    f"Index in position {i} cannot be a tuple; must use slice or list or int"
                )
            out.append(self.parse_index(idx, typ, shape[i], check_shape))
        return out

    def parse_index(self, index, typ, size, check_shape=True):
        if np.issubdtype(typ, np.integer):
            if index >= size:
                if check_shape:
                    raise IndexError(f"Index out of range: index={index}, size={size}")
            if index < 0:
                index += size
                if index < 0:
                    if check_shape:
                        raise IndexError(f"Index out of range: index={index - size}, size={size}")
            return AxisIndex(None, IndexerResolver.normalize_index(index, size, check_shape))
        if typ is list:
            index = [IndexerResolver.normalize_index(i, size, check_shape) for i in index]
            return AxisIndex(len(index), index)
        elif typ is slice:
            if check_shape:
                normalized = index.indices(size)
                return AxisIndex(len(range(*normalized)), slice(*normalized))
            else:
                return AxisIndex(None, index)

        elif typ in {np.ndarray, da.Array}:
            if len(index.shape) != 1:
                raise TypeError(f"Invalid number of dimensions for index: {len(index.shape)}")
            if not np.issubdtype(index.dtype, np.integer):
                raise TypeError(f"Invalid dtype for index: {index.dtype}")
            return AxisIndex(index.shape[0], index)

        elif is_DOnion(index):
            return AxisIndex(None, index)

        else:
            from .scalar import Scalar

            if typ is Scalar:
                if index.dtype.name.startswith("F"):
                    raise TypeError(f"An integer is required for indexing.  Got: {index.dtype}")
                index = index.value.compute()
                return AxisIndex(None, IndexerResolver.normalize_index(index, size))

            from .matrix import Matrix, TransposedMatrix
            from .vector import Vector

            if typ is Vector or typ is Matrix:
                raise TypeError(
                    f"Invalid type for index: {typ.__name__}.\n"
                    f"If you want to apply a mask, perhaps do something like "
                    f"`x.dup(mask={index.name}.S)`.\n"
                    f"If you want to assign with a mask, perhaps do something like "
                    f"`x(mask={index.name}.S) << value`."
                )
            elif typ is TransposedMatrix:
                raise TypeError(f"Invalid type for index: {typ.__name__}.")
            try:
                index = list(index)
            except Exception:
                from .mask import Mask

                if isinstance(index, Mask):
                    raise TypeError(
                        f"Invalid type for index: {typ.__name__}.\n"
                        f"If you want to apply a mask, perhaps do something like "
                        f"`x.dup(mask={index.name})`.\n"
                        f"If you want to assign with a mask, perhaps do something like "
                        f"`x(mask={index.name}) << value`."
                    )
                raise TypeError(f"Invalid type for index: {typ}; unable to convert to list")
            index = [IndexerResolver.normalize_index(i, size, check_shape) for i in index]
        return AxisIndex(len(index), index)

    def get_index(self, dim):
        """Return a new IndexerResolver with index for the selected dimension"""
        rv = object.__new__(IndexerResolver)
        rv.obj = self.obj
        rv.indices = (self.indices[dim],)
        return rv

    @classmethod
    def validate_types(cls, indices):
        for i in indices:
            if not isinstance(i, Integral) and type(i) not in {list, slice, np.ndarray, da.Array}:
                raise TypeError(f"Invalid type for index: {type(i).__name__}.")
        return

    @classmethod
    def normalize_index(cls, index, size, check_size=True):
        if type(index) is get_return_type(gb.Scalar.new(int)):
            # This branch needs a second look: How to work with the lazy index?
            index = index.value.compute()
            if not isinstance(index, Integral):
                raise TypeError("An integer is required for indexing")
        if index >= size:
            if check_size:
                raise IndexError(f"Index out of range: index={index}, size={size}")
        if index < 0:
            index += size
            if index < 0:
                if check_size:
                    raise IndexError(f"Index out of range: index={index - size}, size={size}")
        return int(index)


class Updater:
    def __init__(self, parent, *, mask=None, accum=None, replace=False, input_mask=None):
        if input_mask is not None and mask is not None:
            raise TypeError("mask and input_mask arguments cannot both be given")
        if input_mask is not None and not isinstance(input_mask, Mask):
            raise TypeError(r"Mask must indicate values (M.V) or structure (M.S)")

        self.parent = parent
        self.mask = mask
        self.input_mask = input_mask
        self.accum = accum
        if mask is None:
            self.replace = None
        else:
            self.replace = replace
        self._meta = parent._meta(mask=get_meta(mask), accum=accum, replace=replace)
        # Aggregator specific attribute requirements:
        self.kwargs = {"mask": mask}

    def __delitem__(self, keys):
        # Occurs when user calls `del C(params)[index]`
        if self.parent._is_scalar:
            raise TypeError("Indexing not supported for Scalars")
        resolved_indexes = IndexerResolver(self.parent, keys)
        if resolved_indexes.is_single_element:
            self.parent._delete_element(resolved_indexes)
        else:
            raise TypeError("Remove Element only supports a single index")

    def __getitem__(self, keys):
        if self.input_mask is not None:
            raise TypeError("`input_mask` argument may only be used for extract")

        return Assigner(self, keys)

    def __setitem__(self, keys, obj):
        if self.input_mask is not None:
            raise TypeError("`input_mask` argument may only be used for extract")

        Assigner(self, keys).update(obj)

    def __lshift__(self, delayed):
        # Occurs when user calls C(params) << delayed
        self.update(delayed)

    def update(self, delayed):
        # Occurs when user calls C(params) << delayed
        if self.input_mask is not None:
            if type(delayed) is AmbiguousAssignOrExtract:
                # w(input_mask) << v[index]
                if self.parent is delayed.parent:
                    delayed.parent = delayed.parent.__class__(
                        delayed.parent._delayed, delayed.parent._meta
                    )
                self.parent._update(
                    delayed.new(mask=self.mask, input_mask=self.input_mask),
                    accum=self.accum,
                    mask=self.mask,
                    replace=self.replace,
                )
                return
            else:
                raise TypeError("`input_mask` argument may only be used for extract")

        if isinstance(delayed, Number) or (
            isinstance(delayed, BaseType) and get_meta(delayed)._is_scalar
        ):
            ndim = self.parent.ndim
            if ndim > 0:
                self.__setitem__(_squeeze((slice(None),) * ndim), delayed)
            elif self.accum is not None:
                raise TypeError("Accumulating into Scalars is not supported")
            elif self.mask is not None:
                raise TypeError("Mask not allowed for Scalars")
            return

        if self.mask is None and self.accum is None:
            return self.parent.update(delayed)

        if not (like_DOnion(self.parent) or like_DOnion(delayed)):
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


def _csc_chunk(row_range, col_range, indices, red_columns, track_indices=False):
    row_range = row_range[0]
    nrows = row_range.stop - row_range.start
    if type(indices[0]) is slice:
        s = indices[0]
        indices = np.arange(s.start, s.stop, s.step, dtype=np.int64)
    ncols = indices.size
    idx = indices - row_range.start
    idx_filter = (0 <= idx) & (idx < nrows)
    if red_columns is not None:
        _, red_columns = red_columns.value.to_values()
        col_range = col_range[0]
        col_range = np.arange(col_range.start, col_range.stop, dtype=indices.dtype)
        idx_filter = idx_filter & np.isin(col_range, red_columns)
    if track_indices:
        col_range = col_range[0]
        values = np.arange(col_range.start, col_range.stop, dtype=indices.dtype)
        values = values[idx_filter]
        is_iso = False
    else:
        values = np.array([1], dtype=indices.dtype)
        is_iso = True
    row_indices = idx[idx_filter]
    indptr = np.empty(ncols + 1, np.int64)
    indptr[0] = 0
    np.cumsum(idx_filter, dtype=indptr.dtype, out=indptr[1:])
    return wrap_inner(
        gb.Matrix.ss.import_csc(
            nrows=nrows,
            ncols=ncols,
            indptr=indptr,
            row_indices=row_indices,
            values=values,
            is_iso=is_iso,
            take_ownership=True,
        )
    )


def _fill(inner_vector, rhs):
    rhs = rhs.value if isinstance(rhs, InnerBaseType) else rhs
    inner_vector.value[:] << rhs
    return inner_vector


def reduce_assign(lhs, indices, rhs, dup_op="last", mask=None, accum=None, replace=False):
    # lhs(mask, accum, replace)[i] << rhs
    rhs_is_scalar = not (isinstance(rhs, BaseType) and type(rhs._meta) is gb.Vector)
    if type(indices) is slice:
        chunksz = "auto" if rhs_is_scalar else rhs._delayed.chunks
        indices = build_slice_dask_array_from_chunks(indices, lhs.size, chunksz)
        indices_dtype = np.int64
    elif type(indices) in {list, np.ndarray}:
        chunksz = "auto" if rhs_is_scalar else rhs._delayed.chunks
        name = "indices-array" + tokenize(indices, chunksz)
        indices = da.from_array(np.array(indices), chunks=chunksz, name=name)
        indices_dtype = indices.dtype
    else:
        indices_dtype = indices.dtype

    if rhs_is_scalar:
        # TODO: optimize this branch by avoiding O(N) fill-operation
        if isinstance(rhs, BaseType):
            dtype = rhs._meta.dtype
            np_dtype_ = np_dtype(dtype)
            meta = wrap_inner(rhs._meta)
            rhs = rhs._delayed
        else:
            dtype = type(rhs)
            np_dtype_ = dtype
            meta = np.array([], dtype=np_dtype_)
        rhs_vec = get_return_type(lhs).new(dtype, size=indices.size, chunks=(indices.chunks[0],))
        rhs_vec.__init__(da.map_blocks(_fill, rhs_vec._delayed, rhs, dtype=np_dtype_, meta=meta))
        rhs = rhs_vec

    # create CSC matrix C from indices:
    dtype = indices_dtype
    meta = gb.Matrix.new(dtype)
    lhs_chunk_ranges = build_chunk_ranges_dask_array(lhs._delayed, 0, "lhs-ranges")
    # deal with default
    dup_ops = {"first": gb.monoid.min, "last": gb.monoid.max}
    if dup_op in dup_ops:
        # remove this branch when near-semirings get supported
        indices_chunk_ranges = build_chunk_ranges_dask_array(indices, 0, "indices-ranges")
        delayed = da.core.blockwise(
            *(_csc_chunk, "ij"),
            *(lhs_chunk_ranges, "i"),
            *(indices_chunk_ranges, "j"),
            *(indices, "j"),
            *(None, None),
            track_indices=True,
            dtype=dtype,
            meta=wrap_inner(meta),
        )
        C = get_return_type(meta)(delayed)
        red_columns = C.reduce_rowwise(op=dup_ops[dup_op]).new()
        delayed = da.core.blockwise(
            *(_csc_chunk, "ij"),
            *(lhs_chunk_ranges, "i"),
            *(indices_chunk_ranges, "j"),
            *(indices, "j"),
            *(red_columns._delayed, "i"),
            dtype=dtype,
            meta=wrap_inner(meta),
        )
        C = get_return_type(meta)(delayed)
        dup_op = gb.monoid.any
    else:
        delayed = da.core.blockwise(
            *(_csc_chunk, "ij"),
            *(lhs_chunk_ranges, "i"),
            *(None, None),
            *(indices, "j"),
            *(None, None),
            dtype=dtype,
            meta=wrap_inner(meta),
        )
        C = get_return_type(meta)(delayed)
        red_columns = C.reduce_rowwise(op=gb.monoid.any).new()

    semiring_dup_op_2nd = gb.operator.get_semiring(dup_op, gb.binary.second)
    rhs = C.mxv(rhs, semiring_dup_op_2nd).new(mask=mask)
    if accum is None:
        rhs(mask=~red_columns.S) << lhs
    lhs(mask=mask, accum=accum, replace=replace) << rhs


class Fragmenter:
    """
    stores only that part of the data-chunk selected by the index
    """

    def __init__(self, ndim=None, index=None, mask=None, obj=None, ot=False):
        self.ndim = ndim
        self.index = index
        self.mask = mask
        self.obj = obj
        self.ot = ot


def _ceildiv(a, b):
    return -(a // -b)


def _squeeze(tupl):
    if len(tupl) == 1:
        return tupl[0]
    return tupl


def _get_type_with_ndims(n):
    if n == 0:
        return get_return_type(gb.Scalar.new(int))
    elif n == 1:
        return get_return_type(gb.Vector.new(int))
    else:
        return get_return_type(gb.Matrix.new(int))


def _get_grblas_type_with_ndims(n):
    if n == 0:
        return gb.Scalar
    elif n == 1:
        return gb.Vector
    else:
        return gb.Matrix


def _shape(x, indices):
    shape = ()
    for axis, index in enumerate(indices):
        if type(index) is slice:
            start, stop, step = index.indices(x.shape[axis])
            shape += (len(range(start, stop, step)),)
        elif type(index) in {list, np.ndarray}:
            shape += (len(index),)
        elif type(index) is da.Array:
            shape += (index.shape[0],)
    return shape


def fuse_slice_pair(slice0, slice1, length):
    """computes slice `s` such that array[s] = array[s0][s1] where array has length `length`"""
    start0, stop0, step0 = slice0.indices(length)
    start1, stop1, step1 = slice1.indices(len(range(start0, stop0, step0)))
    stop01 = start0 + stop1 * step0
    return slice(start0 + start1 * step0, None if stop01 < 0 else stop01, step0 * step1)


def fuse_index_pair(i, j, length=None):
    """computes indices `s` such that array[s] = array[i][j] where array has length `length`"""
    if type(i) in {list, np.ndarray}:
        return i[j]
    if length is None:
        raise ValueError("Length argument is missing")
    if type(i) is slice and isinstance(j, Integral):
        a0, _, e0 = i.indices(length)
        return a0 + j * e0
    if type(i) is slice and type(j) in {list, np.ndarray}:
        a0, _, e0 = i.indices(length)

        def f(x):
            return a0 + x * e0

        return [f(x) for x in j] if type(j) is list else list(f(j))
    elif type(i) is slice and type(j) is slice:
        return fuse_slice_pair(i, j, length=length)
    else:
        raise NotImplementedError()


def _chunk_in_slice(chunk_begin, chunk_end, slice_start, slice_stop, slice_step):
    """Returns the part of the chunk that intersects the slice, also
    returning True if the intersection exists, otherwise False.
    Zero-length slices that are located within the chunk also
    return True, otherwise False"""

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
        elif stop == start:  # zero length slice
            idx_within = True
            idx = slice(start, stop, step) if slice_step > 0 else slice(-start, -stop, -step)
            return idx_within, idx
        stop = min(ce, stop)
        idx_within = len(range(start, stop, step)) > 0
        if idx_within:
            idx = slice(start, stop, step) if slice_step > 0 else slice(-start, -stop, -step)
        else:
            idx = None
        return idx_within, idx
    else:
        return False, np.array([], dtype=int)


def _data_x_index_meshpoint_4assign(*args, x_ndim, subassign, obj_offset_axes, ot):
    """
    Returns a Fragmenter object containing only that part of
        indices = args[x_ndim : 2 * x_ndim]
    within the bounds
        x_ranges = args[0 : x_ndim]
    of the target inner Vector/Matrix data-chunk.
    The corresponding portions of the mask
        mask = args[2 * x_ndim]
    and the object
        obj  = args[2 * x_ndim + 1]
    being assigned are also contained in the returned Fragmenter object.
    """
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
    T = (1, 0) if ot else (0, 1)
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
                obj_end = s.start + s.step * (obj_offsets[axis] + y.shape[T[obj_axis]])
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
        elif isinstance(indices[axis], Integral):
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
    # ---------------- end for loop --------------------------
    if idx_within:
        if not obj_is_scalar:
            if ot:
                obj_ = obj_[obj_index_tuple[::-1]].new()
            else:
                obj_ = obj_[_squeeze(obj_index_tuple)].new()
        if subassign and mask is not None:
            mask_ = mask_[_squeeze(obj_index_tuple)].new()
            index_tuple, obj_, mask_ = _uniquify(x_ndim, index_tuple, obj_, mask_, ot)
        else:
            mask_ = None
            index_tuple, obj_, _ = _uniquify(x_ndim, index_tuple, obj_, None, ot)
        return Fragmenter(x_ndim, index_tuple, mask_, obj_, ot)
    else:
        return Fragmenter(x_ndim)


def _uniquify_merged(ot, x, axis=None, keepdims=None, computing_meta=None):
    if x.index is None:
        return x
    x.index, x.obj, x.mask = _uniquify(x.ndim, x.index, x.obj, x.mask, ot)
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
    ot,
):
    """
    Performs the actual GrB_assign:
        old_data(mask, ...)[index] << obj
    or GxB_subassign:
        old_data[index](mask, ...) << obj
    The mask, index, and obj data are found in parameter `new_data`
    The updated old_data is returned.
    """
    x = old_data.value

    mask = new_data.mask if subassign else mask
    mask = mask.value if isinstance(mask, InnerBaseType) else mask
    if mask is not None:
        mask = mask_type(mask)

    if new_data.index is None:
        # this branch takes care of those data chunks that are not targeted by the
        # indices, but which the mask (in GrB_assign and GrB_row/col_assign) covers
        # when replace is True, elements not covered by the mask are deleted
        if not subassign and replace:
            if band_selection:
                # GrB_row/_column assign
                chunk_band = band_selection[band_axis] - band_offset[0].start
                if 0 <= chunk_band and chunk_band < x.shape[band_axis]:
                    band_selection = band_selection.copy()
                    band_selection[band_axis] = chunk_band
                    band_selection = tuple(band_selection)
                    x(mask=mask, replace=replace)[band_selection] << x[band_selection].new()
            else:
                # GrB_assign
                x(mask=mask, replace=replace) << x
        return wrap_inner(x)

    normalized_index = ()
    for i in new_data.index:
        if type(i) is slice and i.start >= 0 and i.stop < 0 and i.step < 0:
            i = slice(i.start, None, i.step)
        normalized_index += (i,)
    index = _squeeze(normalized_index)

    obj = new_data.obj.T if ot else new_data.obj

    if subassign:
        x[index](mask=mask, replace=replace, accum=accum) << obj
    else:
        x(mask=mask, replace=replace, accum=accum)[index] << obj

    return wrap_inner(x)


def _upcast(grblas_object, ndim, axis_is_missing):
    """
    Returns grblas_object upcast to the given number `ndim` of
    dimensions.  The missing axis/axes are determined by means
    of the list `axis_is_missing` of bool datatypes and whose
    length is `ndim`.
    """
    input = grblas_object
    if np.all(axis_is_missing):
        # grblas_object is a scalar
        if ndim == 1:
            # upcast grblas.Scalar to grblas.Vector
            output = gb.Vector.new(input.dtype, size=1)
            if input.value is not None:
                output[0] = input
        else:
            # upcast grblas.Scalar to grblas.Matrix
            output = gb.Matrix.new(input.dtype, nrows=1, ncols=1)
            if input.value is not None:
                output[0, 0] = input
        return output
    elif ndim == 2:
        # grblas_object is a Vector
        if axis_is_missing[0]:
            # upcast grblas.Vector to one-row grblas.Matrix
            return input._as_matrix().T.new()
        elif axis_is_missing[1]:
            # upcast grblas.Vector to one-column grblas.Matrix
            return input._as_matrix()
    return input


def _data_x_index_meshpoint_4extract(
    *args,
    xt,
    input_mask_type,
    mask_type,
    gb_dtype,
):
    """
    Returns only that part of the source inner Vector/Matrix data-chunk x = args[0]
    selected by the indices = args[2 : 2 + x.ndim].
    The number of dimensions of the output is forced to be the same as that of
    x (the source inner Vector/Matrix data-chunk) by upcasting if necessary.
    xt (bool) is a flag that denotes whether x is an InnerMatrix and is transposed
    (True), otherwise it is False.
    """
    x = args[0]
    input_mask = args[1]
    indices = args[2 : x.ndim + 2]
    mask = args[x.ndim + 2]
    x_offsets = args[x.ndim + 3 :]

    T = (1, 0) if xt else (0, 1)

    index_tuple = ()
    index_is_a_number = []
    out_shape = ()
    mask_index_tuple = ()
    within_bounds = True
    for axis in range(x.ndim):
        index_is_a_number += [False]
        index = indices[axis]
        offset = x_offsets[T[axis]][0]
        mask_idx = None
        if type(index) is np.ndarray:
            if type(index[0]) is slice:
                # Note: slice is already aligned with mask if it exists
                s = index[0]
                idx_within, idx = _chunk_in_slice(
                    offset, offset + x.shape[T[axis]], s.start, s.stop, s.step
                )
                if idx_within:
                    out_shape += (len(range(idx.start, idx.stop, idx.step)),)
                    mask_begin = _ceildiv(idx.start - s.start, s.step)
                    mask_end = _ceildiv(idx.stop - s.start, s.step)
                    # beware of negative indices in slice specification!
                    stop = idx.stop - offset
                    idx = slice(idx.start - offset, stop if stop >= 0 else None, idx.step)
                    stop = _ceildiv(stop - s.start, s.step)
                    mask_idx = slice(mask_begin, mask_end, 1)
            else:
                idx = np.array(index) - offset
                idx_filter = (idx >= 0) & (idx < x.shape[T[axis]])
                idx_within = np.any(idx_filter)
                if idx_within:
                    idx = idx[idx_filter]
                    out_shape += (len(idx),)
                    mask_idx = np.argwhere(idx_filter)[:, 0]
        elif isinstance(index, Integral):
            index_is_a_number[-1] = True
            idx = index - offset
            idx_within = (idx >= 0) and (idx < x.shape[T[axis]])
            mask_idx = None
            if idx_within:
                out_shape += (1,)
        else:
            raise NotImplementedError("Index type is unknown.")

        if not idx_within:
            out_shape += (0,)
        index_tuple += (idx,)
        within_bounds = within_bounds and idx_within
        mask_index_tuple += (mask_idx,)
    # ---------------- end for loop --------------------------

    x = x.value
    if within_bounds:
        index_tuple = _squeeze(index_tuple)
        mask_index_tuple = _squeeze(mask_index_tuple)
        mask = mask_type(mask.value[mask_index_tuple].new()) if mask is not None else None
        input_mask = input_mask_type(input_mask.value) if input_mask is not None else None
        x = x.T if xt else x
        out = x[index_tuple].new(gb_dtype, mask=mask, input_mask=input_mask)

        # Now we need to upcast `out` to the required number
        # of dimensions (x.ndim) in order to enable concatenation
        # (in the next blockwise) along the missing axis/axes.
        return wrap_inner(_upcast(out, x.ndim, index_is_a_number))
    else:
        return wrap_inner(type(x).new(gb_dtype, *out_shape))


def _defrag_to_index_chunk(*args, x_chunks, dtype=None):
    """
    Rearranges the concatenated chunk-fragments, so that their order corresponds with the
    the original order of indices of the index chunk.
    """
    ndim = len(x_chunks)
    indices = args[0:ndim]
    fused_fragments = args[ndim].value

    index_tuple = ()
    for axis, idx in enumerate(indices):
        if isinstance(idx, Integral):
            index_tuple += (0,)
            continue
        elif type(idx) is np.ndarray and type(idx[0]) is slice:
            s = idx[0]
            # this branch needs more efficient-handling: (e.g. we should avoid use of np.arange)
            idx = np.arange(s.start, s.stop, s.step, dtype=np.int64)
            if idx.size == 0:
                index_tuple += (s,)
                continue

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
    # ---------------- end for loop --------------------------

    index_tuple = _squeeze(index_tuple)
    return wrap_inner(fused_fragments[index_tuple].new())


def _adjust_meta_to_index(meta, index):
    from .scalar import Scalar, PythonScalar

    # Since grblas does not support indices that are dask arrays
    # this complicates meta deduction.  We therefore substitute
    # any non-Integral type indices with `slice(None)`
    index = index if type(index) is tuple else (index,)
    # Next, we resize `meta` to accept any Integral-type indices:
    numbers = [x for x in index if isinstance(x, (Integral, Scalar, PythonScalar))]
    max_index = np.max(numbers) if numbers else None
    meta = meta.dup()
    if max_index is not None:
        if len(index) == 1:
            meta.resize(max_index + 1)
        else:
            meta.resize(max_index + 1, max_index + 1)

    meta_index = tuple(
        x if isinstance(x, (Integral, Scalar, PythonScalar)) else slice(None) for x in index
    )
    return meta[_squeeze(meta_index)]


class AmbiguousAssignOrExtract:
    def __init__(self, parent, index, meta=None):
        self.parent = parent
        self.index = index
        input_ndim = parent.ndim
        self.keys_0_is_dOnion = input_ndim == 1 and is_DOnion(index)
        self.keys_1_is_dOnion = (
            input_ndim == 2
            and type(index) is tuple
            and len(index) == 2
            and (is_DOnion(index[0]) or is_DOnion(index[1]))
        )
        if parent.is_dOnion or self.keys_0_is_dOnion or self.keys_1_is_dOnion:
            IndexerResolver(self.parent, index, check_shape=False)
            self._meta = _adjust_meta_to_index(parent._meta, index)
            self.has_dOnion = True
        else:
            self.resolved_indices = IndexerResolver(parent, index)
            self._meta = parent._meta[index] if meta is None else meta
            self.has_dOnion = False
            # infix expression requirements:
            shape = tuple(i.size for i in self.resolved_indices.indices if i.size)
            self.ndim = len(shape)
            self.output_type = _get_grblas_type_with_ndims(self.ndim)
            if self.ndim == 1:
                self._size = shape[0]
            elif self.ndim == 2:
                self._nrows = shape[0]
                self._ncols = shape[1]

    def new(self, *, dtype=None, mask=None, input_mask=None, name=None):
        def getitem(parent, at, keys_0, keys_1, dtype, mask, input_mask):
            keys = keys_0 if keys_1 is None else (keys_0, keys_1)
            return AmbiguousAssignOrExtract(parent.T if at else parent, keys).new(
                dtype=dtype, mask=mask, input_mask=input_mask
            )

        if mask is not None:
            _check_mask(mask)
        if input_mask is not None:
            _check_mask(input_mask)

        mask_is_DOnion = mask is not None and mask.is_dOnion
        input_mask_is_DOnion = input_mask is not None and input_mask.is_dOnion
        if (
            self.parent.is_dOnion
            or self.keys_0_is_dOnion
            or self.keys_1_is_dOnion
            or mask_is_DOnion
            or input_mask_is_DOnion
        ):
            meta = self._meta.new(dtype=dtype)

            if type(self.index) is tuple and len(self.index) == 2:
                keys_0, keys_1 = self.index[0], self.index[1]
            else:
                keys_0, keys_1 = self.index, None

            donion = DOnion.multiple_access(
                meta,
                getitem,
                self.parent.dOnion_if,
                self.parent.is_dOnion and getattr(self.parent, "_is_transposed", False),
                *(keys_0, keys_1),
                dtype=dtype,
                mask=None if mask is None else mask.dOnion_if,
                input_mask=None if input_mask is None else input_mask.dOnion_if,
            )
            return get_return_type(meta)(donion)

        # no dOnions
        parent = self.parent
        xt = False  # xt = parent._is_transposed
        dxn = 1  # dxn = -1 if xt else 1
        T = (0, 1)  # T = (1, 0) if xt else (0, 1)
        if parent.ndim == 2 and parent._is_transposed:
            parent = parent._matrix
            xt = True
            dxn = -1
            T = (1, 0)

        x = parent._delayed
        input_shape = parent.shape[::dxn]
        input_ndim = len(input_shape)
        axes = tuple(range(input_ndim))
        x_axes = axes[::dxn]
        indices = tuple(i.index for i in self.resolved_indices.indices)
        out_shape = tuple(i.size for i in self.resolved_indices.indices if i.size is not None)
        out_ndim = len(out_shape)

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

        if input_mask is not None:
            if mask is not None:
                raise TypeError("mask and input_mask arguments cannot both be given")
            if not isinstance(input_mask, Mask):
                raise TypeError(r"Mask must indicate values (M.V) or structure (M.S)")
            if out_ndim == 0:
                raise TypeError("mask is not allowed for single element extraction")

            delayed_input_mask = input_mask.mask._delayed
            grblas_input_mask_type = get_grblas_type(input_mask)

            input_mask_ndim = len(input_mask.mask.shape)
            if input_ndim == input_mask_ndim:
                if input_shape != input_mask.mask.shape:
                    if input_ndim == 1:
                        raise ValueError("Size of `input_mask` does not match size of input")
                    raise ValueError("Shape of `input_mask` does not match shape of input")

                # align `x` and its input_mask in order to fix offsets
                # of `x` before calculating them
                input_mask_axes = axes
                _, (x, delayed_input_mask) = da.core.unify_chunks(
                    x, x_axes, delayed_input_mask, input_mask_axes
                )

            elif input_ndim < input_mask_ndim:
                if input_ndim == 1:
                    raise TypeError("Mask object must be type Vector")
                else:
                    raise ValueError("Shape of `input_mask` does not match shape of input")

            else:
                if out_ndim == input_ndim:
                    raise TypeError(
                        "Got Vector `input_mask` when extracting a submatrix from a Matrix"
                    )
                elif out_ndim < input_ndim:
                    (rem_axis,) = [
                        axis
                        for axis, index in enumerate(self.resolved_indices.indices)
                        if index.size is not None
                    ]
                    if out_ndim == input_mask_ndim:
                        if input_shape[rem_axis] != input_mask.mask.shape[0]:
                            if rem_axis == 0:
                                raise ValueError(
                                    "Size of `input_mask` Vector does not match nrows of Matrix"
                                )
                            else:
                                raise ValueError(
                                    "Size of `input_mask` Vector does not match ncols of Matrix"
                                )
                    input_mask_axes = (axes[rem_axis],)
                    _, (x, delayed_input_mask) = da.core.unify_chunks(
                        x, x_axes, delayed_input_mask, input_mask_axes
                    )

        else:
            delayed_input_mask = None
            grblas_input_mask_type = None

        dtype = np_dtype(meta.dtype)
        if input_ndim in [1, 2]:
            # prepare arguments for blockwise:
            indices_args = []
            offset_args = []
            new_axes = ()
            for axis in axes:
                indx = indices[axis]
                if type(indx) is da.Array:
                    indices_args += [indx, (input_ndim + axis,)]
                    new_axes += (input_ndim + axis,)
                elif type(indx) in {list, np.ndarray}:
                    # convert list, or numpy array to dask array
                    indx = np.array(indx)
                    name = "list_index-" + tokenize(indx, axis)
                    indx = da.core.from_array(indx, chunks="auto", name=name)
                    indices_args += [indx, (input_ndim + axis,)]
                    new_axes += (input_ndim + axis,)
                elif type(indx) is slice:
                    # convert slice to dask array
                    start, stop, step = indx.start, indx.stop, indx.step
                    indx = slice(start, stop, step)
                    slice_len = len(range(start, stop, step))
                    if slice_len == 0 or mask is None:
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
                    indices_args += [indx, (input_ndim + axis,)]
                    new_axes += (input_ndim + axis,)
                else:
                    indices_args += [indx, None]

                offset = build_chunk_offsets_dask_array(x, axis, "offset-")
                offset_args += [offset, (T[axis],)]
            # ---------------- end for loop --------------------------

            fragments_ind = axes + new_axes
            mask_args = [delayed_mask, new_axes if mask is not None else None]
            input_mask_args = [
                delayed_input_mask,
                input_mask_axes if input_mask is not None else None,
            ]

            # this blockwise is essentially a cartesian product of data chunks and index chunks
            # both index and data chunks are fragmented in the process
            fragments = da.core.blockwise(
                *(_data_x_index_meshpoint_4extract, fragments_ind),
                *(x, x_axes),
                *input_mask_args,
                *indices_args,
                *mask_args,
                *offset_args,
                xt=xt,
                input_mask_type=grblas_input_mask_type,
                mask_type=grblas_mask_type,
                gb_dtype=dtype,
                dtype=dtype,
                meta=wrap_inner(meta),
            )

            # this blockwise is essentially an aggregation over the data chunk axes
            extracts_axes = tuple((i + input_ndim) for i in x_axes if i + input_ndim in new_axes)
            delayed = da.core.blockwise(
                *(_defrag_to_index_chunk, extracts_axes),
                *indices_args,
                *(fragments, fragments_ind),
                x_chunks=x.chunks[::dxn],
                concatenate=True,
                dtype=dtype,
                meta=wrap_inner(meta),
            )

            return get_return_type(meta)(delayed)

        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return Assigner(self.parent(*args, **kwargs), self.index, subassign=True)

    def update(self, obj):
        if getattr(self.parent, "_is_transposed", False):
            raise TypeError("'TransposedMatrix' object does not support item assignment")

        if is_DOnion(self.parent._delayed):
            self.parent.__setitem__(self.index, obj)
            return

        Assigner(Updater(self.parent), self.index).update(obj)

    def __lshift__(self, rhs):
        self.update(rhs)

    @property
    def value(self):
        self._meta.value
        scalar = self.new()
        return scalar.value


def _uniquify(ndim, index, obj, mask=None, ot=False):
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
    T = (1, 0) if ot else (0, 1)
    unique_indices_tuple = ()
    obj_axis = 0
    for axis in range(ndim):
        indx = index[axis]
        if type(indx) is not slice and not isinstance(indx, Integral):
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
                    obj = extract(obj, obj_indx, T[obj_axis])
                if mask is not None:
                    mask = extract(mask, obj_indx, obj_axis)
            obj_axis += 1
        unique_indices_tuple += (indx,)
    # ---------------- end for loop --------------------------
    return unique_indices_tuple, obj, mask


def _identity_func(x, axis, keepdims):
    return x


class Assigner:
    def __init__(self, updater, index, subassign=False):
        self.updater = updater
        self.parent = updater.parent
        self._meta = updater.parent._meta
        self.subassign = subassign

        input_ndim = self.parent.ndim
        self.keys_0_is_dOnion = input_ndim == 1 and is_DOnion(index)
        self.keys_1_is_dOnion = (
            input_ndim == 2
            and type(index) is tuple
            and len(index) == 2
            and (is_DOnion(index[0]) or is_DOnion(index[1]))
        )
        if self.parent.is_dOnion or self.keys_0_is_dOnion or self.keys_1_is_dOnion:
            IndexerResolver(self.parent, index, check_shape=False)
            self.index = index
        else:
            self.resolved_indices = IndexerResolver(self.parent, index).indices
            self.index = tuple(i.index for i in self.resolved_indices)

    def update(self, obj):
        def setitem(lhs, mask, accum, replace, keys_0, keys_1, obj, ot, subassign, in_dOnion=False):
            keys = (keys_0,) if keys_1 is None else (keys_0, keys_1)
            updater = Updater(lhs, mask=mask, accum=accum, replace=replace)
            Assigner(updater, keys, subassign=subassign).update(obj.T if ot else obj)
            if in_dOnion:
                return lhs

        # check for dOnions:
        lhs = self.parent
        updater = self.updater
        if (
            lhs.is_dOnion
            or updater.mask is not None
            and updater.mask.is_dOnion
            or self.keys_0_is_dOnion
            or self.keys_1_is_dOnion
            or getattr(obj, "is_dOnion", False)
        ):
            lhs_ = lhs.__class__(lhs._delayed, meta=lhs._meta)
            mask = None if updater.mask is None else updater.mask.dOnion_if

            if type(self.index) is tuple and len(self.index) == 2:
                keys_0, keys_1 = self.index[0], self.index[1]
            else:
                keys_0, keys_1 = self.index, None

            donion = DOnion.multiple_access(
                lhs._meta,
                setitem,
                lhs_.dOnion_if,
                mask,
                updater.accum,
                updater.replace,
                keys_0,
                keys_1,
                getattr(obj, "dOnion_if", obj),
                getattr(obj, "is_dOnion", False) and getattr(obj, "_is_transposed", False),
                self.subassign,
                in_dOnion=True,
            )
            lhs.__init__(donion, meta=lhs._meta)
            return

        # no dOnions
        if not (isinstance(obj, BaseType) or isinstance(obj, Number)):
            try:
                obj_transposed = obj._is_transposed
            except AttributeError:
                raise TypeError("Bad type for argument `obj`")
            obj = obj._matrix
        else:
            obj_transposed = False
        obj_is_scalar = not (isinstance(obj, BaseType) and obj.ndim > 0)
        parent = self.parent
        mask = self.updater.mask
        replace = self.updater.replace
        subassign = self.subassign
        indices = self.index
        meta = self._meta
        out_shape = _shape(parent, indices)
        ndim = len(parent.shape)
        out_dim = len(out_shape)
        dtype = np_dtype(meta.dtype)
        if mask is not None:
            if len(out_shape) == 0:
                if ndim > 1:
                    if mask.mask.ndim == 2:
                        raise TypeError("Single element assign does not accept a submask")
                    mask_collection_type = type(mask.mask).__name__
                    parent_type = type(parent).__name__
                    raise TypeError(
                        f"Unable to use {mask_collection_type} mask on single element "
                        f"assignment to a {parent_type}"
                    )
            if subassign:
                if len(out_shape) == 0:
                    if ndim == 1:
                        raise TypeError("Single element assign does not accept a submask")
                if out_dim != mask.mask.ndim:
                    if ndim == 1:
                        raise TypeError(
                            f"Mask object must be type {_get_type_with_ndims(out_dim).__name__}"
                        )
                    elif mask.mask.ndim == 1:
                        raise TypeError(
                            "Unable to use Vector mask on Matrix assignment to a Matrix"
                        )
                    else:
                        raise TypeError(
                            "Indices for subassign imply Vector submask, but got Matrix mask "
                            "instead"
                        )
                if out_shape != mask.mask.shape:
                    raise DimensionMismatch()
            else:
                if parent.shape != mask.mask.shape:
                    if not (ndim == 2 and out_dim == 1):
                        raise DimensionMismatch()
                    else:
                        (rem_axis,) = [
                            axis
                            for axis, index in enumerate(self.resolved_indices)
                            if index.size is not None
                        ]
                        if parent.shape[rem_axis] != out_shape[0]:
                            raise DimensionMismatch()
                else:
                    if ndim == 2 and out_dim == 1:
                        (int_axis,) = [
                            axis
                            for axis, index in enumerate(self.resolved_indices)
                            if index.size is None
                        ]
                        indices = list(indices)
                        indices[int_axis] = [indices[int_axis]]
                        indices = tuple(indices)
                        out_shape = _shape(parent, indices)
                        out_dim = 2
                        if not obj_is_scalar:
                            obj = obj._as_matrix()
                            if int_axis == 0:
                                obj_transposed = True
                            else:
                                obj_transposed = False

            assert isinstance(mask, Mask)
            assert meta.nvals == 0
            delayed_mask = mask.mask._delayed
            grblas_mask_type = get_grblas_type(mask)
        else:
            delayed_mask = None
            grblas_mask_type = None

        if not obj_is_scalar:
            obj_shape = obj.shape[::-1] if obj_transposed else obj.shape
            if len(obj_shape) != out_dim:
                raise TypeError(
                    f"Bad type of RHS in single element assignment.\n"
                    f"    - Expected type: Number or Scalar.\n"
                    f"    - Got: {type(obj)}."
                )
            if out_shape != obj_shape:
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
                    if mask.mask.shape != obj_shape:
                        raise DimensionMismatch()
                    mask_ind = tuple(range(obj._delayed.ndim))
                    obj_ind = mask_ind[::-1] if obj_transposed else mask_ind
                    _, (obj._delayed, delayed_mask) = da.core.unify_chunks(
                        obj._delayed, obj_ind, delayed_mask, mask_ind
                    )
            elif mask is not None:
                # align `x` and its mask in order to fix offsets
                # of `x` before calculating them
                all_ints = np.all(tuple(isinstance(i, Integral) for i in indices))
                mask_ind = tuple(
                    i for i in x_ind if all_ints or not isinstance(indices[i], Integral)
                )
                _, (x, delayed_mask) = da.core.unify_chunks(x, x_ind, delayed_mask, mask_ind)

            indices_args = []
            x_ranges_args = []
            obj_offset_args = []
            obj_offset_axes = ()
            index_axes = ()
            # Note: the blockwise kwarg `new_axes` is non-empty only in one case:
            # where index is a slice and obj is a scalar and there is no mask
            T = (1, 0) if obj_transposed else (0, 1)
            new_axes = dict()
            obj_axis = 0
            for axis in range(ndim):
                indx = indices[axis]
                if type(indx) is da.Array:
                    indices_args += [indx, (ndim + axis,)]
                    index_axes += (ndim + axis,)
                elif type(indx) is slice:
                    s = indx
                    indices_args += [s, None]
                    index_axes += (ndim + axis,)
                    if not obj_is_scalar:
                        obj_offset = build_chunk_offsets_dask_array(
                            obj._delayed, T[obj_axis], "obj_offset-"
                        )
                        obj_offset_args += [obj_offset, (ndim + axis,)]
                        obj_offset_axes += (axis,)
                    elif subassign and mask is not None:
                        if mask.mask.shape[obj_axis] != len(range(s.start, s.stop, s.step)):
                            raise DimensionMismatch()
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
                elif isinstance(indx, Integral):
                    indices_args += [indx, None]
                    obj_axis -= 1
                else:
                    raise NotImplementedError()

                x_ranges = build_chunk_ranges_dask_array(x, axis, "x_ranges-")
                x_ranges_args += [x_ranges, (axis,)]
                obj_axis += 1

            # ---------------- end for loop --------------------------
            if subassign:
                mask_args = [delayed_mask, (index_axes if mask is not None else None)]
            else:
                mask_args = [None, None]

            if not obj_is_scalar:
                dxn = -1 if obj_transposed else 1
                obj_args = [obj._delayed, index_axes[::dxn]]
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
                *(_data_x_index_meshpoint_4assign, fragments_ind),
                *x_ranges_args,
                *indices_args,
                *mask_args,
                *obj_args,
                *obj_offset_args,
                x_ndim=ndim,
                subassign=subassign,
                obj_offset_axes=obj_offset_axes,
                ot=obj_transposed,
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
                    aggregate_func = partial(_uniquify_merged, obj_transposed)
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
            # for band (that is, row or column) assign:
            band_axis = [axis for axis, i in enumerate(indices) if isinstance(i, Integral)]
            is_row_or_col_assign = len(band_axis) == 1
            if is_row_or_col_assign:
                band_selection = [i if isinstance(i, Integral) else slice(None) for i in indices]
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
                ot=obj_transposed,
                dtype=dtype,
                meta=wrap_inner(parent._meta),
            )
            parent.__init__(delayed)
        else:
            raise TypeError("Assignment to Scalars is not supported.")

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


def _reduce_axis(at, op, gb_dtype, x, axis=None, keepdims=None, computing_meta=None, dtype=None):
    """Call reduce_rowwise or reduce_columnwise on each chunk"""
    if computing_meta:
        return np.empty(0, dtype=dtype)
    (axis,) = axis
    T = (1, 0) if at else (0, 1)
    axis = T[axis]
    if axis == 1:
        return wrap_inner(_transpose_if(x, at).reduce_rowwise(op).new(dtype=gb_dtype))
    if axis == 0:
        return wrap_inner(_transpose_if(x, at).reduce_columnwise(op).new(dtype=gb_dtype))


def _reduce_axis_combine(op, x, axis=None, keepdims=None, computing_meta=None, dtype=None):
    """Combine results from _reduce_axis on each chunk"""
    if computing_meta:
        return np.empty(0, dtype=dtype)
    (axis,) = axis
    if type(x) is list:

        def _add_blocks(monoid_, x, y):
            return x.ewise_add(y, monoid_).new()

        vals = [inner.value for inner in x]
        return wrap_inner(reduce(partial(_add_blocks, op), vals))
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


def _transpose_if(inner_x, xt):
    if xt:
        return inner_x.value.T
    return inner_x.value


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
        ot = frag1.ot
        T = (1, 0) if ot else (0, 1)
        if isinstance(obj, gb.base.BaseType) and type(obj) in {gb.Vector, gb.Matrix}:
            concat = da.core.concatenate_lookup.dispatch(type(wrap_inner(obj)))
            obj = concat([wrap_inner(frag1.obj), wrap_inner(frag2.obj)], axis=T[axis]).value
        out.obj = obj
        out.ot = ot

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
            axis for axis, index in enumerate(seq_[0].index) if not isinstance(index, Integral)
        ]
        base_axis = non_singleton_axes[axis]
        return reduce(partial(concatenate_fragments, axis=axis, base_axis=base_axis), seq_)
    else:
        return seq[0]
