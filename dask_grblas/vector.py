import dask.array as da
import numpy as np
import grblas as gb

from numbers import Integral
from dask.base import tokenize
from dask.delayed import Delayed, delayed
from grblas import binary, monoid, semiring
from grblas.dtypes import lookup_dtype
from grblas.exceptions import IndexOutOfBound, DimensionMismatch

from . import _automethods
from .base import BaseType, InnerBaseType, _nvals, DOnion, Box, any_dOnions
from .expr import AmbiguousAssignOrExtract, IndexerResolver, GbDelayed, Updater, Assigner
from .mask import StructuralMask, ValueMask
from ._ss.vector import ss
from .utils import (
    pack_args,
    pack_kwargs,
    np_dtype,
    get_return_type,
    wrap_inner,
    build_ranges_dask_array_from_chunks,
    build_chunk_ranges_dask_array,
    build_chunk_offsets_dask_array,
)


class InnerVector(InnerBaseType):
    ndim = 1

    def __init__(self, grblas_vector):
        assert type(grblas_vector) is gb.Vector
        self.value = grblas_vector
        self.size = grblas_vector.size
        self.shape = grblas_vector.shape
        self.dtype = np_dtype(grblas_vector.dtype)

    def __getitem__(self, index):
        # This always copies!
        assert type(index) is tuple
        if len(index) == 1:
            index = index[0]
            value = self.value[index].new()
            return wrap_inner(value)
        elif len(index) == 2:
            if index[0] is None:
                index = index[1]
                if type(index) is int:
                    # [None, 1]
                    value = self.value[[index]].new()
                    return InnerVector(value)
                elif type(index) is slice and index == slice(None):
                    # [None, :]
                    matrix_value = gb.Matrix.new(self.value.dtype, 1, self.value.size)
                    matrix_value[0, :] = self.value
                else:
                    # [None, :5], [None, [1, 2]], etc
                    value = self.value[index].new()
                    matrix_value = gb.Matrix.new(self.value.dtype, 1, value.size)
                    matrix_value[0, :] = value
                return InnerMatrix(matrix_value)
            elif index[1] is None:
                index = index[0]
                if type(index) is int:
                    # [1, None]
                    value = self.value[[index]].new()
                    return InnerVector(value)
                elif type(index) is slice and index == slice(None):
                    # [:, None]
                    # matrix_value = self.value._as_matrix()  # TODO: grblas >=1.3.15
                    matrix_value = gb.Matrix.new(self.value.dtype, self.value.size, 1)
                    matrix_value[:, 0] = self.value
                else:
                    # [:5, None], [[1, 2], None], etc
                    value = self.value[index].new()
                    matrix_value = gb.Matrix.new(self.value.dtype, value.size, 1)
                    matrix_value[:, 0] = value
                return InnerMatrix(matrix_value)
        raise IndexError(f"Too many indices for vector: {index}")


class Vector(BaseType):
    __slots__ = ("ss",)
    ndim = 1
    __abs__ = gb.Vector.__abs__
    __add__ = gb.Vector.__add__
    __divmod__ = gb.Vector.__divmod__
    __eq__ = gb.Vector.__eq__
    __floordiv__ = gb.Vector.__floordiv__
    __ge__ = gb.Vector.__ge__
    __gt__ = gb.Vector.__gt__
    __iadd__ = gb.Vector.__iadd__
    __iand__ = gb.Vector.__iand__
    __ifloordiv__ = gb.Vector.__ifloordiv__
    __imod__ = gb.Vector.__imod__
    __imul__ = gb.Vector.__imul__
    __invert__ = gb.Vector.__invert__
    __ior__ = gb.Vector.__ior__
    __ipow__ = gb.Vector.__ipow__
    __isub__ = gb.Vector.__isub__
    __itruediv__ = gb.Vector.__itruediv__
    __ixor__ = gb.Vector.__ixor__
    __le__ = gb.Vector.__le__
    __lt__ = gb.Vector.__lt__
    __mod__ = gb.Vector.__mod__
    __mul__ = gb.Vector.__mul__
    __ne__ = gb.Vector.__ne__
    __neg__ = gb.Vector.__neg__
    __pow__ = gb.Vector.__pow__
    __radd__ = gb.Vector.__radd__
    __rdivmod__ = gb.Vector.__rdivmod__
    __rfloordiv__ = gb.Vector.__rfloordiv__
    __rmod__ = gb.Vector.__rmod__
    __rmul__ = gb.Vector.__rmul__
    __rpow__ = gb.Vector.__rpow__
    __rsub__ = gb.Vector.__rsub__
    __rtruediv__ = gb.Vector.__rtruediv__
    __rxor__ = gb.Vector.__rxor__
    __sub__ = gb.Vector.__sub__
    __truediv__ = gb.Vector.__truediv__
    __xor__ = gb.Vector.__xor__

    @classmethod
    def from_delayed(cls, vector, dtype, size, *, nvals=None, name=None):
        if not isinstance(vector, Delayed):
            raise TypeError(
                "Value is not a dask delayed object.  "
                "Please use dask.delayed to create a grblas.Vector"
            )
        inner = delayed(InnerVector)(vector)
        value = da.from_delayed(inner, (size,), dtype=np_dtype(dtype), name=name)
        return cls(value, nvals=nvals)

    @classmethod
    def from_vector(cls, vector, *, chunks=None, name=None):
        if not isinstance(vector, gb.Vector):
            raise TypeError("Value is not a grblas.Vector")
        if chunks is not None:
            raise NotImplementedError()
        return cls.from_delayed(delayed(vector), vector.dtype, vector.size, name=name)

    @classmethod
    def from_values(
        cls,
        indices,
        values,
        /,
        size=None,
        *,
        dup_op=None,
        dtype=None,
        chunks="auto",
        name=None,
    ):
        if hasattr(values, "dtype"):
            dtype = lookup_dtype(values.dtype if dtype is None else dtype)

        meta = gb.Vector.new(dtype, size=size if isinstance(size, Integral) else 0)

        # check for any DOnions:
        args = pack_args(indices, values, size)
        kwargs = pack_kwargs(dup_op=dup_op, dtype=dtype, chunks=chunks, name=name)
        if any_dOnions(*args, **kwargs):
            # dive into DOnion(s):
            out_donion = DOnion.multi_access(meta, Vector.from_values, *args, **kwargs)
            return Vector(out_donion, meta=meta)

        # no DOnions
        if type(indices) is da.Array or type(values) is da.Array:
            size_ = size
            if type(indices) in {tuple, list, np.ndarray}:
                size_ = size or (np.max(indices) + 1)
                indices = da.asarray(indices)
            if type(values) in {tuple, list, np.ndarray}:
                values = da.asarray(values)

            np_idtype_ = np_dtype(lookup_dtype(indices.dtype))
            if isinstance(size_, Integral):
                size = size_
                chunks = da.core.normalize_chunks(chunks, (size,), dtype=np_idtype_)
            else:
                if indices.size == 0:
                    raise ValueError("No indices provided. Unable to infer size.")

                if indices.dtype.kind not in "ui":
                    raise ValueError(f"indices must be integers, not {indices.dtype}")

                # Note: uint + int = float which numpy cannot cast to uint.  So we
                # ensure the same dtype for each summand here:
                size = size_
                if size is None:
                    size = da.max(indices) + np.asarray(1, dtype=indices.dtype)
                # Here `size` is a dask 0d-array whose computed value is
                # used to determine the size of the Vector to be returned.
                # But since we do not want to compute anything just now,
                # we instead create a "dOnion" (dask Onion) object.  This
                # effectively means that we will use the inner value of
                # `size` to create the new Vector:
                args = pack_args(indices, values)
                kwargs = pack_kwargs(dup_op=dup_op, dtype=dtype, chunks=chunks, name=name)
                donion = DOnion.sprout(size, meta, Vector.from_values, *args, **kwargs)
                return Vector(donion, meta=meta)

            # output shape `(size,)` is completely determined
            if indices.size > 0:
                if indices.size != values.size:
                    raise ValueError("`indices` and `values` lengths must match")

            vdtype = dtype
            np_vdtype_ = np_dtype(vdtype)

            name_ = name
            name = str(name) if name else ""
            name = name + "-index-ranges" + tokenize(cls, chunks[0])
            index_ranges = build_ranges_dask_array_from_chunks(chunks[0], name)
            fragments = da.core.blockwise(
                *(_pick1D, "ij"),
                *(indices, "j"),
                *(values, "j"),
                *(index_ranges, "i"),
                size=size,
                dtype=np_vdtype_,
                meta=np.array([]),
            )
            meta = InnerVector(gb.Vector.new(vdtype, size=size))
            delayed = da.core.blockwise(
                *(_from_values1D, "i"),
                *(fragments, "ij"),
                *(index_ranges, "i"),
                concatenate=False,
                dup_op=dup_op,
                gb_dtype=dtype,
                dtype=np_vdtype_,
                meta=meta,
                name=name_,
            )
            return Vector(delayed)

        chunks = None
        vector = gb.Vector.from_values(indices, values, size=size, dup_op=dup_op, dtype=dtype)
        return cls.from_vector(vector, chunks=chunks, name=name)

    @classmethod
    def new(cls, dtype, size=0, *, chunks="auto", name=None):
        if any_dOnions(size):
            meta = gb.Vector.new(dtype)
            donion = DOnion.multi_access(meta, cls.new, dtype, size=size, chunks=chunks, name=name)
            return Vector(donion, meta=meta)

        if type(size) is Box:
            size = size.content

        if size > 0:
            chunks = da.core.normalize_chunks(chunks, (size,), dtype=int)
            meta = gb.Vector.new(dtype)
            vdtype = meta.dtype
            np_vdtype_ = np_dtype(vdtype)
            chunksz = build_ranges_dask_array_from_chunks(chunks[0], "ranges-" + tokenize(chunks))
            delayed_ = da.map_blocks(
                _new_Vector_chunk,
                chunksz,
                gb_dtype=vdtype,
                dtype=np_vdtype_,
                meta=InnerVector(meta),
            )
            return Vector(delayed_, nvals=0)

        vector = gb.Vector.new(dtype, size)
        return cls.from_delayed(delayed(vector), vector.dtype, vector.size, nvals=0, name=name)

    def __init__(self, delayed, meta=None, nvals=None):
        # We recommend always using __init__() to set the attribute
        # ._delayed indirectly rather than directly.
        # Note: `nvals` is provided here as a parameter mainly for
        # optimization purposes.  A value for `nvals` may be given
        # if it is already known  at the time of initialization of
        # this Vector,  otherwise its value should be left as None
        # (the default)
        assert type(delayed) in {da.Array, DOnion}
        self._delayed = delayed
        if type(delayed) is da.Array:
            assert delayed.ndim == 1
            if meta is None:
                meta = gb.Vector.new(delayed.dtype, delayed.shape[0])
        else:
            if meta is None:
                meta = gb.Vector.new(delayed.dtype)
        self._meta = meta
        self.dtype = meta.dtype
        self._size = self.size
        self._nvals = nvals
        # Add ss extension methods
        self.ss = ss(self)

    def _as_matrix(self):
        """Cast this Vector to a Matrix (such as a column vector).

        This is SuiteSparse-specific and may change in the future.
        This copies the vector.
        """
        from .matrix import Matrix

        x = self._delayed
        x = x.map_blocks(
            _as_matrix,
            chunks=(x.chunks[0], (1,)),
            new_axis=1,
            dtype=x.dtype,
            meta=InnerMatrix(gb.Matrix.new(self.dtype)),
        )
        return Matrix(x)

    @property
    def S(self):
        return StructuralMask(self)

    @property
    def V(self):
        return ValueMask(self)

    @property
    def size(self):
        if self.is_dOnion:
            return DOnion.multi_access(self._meta.size, getattr, self, "size")
        return self._meta.size

    @property
    def shape(self):
        if self.is_dOnion:
            return (self.size,)
        return self._meta.shape

    def resize(self, size, inplace=True, chunks="auto"):
        chunks = da.core.normalize_chunks(chunks, (size,), dtype=np.int64)
        output_ranges = build_ranges_dask_array_from_chunks(chunks[0], "output_ranges-")

        x = self._optional_dup()
        _meta = x._meta
        dtype_ = np_dtype(self.dtype)
        index_ranges = build_chunk_ranges_dask_array(x, 0, "index_ranges-")
        x = da.core.blockwise(
            *(_resize, "ij"),
            *(output_ranges, "j"),
            *(x, "i"),
            *(index_ranges, "i"),
            old_size=self.size,
            new_size=size,
            dtype=dtype_,
            meta=np.array([[]]),
        )
        x = da.core.blockwise(
            *(_identity, "j"),
            *(x, "ij"),
            concatenate=True,
            dtype=dtype_,
            meta=_meta,
        )

        if size >= self.size:
            nvals = self.nvals
        else:
            nvals = None

        if inplace:
            self.__init__(x, nvals=nvals)
        else:
            return Vector(x, nvals=nvals)

    def diag(self, k=0, dtype=None, chunks="auto"):
        return self._diag(k=k, dtype=dtype, chunks=chunks)

    def _diag(self, k=0, dtype=None, chunks="auto"):
        nrows = self.size + abs(k)
        kdiag_col_start = max(0, k)

        v = self._delayed
        input_ranges = build_chunk_ranges_dask_array(v, 0, "input-ranges-")

        chunks = da.core.normalize_chunks(chunks, (nrows, nrows), dtype=np.int64)
        rname = "-row-ranges" + tokenize(self, chunks[0])
        cname = "-col-ranges" + tokenize(self, chunks[1])
        row_ranges = build_ranges_dask_array_from_chunks(chunks[0], rname)
        col_ranges = build_ranges_dask_array_from_chunks(chunks[1], cname)

        gb_dtype = self.dtype if dtype is None else lookup_dtype(dtype)
        dtype = np_dtype(gb_dtype)
        fragments = da.core.blockwise(
            *(_chunk_diag, "ikj"),
            *(v, "k"),
            *(input_ranges, "k"),
            *(row_ranges, "i"),
            *(col_ranges, "j"),
            k=k,
            kdiag_col_start=kdiag_col_start,
            kdiag_size=self.size,
            gb_dtype=gb_dtype,
            dtype=dtype,
            meta=np.array([[[]]]),
        )
        meta = gb.Matrix.new(gb_dtype)
        delayed = da.reduction(
            fragments,
            _identity,
            _identity,
            axis=1,
            dtype=dtype,
            meta=wrap_inner(meta),
        )

        nvals = 0 if self._nvals == 0 else None
        return get_return_type(meta)(delayed, nvals)

    def rechunk(self, inplace=False, chunks="auto"):
        chunks = da.core.normalize_chunks(chunks, self.shape, dtype=np.int64)
        if inplace:
            self.resize(*self.shape, chunks=chunks)
        else:
            return self.resize(*self.shape, chunks=chunks, inplace=False)

    def __getitem__(self, keys):
        resolved_indexes = IndexerResolver(self, keys)
        shape = resolved_indexes.shape
        if not shape:
            from .scalar import ScalarIndexExpr

            return ScalarIndexExpr(self, resolved_indexes)
        else:
            return VectorIndexExpr(self, resolved_indexes, *shape)

    def __delitem__(self, keys):
        del Updater(self)[keys]

    def __setitem__(self, index, delayed):
        Assigner(Updater(self), index).update(delayed)

    def __contains__(self, index):
        extractor = self[index]
        if not extractor.resolved_indices.is_single_element:
            raise TypeError(
                f"Invalid index to Vector contains: {index!r}.  An integer is expected.  "
                "Doing `index in my_vector` checks whether a value is present at that index."
            )
        scalar = extractor.new(name="s_contains")
        return not scalar.is_empty

    def __iter__(self):
        indices, values = self.to_values()
        return indices.flat

    def ewise_add(self, other, op=monoid.plus, *, require_monoid=True):
        gb_types = (gb.Vector,)
        other = self._expect_type(other, (Vector,) + gb_types, within="ewise_add", argname="other")

        try:
            meta = self._meta.ewise_add(other._meta, op=op, require_monoid=require_monoid)
        except DimensionMismatch:
            if any_dOnions(self, other):
                meta = self._meta.ewise_add(self._meta, op=op, require_monoid=require_monoid)
            else:
                raise

        return VectorExpression(
            self, "ewise_add", other, op, require_monoid=require_monoid, meta=meta
        )

    def ewise_mult(self, other, op=binary.times):
        gb_types = (gb.Vector,)
        other = self._expect_type(other, (Vector,) + gb_types, within="ewise_mult", argname="other")

        try:
            meta = self._meta.ewise_mult(other._meta, op=op)
        except DimensionMismatch:
            if any_dOnions(self, other):
                meta = self._meta.ewise_add(self._meta, op=op)
            else:
                raise

        meta = self._meta.ewise_mult(other._meta, op=op)
        return VectorExpression(self, "ewise_mult", other, op, meta=meta)

    # Unofficial methods
    def inner(self, other, op=semiring.plus_times):
        """
        Vector-vector inner (or dot) product. Result is a Scalar.

        Default op is semiring.plus_times

        *This is not a standard GraphBLAS function*
        """
        pass
        # method_name = "inner"
        # other = self._expect_type(other, Vector, within=method_name, argname="other", op=op)
        # op = get_typed_op(op, self.dtype, other.dtype, kind="semiring")
        # self._expect_op(op, "Semiring", within=method_name, argname="op")
        # expr = ScalarExpression(
        #     method_name,
        #     "GrB_vxm",
        #     [self, _VectorAsMatrix(other)],
        #     op=op,
        # )
        # if self._size != other._size:
        #     expr.new(name="")  # incompatible shape; raise now
        # return expr

    def outer(self, other, op=binary.times):
        """
        Vector-vector outer (or cross) product. Result is a Matrix.

        Default op is binary.times

        *This is not a standard GraphBLAS function*
        """
        pass
        # from .matrix import MatrixExpression
        #
        # method_name = "outer"
        # other = self._expect_type(other, Vector, within=method_name, argname="other", op=op)
        # op = get_typed_op(op, self.dtype, other.dtype, kind="binary")
        # self._expect_op(op, ("BinaryOp", "Monoid"), within=method_name, argname="op")
        # if op.opclass == "Monoid":
        #     op = op.binaryop
        # op = get_semiring(monoid.any, op)
        # expr = MatrixExpression(
        #     method_name,
        #     "GrB_mxm",
        #     [_VectorAsMatrix(self), _VectorAsMatrix(other)],
        #     op=op,
        #     nrows=self._size,
        #     ncols=other._size,
        #     bt=True,
        # )
        # return expr

    def vxm(self, other, op=semiring.plus_times):
        from .matrix import Matrix, TransposedMatrix

        gb_types = (gb.Matrix, gb.matrix.TransposedMatrix)
        other = self._expect_type(
            other, (Matrix, TransposedMatrix) + gb_types, within="vxm", argname="other"
        )
        try:
            meta = self._meta.vxm(other._meta, op=op)
        except DimensionMismatch:
            if any_dOnions(self, other):
                other_meta = gb.Matrix.new(
                    dtype=other._meta.dtype, nrows=self._meta.size, ncols=other._meta.ncols
                )
                meta = self._meta.vxm(other_meta, op=op)
            else:
                raise
        return VectorExpression(self, "vxm", other, op, meta=meta, size=other.ncols)

    def apply(self, op, right=None, *, left=None):
        from .scalar import Scalar

        left_meta = left
        right_meta = right

        if type(left) is Scalar:
            left_meta = left.dtype.np_type(0)
        if type(right) is Scalar:
            right_meta = right.dtype.np_type(0)

        meta = self._meta.apply(op=op, left=left_meta, right=right_meta)
        return VectorExpression(self, "apply", op, right, meta=meta, left=left)

    def reduce(self, op=monoid.plus, *, allow_empty=True):
        from .scalar import ScalarExpression

        meta = self._meta.reduce(op)
        return ScalarExpression(self, "reduce", op, meta=meta, allow_empty=allow_empty)

    def build(self, indices, values, *, size=None, chunks=None, dup_op=None, clear=False):
        if clear:
            self.clear()
        elif self.nvals.compute() > 0:
            raise gb.exceptions.OutputNotEmpty

        if size is None:
            size = self._size
        self.resize(size)

        if chunks is not None:
            self.rechunk(inplace=True, chunks=chunks)

        x = self._optional_dup()
        if type(indices) is list:
            if np.max(indices) >= self._size:
                raise IndexOutOfBound
            indices = da.core.from_array(np.array(indices), name="indices-" + tokenize(indices))
        else:
            if da.max(indices).compute() >= self._size:
                raise IndexOutOfBound
        if type(values) is list:
            values = da.core.from_array(np.array(values), name="values-" + tokenize(values))

        idtype = gb.Matrix.new(indices.dtype).dtype
        np_idtype_ = np_dtype(idtype)
        vdtype = gb.Matrix.new(values.dtype).dtype
        np_vdtype_ = np_dtype(vdtype)

        iname = "-index-ranges" + tokenize(x, x.chunks[0])
        index_ranges = build_chunk_ranges_dask_array(x, 0, iname)
        fragments = da.core.blockwise(
            *(_pick1D, "ij"),
            *(indices, "j"),
            *(values, "j"),
            *(index_ranges, "i"),
            dtype=np_idtype_,
            meta=np.array([[]]),
        )
        meta = InnerVector(gb.Vector.new(vdtype))
        delayed = da.core.blockwise(
            *(_build_1D_chunk, "i"),
            *(x, "i"),
            *(index_ranges, "i"),
            *(fragments, "ij"),
            dup_op=dup_op,
            concatenate=False,
            dtype=np_vdtype_,
            meta=meta,
        )
        self.__init__(delayed)
        # # This doesn't do anything special yet.  Should we have name= and chunks= keywords?
        # # TODO: raise if output is not empty
        # # This operation could, perhaps, partition indices and values if there are chunks
        # vector = gb.Vector.new(self.dtype, size=self.size)
        # vector.build(indices, values, dup_op=dup_op)
        # self.__init__(Vector.from_vector(vector)._delayed)

    def to_values(self, dtype=None, chunks="auto"):
        dtype = lookup_dtype(self.dtype if dtype is None else dtype)
        meta_i, meta_v = self._meta.to_values(dtype)

        x = self._delayed
        if type(x) is DOnion:
            meta = np.array([])
            result = x.getattr(meta, "to_values", dtype=dtype, chunks=chunks)
            indices = result.getattr(meta_i, "__getitem__", 0)
            values = result.getattr(meta_v, "__getitem__", 1)
            return indices, values

        # get dask array of nvals for each chunk:
        nvals_array = da.core.blockwise(
            *(_nvals, "i"), *(x, "i"), adjust_chunks={"i": 1}, dtype=np.int64, meta=np.array([])
        )

        # accumulate dask array to get index-ranges of the output (indices, values)
        stops_ = da.cumsum(nvals_array)  # BEWARE: this function rechunks!
        starts = da.roll(stops_, 1)
        starts = starts.copy() if starts.size == 1 else starts  # bug!!
        starts[0] = 0
        nnz = stops_[-1]
        starts = starts.rechunk(1)
        stops_ = stops_.rechunk(1)

        def _to_values(x, starts, stops_, dtype, chunks, nnz):
            # the following changes the `.chunks` attribute of `starts` and `stops_` so that
            # `blockwise()` can align them with `x`
            starts = da.core.Array(starts.dask, starts.name, x.chunks, starts.dtype, meta=x._meta)
            stops_ = da.core.Array(stops_.dask, stops_.name, x.chunks, stops_.dtype, meta=x._meta)

            chunks = da.core.normalize_chunks(chunks, (nnz,), dtype=np.int64)
            output_ranges = build_ranges_dask_array_from_chunks(chunks[0], "output_ranges-")

            gb_dtype = lookup_dtype(dtype)
            dtype_ = np_dtype(gb_dtype)
            index_offsets = build_chunk_offsets_dask_array(x, 0, "index_offset-")
            x = da.core.blockwise(
                *(VectorTupleExtractor, "ij"),
                *(output_ranges, "j"),
                *(x, "i"),
                *(index_offsets, "i"),
                *(starts, "i"),
                *(stops_, "i"),
                gb_dtype=gb_dtype,
                dtype=dtype_,
                meta=np.array([[]]),
            )
            return da.reduction(
                x, _identity, _flatten, axis=0, concatenate=False, dtype=dtype_, meta=np.array([])
            )

        # since the size of the output (indices, values) depends on nnz, a delayed quantity,
        # we need to return (indices, values) as DOnions (twice-delayed dask-arrays)
        meta = np.array([])
        iv_donion = DOnion.sprout(nnz, meta, _to_values, x, starts, stops_, dtype, chunks)

        dtype_i = np_dtype(lookup_dtype(meta_i.dtype))
        indices = iv_donion.deep_extract(
            meta_i, da.map_blocks, _get_indices, dtype=dtype_i, meta=meta_i
        )
        dtype_v = np_dtype(lookup_dtype(meta_v.dtype))
        values = iv_donion.deep_extract(
            meta_v, da.map_blocks, _get_values, dtype=dtype_v, meta=meta_v
        )
        return indices, values

    def isequal(self, other, *, check_dtype=False):
        other = self._expect_type(other, (Vector, gb.Vector), within="isequal", argname="other")
        return super().isequal(other, check_dtype=check_dtype)

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        other = self._expect_type(other, (Vector, gb.Vector), within="isclose", argname="other")
        return super().isclose(other, rel_tol=rel_tol, abs_tol=abs_tol, check_dtype=check_dtype)

    def _delete_element(self, resolved_indexes):
        idx = resolved_indexes.indices[0]
        delayed = self._optional_dup()
        index_ranges = build_chunk_ranges_dask_array(
            delayed, 0, "index-ranges-" + tokenize(delayed, 0)
        )
        deleted = da.core.blockwise(
            *(_delitem_chunk, "i"),
            *(delayed, "i"),
            *(index_ranges, "i"),
            *(idx.index, None),
            dtype=delayed.dtype,
            meta=delayed._meta,
        )
        self.__init__(deleted)

    @property
    def _carg(self):
        pass
        # return self.gb_obj[0]


Vector.ss = gb.utils.class_property(Vector.ss, ss)


class VectorExpression(GbDelayed):
    __slots__ = ()
    output_type = gb.Vector
    ndim = 1
    _is_scalar = False

    # automethods:
    __and__ = gb.vector.VectorExpression.__and__
    __bool__ = gb.vector.VectorExpression.__bool__
    __or__ = gb.vector.VectorExpression.__or__
    _get_value = _automethods._get_value
    S = gb.vector.VectorExpression.S
    V = gb.vector.VectorExpression.V
    apply = gb.vector.VectorExpression.apply
    ewise_add = gb.vector.VectorExpression.ewise_add
    ewise_mult = gb.vector.VectorExpression.ewise_mult
    isclose = gb.vector.VectorExpression.isclose
    isequal = gb.vector.VectorExpression.isequal
    nvals = gb.vector.VectorExpression.nvals
    reduce = gb.vector.VectorExpression.reduce
    shape = gb.vector.VectorExpression.shape
    size = gb.vector.VectorExpression.size
    vxm = gb.vector.VectorExpression.vxm

    # infix sugar:
    __abs__ = gb.vector.VectorExpression.__abs__
    __add__ = gb.vector.VectorExpression.__add__
    __divmod__ = gb.vector.VectorExpression.__divmod__
    __eq__ = gb.vector.VectorExpression.__eq__
    __floordiv__ = gb.vector.VectorExpression.__floordiv__
    __ge__ = gb.vector.VectorExpression.__ge__
    __gt__ = gb.vector.VectorExpression.__gt__
    __invert__ = gb.vector.VectorExpression.__invert__
    __le__ = gb.vector.VectorExpression.__le__
    __lt__ = gb.vector.VectorExpression.__lt__
    __mod__ = gb.vector.VectorExpression.__mod__
    __mul__ = gb.vector.VectorExpression.__mul__
    __ne__ = gb.vector.VectorExpression.__ne__
    __neg__ = gb.vector.VectorExpression.__neg__
    __pow__ = gb.vector.VectorExpression.__pow__
    __radd__ = gb.vector.VectorExpression.__radd__
    __rdivmod__ = gb.vector.VectorExpression.__rdivmod__
    __rfloordiv__ = gb.vector.VectorExpression.__rfloordiv__
    __rmod__ = gb.vector.VectorExpression.__rmod__
    __rmul__ = gb.vector.VectorExpression.__rmul__
    __rpow__ = gb.vector.VectorExpression.__rpow__
    __rsub__ = gb.vector.VectorExpression.__rsub__
    __rtruediv__ = gb.vector.VectorExpression.__rtruediv__
    __rxor__ = gb.vector.VectorExpression.__rxor__
    __sub__ = gb.vector.VectorExpression.__sub__
    __truediv__ = gb.vector.VectorExpression.__truediv__
    __xor__ = gb.vector.VectorExpression.__xor__

    # bad sugar:
    __itruediv__ = gb.vector.VectorExpression.__itruediv__
    __imul__ = gb.vector.VectorExpression.__imul__
    __imatmul__ = gb.vector.VectorExpression.__imatmul__
    __iadd__ = gb.vector.VectorExpression.__iadd__
    __iand__ = gb.vector.VectorExpression.__iand__
    __ipow__ = gb.vector.VectorExpression.__ipow__
    __imod__ = gb.vector.VectorExpression.__imod__
    __isub__ = gb.vector.VectorExpression.__isub__
    __ixor__ = gb.vector.VectorExpression.__ixor__
    __ifloordiv__ = gb.vector.VectorExpression.__ifloordiv__
    __ior__ = gb.vector.VectorExpression.__ior__

    def __init__(
        self,
        parent,
        method_name,
        *args,
        meta=None,
        size=None,
        **kwargs,
    ):
        super().__init__(
            parent,
            method_name,
            *args,
            meta=meta,
            **kwargs,
        )
        if size is None:
            size = self.parent._size
        self._size = size

    # def __getattr__(self, item):
    #     return getattr(gb.vector.VectorExpression, item)

    # def construct_output(self, dtype=None, *, name=None):
    #     if dtype is None:
    #         dtype = self.dtype
    #     size = 0 if self._size.is_dOnion else self._size
    #     return Vector.new(dtype, size, name=name)


class VectorIndexExpr(AmbiguousAssignOrExtract):
    __slots__ = "_size"
    ndim = 1
    output_type = gb.Vector

    def __init__(self, parent, resolved_indexes, size):
        super().__init__(parent, resolved_indexes)
        self._size = size

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return (self._size,)

    # Begin auto-generated code: Vector
    _get_value = _automethods._get_value
    S = gb.vector.VectorIndexExpr.S
    V = gb.vector.VectorIndexExpr.V
    __and__ = gb.vector.VectorIndexExpr.__and__
    __contains__ = gb.vector.VectorIndexExpr.__contains__
    __or__ = gb.vector.VectorIndexExpr.__or__
    apply = gb.vector.VectorIndexExpr.apply
    ewise_add = gb.vector.VectorIndexExpr.ewise_add
    ewise_mult = gb.vector.VectorIndexExpr.ewise_mult
    isclose = gb.vector.VectorIndexExpr.isclose
    isequal = gb.vector.VectorIndexExpr.isequal
    nvals = gb.vector.VectorIndexExpr.nvals
    reduce = gb.vector.VectorIndexExpr.reduce
    vxm = gb.vector.VectorIndexExpr.vxm

    # infix sugar:
    __abs__ = gb.vector.VectorIndexExpr.__abs__
    __add__ = gb.vector.VectorIndexExpr.__add__
    __divmod__ = gb.vector.VectorIndexExpr.__divmod__
    __eq__ = gb.vector.VectorIndexExpr.__eq__
    __floordiv__ = gb.vector.VectorIndexExpr.__floordiv__
    __ge__ = gb.vector.VectorIndexExpr.__ge__
    __gt__ = gb.vector.VectorIndexExpr.__gt__
    __invert__ = gb.vector.VectorIndexExpr.__invert__
    __le__ = gb.vector.VectorIndexExpr.__le__
    __lt__ = gb.vector.VectorIndexExpr.__lt__
    __mod__ = gb.vector.VectorIndexExpr.__mod__
    __mul__ = gb.vector.VectorIndexExpr.__mul__
    __ne__ = gb.vector.VectorIndexExpr.__ne__
    __neg__ = gb.vector.VectorIndexExpr.__neg__
    __pow__ = gb.vector.VectorIndexExpr.__pow__
    __radd__ = gb.vector.VectorIndexExpr.__radd__
    __rdivmod__ = gb.vector.VectorIndexExpr.__rdivmod__
    __rfloordiv__ = gb.vector.VectorIndexExpr.__rfloordiv__
    __rmod__ = gb.vector.VectorIndexExpr.__rmod__
    __rmul__ = gb.vector.VectorIndexExpr.__rmul__
    __rpow__ = gb.vector.VectorIndexExpr.__rpow__
    __rsub__ = gb.vector.VectorIndexExpr.__rsub__
    __rtruediv__ = gb.vector.VectorIndexExpr.__rtruediv__
    __rxor__ = gb.vector.VectorIndexExpr.__rxor__
    __sub__ = gb.vector.VectorIndexExpr.__sub__
    __truediv__ = gb.vector.VectorIndexExpr.__truediv__
    __xor__ = gb.vector.VectorIndexExpr.__xor__

    # bad sugar:
    __array__ = gb.vector.VectorIndexExpr.__array__
    __bool__ = gb.vector.VectorIndexExpr.__bool__
    __iadd__ = gb.vector.VectorIndexExpr.__iadd__
    __iand__ = gb.vector.VectorIndexExpr.__iand__
    __ifloordiv__ = gb.vector.VectorIndexExpr.__ifloordiv__
    __imatmul__ = gb.vector.VectorIndexExpr.__imatmul__
    __imod__ = gb.vector.VectorIndexExpr.__imod__
    __imul__ = gb.vector.VectorIndexExpr.__imul__
    __ior__ = gb.vector.VectorIndexExpr.__ior__
    __ipow__ = gb.vector.VectorIndexExpr.__ipow__
    __isub__ = gb.vector.VectorIndexExpr.__isub__
    __itruediv__ = gb.vector.VectorIndexExpr.__itruediv__
    __ixor__ = gb.vector.VectorIndexExpr.__ixor__


def _chunk_diag(
    inner_vector,
    input_range,
    row_range,
    col_range,
    k,
    kdiag_col_start,
    kdiag_size,
    gb_dtype,
):
    """
    Return new matrix chunk with shape = (nrows, x)

    where nrows = row_range.stop - row_range.start and
    x is determined by various conditions.

    The returned matrix is either empty or contains a piece of
    the k-diagonal given by inner_vector
    """
    vector = inner_vector.value
    vec_chunk = input_range[0]
    rows = row_range[0]
    cols = col_range[0]
    nrows = rows.stop - rows.start
    ncols = cols.stop - cols.start
    kdiag_col_stop_ = kdiag_col_start + kdiag_size

    # CHANGE REFERENCE POINT: to global matrix row 0 col 0
    kdiag_chunk_col_start = vec_chunk.start + kdiag_col_start
    kdiag_chunk_col_stop_ = vec_chunk.stop + kdiag_col_start

    # intersect matrix chunk column range with k-diagonal chunk column-range
    if cols.start < kdiag_chunk_col_stop_ and kdiag_chunk_col_start < cols.stop:
        out_col_start = max(kdiag_chunk_col_start, cols.start)
        out_col_stop_ = min(kdiag_chunk_col_stop_, cols.stop)

        # where does the diagonal intersect these column bounds?
        # equation of diagonal: i = j - k
        kdiag_nt_out_col_start = out_col_start - k
        kdiag_nt_out_col_stop_ = out_col_stop_ - k

        # is intersection head or tail chunk of vector
        vec_nt_is_head = out_col_start == kdiag_col_start
        vec_nt_is_tail = out_col_stop_ == kdiag_col_stop_

        if vec_nt_is_head:
            # expand to the left
            out_col_start = min(cols.start, out_col_start)
        if vec_nt_is_tail:
            # expand to the right
            out_col_stop_ = max(cols.stop, out_col_stop_)

        ncols = out_col_stop_ - out_col_start
        matrix = gb.Matrix.new(gb_dtype, nrows=nrows, ncols=ncols)

        # return empty matrix if k-diagonal does not touch it
        if rows.stop <= kdiag_nt_out_col_start or kdiag_nt_out_col_stop_ <= rows.start:
            return wrap_inner(matrix)

        # find indices of entry and exit points of k-diagonal
        kdiag_nt_row_start = max(kdiag_nt_out_col_start, rows.start)
        kdiag_nt_row_stop_ = min(kdiag_nt_out_col_stop_, rows.stop)
        # equation of diagonal: j = i + k
        kdiag_nt_col_start = kdiag_nt_row_start + k
        kdiag_nt_col_stop_ = kdiag_nt_row_stop_ + k

        # extract intersecting vector and convert to diagonal matrix:
        # CHANGE REFERENCE POINT: to vector chunk index 0
        vec_nt_start = kdiag_nt_col_start - kdiag_col_start - vec_chunk.start
        vec_nt_stop_ = kdiag_nt_col_stop_ - kdiag_col_start - vec_chunk.start
        vector_nt = vector[vec_nt_start:vec_nt_stop_].new()
        diag_matrix = gb.ss.diag(vector_nt, k=0, dtype=gb_dtype)

        # insert diag_matrix into matrix chunk:
        # destination row index range
        i0 = kdiag_nt_row_start - rows.start
        i1 = kdiag_nt_row_stop_ - rows.start
        # destination column index range
        j0 = kdiag_nt_col_start - out_col_start
        j1 = kdiag_nt_col_stop_ - out_col_start
        matrix[i0:i1, j0:j1] << diag_matrix
        return wrap_inner(matrix)

    width = 0
    if vec_chunk.start == 0:
        if cols.stop <= kdiag_col_start:
            width = ncols
    if vec_chunk.stop == kdiag_size:
        if cols.start >= kdiag_col_stop_:
            width = ncols
    matrix = gb.Matrix.new(gb_dtype, nrows=nrows, ncols=width)
    return wrap_inner(matrix)


def _resize(output_range, inner_vector, index_range, old_size, new_size):
    if output_range[0].start < index_range[0].stop and index_range[0].start < output_range[0].stop:
        start = max(output_range[0].start, index_range[0].start)
        stop = min(output_range[0].stop, index_range[0].stop)
        start = start - index_range[0].start
        stop = stop - index_range[0].start
        if (
            index_range[0].stop == old_size
            and new_size > old_size
            and stop < output_range[0].stop - index_range[0].start
        ):
            new_vec = inner_vector.value[start:stop].new()
            new_vec.resize(output_range[0].stop - index_range[0].start - start)
            return InnerVector(new_vec)
        elif start == 0 and stop == inner_vector.size:
            return inner_vector
        else:
            return InnerVector(inner_vector.value[start:stop].new())
    elif index_range[0].stop == old_size and old_size <= output_range[0].start:
        return InnerVector(
            gb.Vector.new(
                dtype=inner_vector.dtype, size=output_range[0].stop - output_range[0].start
            )
        )
    else:
        return InnerVector(gb.Vector.new(inner_vector.dtype, size=0))


def _as_matrix(x):
    return InnerMatrix(x.value._as_matrix())


def _delitem_chunk(inner_vec, chunk_range, index):
    if isinstance(index, gb.Scalar):
        index = index.value
    if chunk_range[0].start <= index and index < chunk_range[0].stop:
        del inner_vec.value[index - chunk_range[0].start]
    return InnerVector(inner_vec.value)


def _new_Vector_chunk(chunk_range, gb_dtype):
    return InnerVector(gb.Vector.new(gb_dtype, size=chunk_range[0].stop - chunk_range[0].start))


def _build_1D_chunk(inner_vector, out_index_range, fragments, dup_op=None):
    """
    Reassembles filtered tuples (row, val) in the list `fragments`
    obtained from _pick1D() for the chunk within the given index-
    range (`out_index_range`) and returns chunk `inner_vector`
    built using these tuples.
    """
    indices = np.concatenate([indices for (indices, _) in fragments])
    vals = np.concatenate([vals for (_, vals) in fragments])
    size = out_index_range[0].stop - out_index_range[0].start
    inner_vector.value.build(indices, vals, size=size, dup_op=dup_op)
    return InnerVector(inner_vector.value)


def _from_values1D(fragments, index_range, dup_op=None, gb_dtype=None):
    inds = np.concatenate([inds for (inds, _) in fragments])
    vals = np.concatenate([vals for (_, vals) in fragments])
    size = index_range[0].stop - index_range[0].start
    if inds.size == 0:
        return InnerVector(gb.Vector.new(gb_dtype, size=size))
    return InnerVector(gb.Vector.from_values(inds, vals, size=size, dup_op=dup_op, dtype=gb_dtype))


def _pick1D(indices, values, index_range, size):
    # validate indices:
    indices = np.where(indices < 0, indices + size, indices)
    bad_indices = (indices < 0) | (size <= indices)
    if np.any(bad_indices):
        raise IndexOutOfBound

    # filter into chunk:
    index_range = index_range[0]
    indices_in = (index_range.start <= indices) & (indices < index_range.stop)
    indices = indices[indices_in] - index_range.start
    values = values[indices_in]
    return (indices, values)


def _get_indices(tuple_extractor):
    return tuple_extractor.indices


def _get_values(tuple_extractor):
    return tuple_extractor.values


def _identity(chunk, keepdims=None, axis=None):
    return chunk


def _flatten(x, axis=None, keepdims=None):
    if type(x) is list:
        x[0].indices = np.concatenate([y.indices for y in x])
        x[0].values = np.concatenate([y.values for y in x])
        return x[0]
    else:
        return x


class VectorTupleExtractor:
    def __init__(
        self, output_range, inner_vector, index_offset, nval_start, nval_stop, gb_dtype=None
    ):
        self.indices, self.values = inner_vector.value.to_values(gb_dtype)
        if output_range[0].start < nval_stop[0] and nval_start[0] < output_range[0].stop:
            start = max(output_range[0].start, nval_start[0])
            stop = min(output_range[0].stop, nval_stop[0])
            self.indices += index_offset[0]
            start = start - nval_start[0]
            stop = stop - nval_start[0]
            self.indices = self.indices[start:stop]
            self.values = self.values[start:stop]
        else:
            self.indices = np.array([], dtype=self.indices.dtype)
            self.values = np.array([], dtype=self.values.dtype)


class TupleExtractor:
    def __init__(self, grblas_inner_vector, index_offset, gb_dtype=None):
        self.indices, self.values = grblas_inner_vector.value.to_values(gb_dtype)
        self.indices += index_offset[0]


@da.core.concatenate_lookup.register(InnerVector)
def _concat_vector(seq, axis=0):
    if axis != 0:
        raise ValueError(f"Can only concatenate for axis 0.  Got {axis}")
    # return InnerVector(gb.ss.concat([item.value for item in seq]))  # TODO: grblas >=1.3.15
    size = sum(x.size for x in seq)
    value = gb.Vector.new(seq[0].value.dtype, size)
    start = end = 0
    for x in seq:
        end += x.size
        value[start:end] = x.value
        start = end
    return InnerVector(value)


gb.utils._output_types[Vector] = gb.Vector
gb.utils._output_types[VectorExpression] = gb.Vector
gb.utils._output_types[VectorIndexExpr] = gb.Vector
from .matrix import InnerMatrix  # noqa isort:skip
