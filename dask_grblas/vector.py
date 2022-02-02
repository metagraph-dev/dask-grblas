from numbers import Number
import dask.array as da
import numpy as np
import grblas as gb
from dask.base import tokenize
from dask.delayed import Delayed, delayed
from grblas import binary, monoid, semiring

from .base import BaseType, InnerBaseType, _nvals
from .expr import AmbiguousAssignOrExtract, GbDelayed, Updater, Assigner
from .mask import StructuralMask, ValueMask
from .utils import (
    np_dtype,
    get_grblas_type,
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
    ndim = 1

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
        trust_size=False,
        dup_op=None,
        dtype=None,
        chunks="auto",
        name=None,
    ):
        # TODO:
        # dup_op support for dask_array indices/values (use reduce_assign?)
        if dup_op is None and type(indices) is da.Array and type(values) is da.Array:
            if not trust_size or size is None:
                # this branch is an expensive operation:
                implied_size = 1 + da.max(indices).compute()
                if size is not None and implied_size > size:
                    raise Exception()
                size = implied_size if size is None else size

            idtype = gb.Vector.new(indices.dtype).dtype
            np_idtype_ = np_dtype(idtype)
            vdtype = gb.Vector.new(values.dtype).dtype
            np_vdtype_ = np_dtype(vdtype)
            chunks = da.core.normalize_chunks(chunks, (size,), dtype=np_idtype_)
            name_ = name
            name = str(name) if name else ""
            name = name + "-index-ranges" + tokenize(cls, chunks[0])
            index_ranges = build_ranges_dask_array_from_chunks(chunks[0], name)
            fragments = da.core.blockwise(
                *(_pick1D, "ij"),
                *(indices, "j"),
                *(values, "j"),
                *(index_ranges, "i"),
                dtype=np_vdtype_,
                meta=np.array([]),
            )
            meta = InnerVector(gb.Vector.new(vdtype))
            delayed = da.core.blockwise(
                *(_from_values1D, "i"),
                *(fragments, "ij"),
                *(index_ranges, "i"),
                concatenate=False,
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
        if size > 0:
            chunks = da.core.normalize_chunks(chunks, (size,), dtype=int)
            meta = gb.Vector.new(dtype)
            vdtype = meta.dtype
            np_vdtype_ = np_dtype(vdtype)
            chunksz = build_ranges_dask_array_from_chunks(chunks[0], "ranges-" + tokenize(chunks))
            delayed_ = da.map_blocks(
                _new_Vector_chunk, chunksz, gb_dtype=vdtype, dtype=np_vdtype_, meta=InnerVector(meta)
            )
            return Vector(delayed_, nvals=0)

        vector = gb.Vector.new(dtype, size)
        return cls.from_delayed(delayed(vector), vector.dtype, vector.size, nvals=0, name=name)

    def __init__(self, delayed, meta=None, nvals=None):
        # Note: `nvals` is provided here as a parameter mainly for
        # optimization purposes.  A value for `nvals` may be given
        # if it is already known  at the time of initialization of
        # this Vector,  otherwise its value should be left as None
        # (the default)
        assert type(delayed) is da.Array
        assert delayed.ndim == 1
        self._delayed = delayed
        if meta is None:
            meta = gb.Vector.new(delayed.dtype, delayed.shape[0])
        self._meta = meta
        self._size = meta.size
        self.dtype = meta.dtype
        self._nvals = nvals

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
        return self._meta.size

    @property
    def shape(self):
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

    def rechunk(self, inplace=False, chunks="auto"):
        chunks = da.core.normalize_chunks(chunks, self.shape, dtype=np.int64)
        if inplace:
            self.resize(*self.shape, chunks=chunks)
        else:
            return self.resize(*self.shape, chunks=chunks, inplace=False)
        # chunks = da.core.normalize_chunks(chunks, self.shape, dtype=np.int64)
        # id = self.to_values()
        # new = Vector.from_values(*id, *self.shape, trust_size=True, chunks=chunks)
        # if inplace:
        #     self.__init__(new._delayed)
        # else:
        #     return new

    def __getitem__(self, index):
        return AmbiguousAssignOrExtract(self, index)

    def __delitem__(self, keys):
        del Updater(self)[keys]

        # del self._meta[index]
        # delayed = self._optional_dup()
        # TODO: normalize index
        # delayed = delayed.map_blocks(
        #     _delitem,
        #     index,
        #     dtype=np_dtype(self.dtype),
        # )
        # raise NotImplementedError()

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
        assert type(other) is Vector
        meta = self._meta.ewise_add(other._meta, op=op, require_monoid=require_monoid)
        return GbDelayed(self, "ewise_add", other, op, require_monoid=require_monoid, meta=meta)

    def ewise_mult(self, other, op=binary.times):
        assert type(other) is Vector
        meta = self._meta.ewise_mult(other._meta, op=op)
        return GbDelayed(self, "ewise_mult", other, op, meta=meta)

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

        assert type(other) in (Matrix, TransposedMatrix)
        meta = self._meta.vxm(other._meta, op=op)
        return GbDelayed(self, "vxm", other, op, meta=meta)

    def apply(self, op, right=None, *, left=None):
        from .scalar import Scalar

        left_meta = left
        right_meta = right

        if type(left) is Scalar:
            left_meta = left.dtype.np_type(0)
        if type(right) is Scalar:
            right_meta = right.dtype.np_type(0)

        meta = self._meta.apply(op=op, left=left_meta, right=right_meta)
        return GbDelayed(self, "apply", op, right, meta=meta, left=left)

    def reduce(self, op=monoid.plus):
        meta = self._meta.reduce(op)
        return GbDelayed(self, "reduce", op, meta=meta)

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
                raise gb.exceptions.IndexOutOfBound
            indices = da.core.from_array(np.array(indices), name="indices-" + tokenize(indices))
        else:
            if da.max(indices).compute() >= self._size:
                raise gb.exceptions.IndexOutOfBound
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
        x = self._delayed
        nvals_array = da.core.blockwise(
            *(_nvals, "i"), *(x, "i"), adjust_chunks={"i": 1}, dtype=np.int64, meta=np.array([])
        ).compute()

        stops = np.cumsum(nvals_array)
        starts = np.roll(stops, 1)
        starts[0] = 0
        nnz = stops[-1]

        starts = starts.reshape(nvals_array.shape)
        starts = da.from_array(starts, chunks=1, name="starts" + tokenize(starts))
        starts = da.core.Array(starts.dask, starts.name, x.chunks, starts.dtype, meta=x._meta)

        stops = stops.reshape(nvals_array.shape)
        stops = da.from_array(stops, chunks=1, name="stops" + tokenize(stops))
        stops = da.core.Array(stops.dask, stops.name, x.chunks, stops.dtype, meta=x._meta)

        chunks = da.core.normalize_chunks(chunks, (nnz,), dtype=np.int64)
        output_ranges = build_ranges_dask_array_from_chunks(chunks[0], "output_ranges-")

        dtype_ = np_dtype(self.dtype)
        index_offsets = build_chunk_offsets_dask_array(x, 0, "index_offset-")
        x = da.core.blockwise(
            *(VectorTupleExtractor, "ij"),
            *(output_ranges, "j"),
            *(x, "i"),
            *(index_offsets, "i"),
            *(starts, "i"),
            *(stops, "i"),
            gb_dtype=dtype,
            dtype=dtype_,
            meta=np.array([[]]),
        )
        x = da.reduction(
            x, _identity, _flatten, axis=0, concatenate=False, dtype=dtype_, meta=np.array([])
        )

        meta_i, meta_v = self._meta.to_values(dtype)
        indices = da.map_blocks(_get_indices, x, dtype=meta_i.dtype, meta=meta_i)
        values = da.map_blocks(_get_values, x, dtype=meta_v.dtype, meta=meta_v)
        return indices, values

        # delayed = self._delayed
        # dtype_ = np_dtype(self.dtype)
        # meta_i, meta_v = self._meta.to_values(dtype)
        # meta = np.array([])
        # offsets = build_chunk_offsets_dask_array(delayed, 0, "index_offset-")
        # x = da.map_blocks(TupleExtractor, delayed, offsets, gb_dtype=dtype, dtype=dtype_, meta=meta)
        # indices = da.map_blocks(_get_indices, x, dtype=meta_i.dtype, meta=meta)
        # values = da.map_blocks(_get_values, x, dtype=meta_v.dtype, meta=meta)
        # return indices, values

    def isequal(self, other, *, check_dtype=False):
        other = self._expect_type(other, Vector, within="isequal", argname="other")
        return super().isequal(other, check_dtype=check_dtype)

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        other = self._expect_type(other, Vector, within="isclose", argname="other")
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


def _from_values1D(fragments, index_range, gb_dtype=None):
    inds = np.concatenate([inds for (inds, _) in fragments])
    vals = np.concatenate([vals for (_, vals) in fragments])
    size = index_range[0].stop - index_range[0].start
    return InnerVector(gb.Vector.from_values(inds, vals, size=size, dtype=gb_dtype))


def _pick1D(indices, values, index_range):
    index_range = index_range[0]
    indices_in = (index_range.start <= indices) & (indices < index_range.stop)
    indices = indices[indices_in] - index_range.start
    values = values[indices_in]
    return (indices, values)


def _get_indices(tuple_extractor):
    return tuple_extractor.indices


def _get_values(tuple_extractor):
    return tuple_extractor.values


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


gb.utils._output_types[Vector] = gb.Vector
from .matrix import InnerMatrix  # noqa isort:skip
