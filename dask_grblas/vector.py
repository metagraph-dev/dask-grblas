import dask.array as da
import numpy as np
import grblas as gb
from dask.base import tokenize
from dask.delayed import Delayed, delayed
from grblas import binary, monoid, semiring

from .base import BaseType, InnerBaseType
from .expr import AmbiguousAssignOrExtract, GbDelayed, Updater, Assigner
from .mask import StructuralMask, ValueMask
from .utils import np_dtype, get_grblas_type, wrap_inner, build_ranges_dask_array_from_chunks, build_chunk_offsets_dask_array


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
    @classmethod
    def from_delayed(cls, vector, dtype, size, *, name=None):
        if not isinstance(vector, Delayed):
            raise TypeError(
                "Value is not a dask delayed object.  "
                "Please use dask.delayed to create a grblas.Vector"
            )
        inner = delayed(InnerVector)(vector)
        value = da.from_delayed(inner, (size,), dtype=np_dtype(dtype), name=name)
        return cls(value)

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
        *,
        size=None,
        dup_op=None,
        dtype=None,
        chunks="auto",
        name=None,
    ):
        if type(indices) is da.Array and type(values) is da.Array:
            implied_size = 1 + da.max(indices).compute()
            if size is not None and implied_size > size:
                raise Exception()

            size = implied_size if size is None else size
            if dtype is None:
                dtype = gb.Vector.new(values.dtype, size).dtype
            np_dtype_ = np_dtype(dtype)
            chunks = da.core.normalize_chunks(chunks, (size,), dtype=np_dtype_)
            name_ = name
            name = str(name) if name else ""
            name = name + "-index-ranges" + tokenize(chunks[0])
            index_ranges = build_ranges_dask_array_from_chunks(chunks[0], name)
            fragments = da.core.blockwise(
                *(_pick1D, "ij"),
                *(indices, "j"),
                *(values, "j"),
                *(index_ranges, "i"),
                dtype=np_dtype_,
                meta=np.array([]),
            )
            meta = InnerVector(gb.Vector.new(dtype))
            delayed = da.core.blockwise(
                *(_from_values1D, "i"),
                *(fragments, "ij"),
                *(index_ranges, "i"),
                concatenate=False,
                gb_dtype=dtype,
                dtype=np_dtype_,
                meta=meta,
                name=name_,
            )
            return Vector(delayed)

        vector = gb.Vector.from_values(indices, values, size=size, dup_op=dup_op, dtype=dtype)
        return cls.from_vector(vector, chunks=chunks, name=name)

    @classmethod
    def new(cls, dtype, size=0, *, name=None):
        vector = gb.Vector.new(dtype, size)
        return cls.from_delayed(delayed(vector), vector.dtype, vector.size, name=name)

    def __init__(self, delayed, meta=None):
        assert type(delayed) is da.Array
        assert delayed.ndim == 1
        self._delayed = delayed
        if meta is None:
            meta = gb.Vector.new(delayed.dtype, delayed.shape[0])
        self._meta = meta
        self.dtype = meta.dtype

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

    def resize(self, size):
        self._meta.resize(size)
        raise NotImplementedError()

    def __getitem__(self, index):
        return AmbiguousAssignOrExtract(self, index)

    def __delitem__(self, index):
        del self._meta[index]
        # delayed = self._optional_dup()
        # TODO: normalize index
        # delayed = delayed.map_blocks(
        #     _delitem,
        #     index,
        #     dtype=np_dtype(self.dtype),
        # )
        raise NotImplementedError()

    def __setitem__(self, index, delayed):
        Assigner(Updater(self), index).update(delayed)

    def ewise_add(self, other, op=monoid.plus, *, require_monoid=True):
        assert type(other) is Vector
        meta = self._meta.ewise_add(other._meta, op=op, require_monoid=require_monoid)
        return GbDelayed(self, "ewise_add", other, op, require_monoid=require_monoid, meta=meta)

    def ewise_mult(self, other, op=binary.times):
        assert type(other) is Vector
        meta = self._meta.ewise_mult(other._meta, op=op)
        return GbDelayed(self, "ewise_mult", other, op, meta=meta)

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

    def build(self, indices, values, *, dup_op=None, clear=False):
        # This doesn't do anything special yet.  Should we have name= and chunks= keywords?
        # TODO: raise if output is not empty
        # This operation could, perhaps, partition indices and values if there are chunks
        vector = gb.Vector.new(self.dtype, size=self.size)
        vector.build(indices, values, dup_op=dup_op)
        self._delayed = Vector.from_vector(vector)._delayed

    def to_values(self):
        delayed = self._delayed
        dtype = np_dtype(self.dtype)
        meta_i, meta_v = self._meta.to_values()
        meta = np.array([])
        offsets = build_chunk_offsets_dask_array(delayed, 0, "index_offset-")
        x = da.map_blocks(TupleExtractor, delayed, offsets, dtype=dtype, meta=meta)
        indices = da.map_blocks(_get_indices, x, dtype=meta_i.dtype, meta=meta)
        values = da.map_blocks(_get_values, x, dtype=meta_v.dtype, meta=meta)
        return indices, values

    def isequal(self, other, *, check_dtype=False):
        other = self._expect_type(other, Vector, within="isequal", argname="other")
        return super().isequal(other, check_dtype=check_dtype)

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        other = self._expect_type(other, Vector, within="isclose", argname="other")
        return super().isclose(other, rel_tol=rel_tol, abs_tol=abs_tol, check_dtype=check_dtype)


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


class TupleExtractor:
    def __init__(self, grblas_inner_vector, index_offset):
        self.indices, self.values = grblas_inner_vector.value.to_values()
        self.indices += index_offset[0]


from .matrix import InnerMatrix  # noqa isort:skip
