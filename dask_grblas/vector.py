import dask.array as da
import grblas as gb
from dask.delayed import Delayed, delayed
from grblas import binary, monoid, semiring

from .base import BaseType, InnerBaseType
from .expr import AmbiguousAssignOrExtract, Assigner, GbDelayed, Updater
from .mask import StructuralMask, ValueMask
from .utils import np_dtype, wrap_inner


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
        dtype=None,
        *,
        size=None,
        dup_op=None,
        chunks=None,
        name=None,
    ):
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

    def __delitem__(self, keys):
        del self._meta[keys]
        # delayed = self._optional_dup()
        # TODO: normalize keys
        # delayed = delayed.map_blocks(
        #     _delitem,
        #     keys,
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

    def build(self, indices, values, *, dup_op=None, clear=False, size=None):
        # This doesn't do anything special yet.  Should we have name= and chunks= keywords?
        # TODO: raise if output is not empty
        # This operation could, perhaps, partition indices and values if there are chunks
        if size is None:
            size = self.size
        vector = gb.Vector.new(self.dtype, size=size, name=self.name)
        vector.build(indices, values, dup_op=dup_op)
        v = Vector.from_vector(vector)
        self._delayed = v._delayed
        self._meta = v._meta

    def to_values(self, dtype=None):
        # TODO: make this lazy; can we do something smart with this?
        return self.compute().to_values(dtype=dtype)

    def isequal(self, other, *, check_dtype=False):
        other = self._expect_type(other, Vector, within="isequal", argname="other")
        return super().isequal(other, check_dtype=check_dtype)

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        other = self._expect_type(other, Vector, within="isclose", argname="other")
        return super().isclose(other, rel_tol=rel_tol, abs_tol=abs_tol, check_dtype=check_dtype)


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


from .matrix import InnerMatrix  # noqa isort:skip
