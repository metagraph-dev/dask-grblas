import dask.array as da
import grblas as gb
from dask.delayed import Delayed, delayed
from grblas import binary, monoid, semiring

from .base import BaseType, InnerBaseType
from .expr import AmbiguousAssignOrExtract, GbDelayed, Updater
from .mask import StructuralMask, ValueMask
from .utils import np_dtype, wrap_inner


class InnerMatrix(InnerBaseType):
    ndim = 2

    def __init__(self, grblas_matrix):
        assert type(grblas_matrix) is gb.Matrix
        self.value = grblas_matrix
        self.shape = grblas_matrix.shape
        self.dtype = np_dtype(grblas_matrix.dtype)

    def __getitem__(self, index):
        # This always copies!
        assert type(index) is tuple and len(index) == 2
        value = self.value[index].new()
        return wrap_inner(value)


class Matrix(BaseType):
    ndim = 2
    _is_transposed = False

    @classmethod
    def from_delayed(cls, matrix, dtype, nrows, ncols, *, name=None):
        if not isinstance(matrix, Delayed):
            raise TypeError(
                "Value is not a dask delayed object.  "
                "Please use dask.delayed to create a grblas.Matrix"
            )
        inner = delayed(InnerMatrix)(matrix)
        value = da.from_delayed(inner, (nrows, ncols), dtype=np_dtype(dtype), name=name)
        return cls(value)

    @classmethod
    def from_matrix(cls, matrix, chunks=None, *, name=None):
        if not isinstance(matrix, gb.Matrix):
            raise TypeError("Value is not a grblas.Matrix")
        if chunks is not None:
            raise NotImplementedError()
        return cls.from_delayed(delayed(matrix), matrix.dtype, *matrix.shape, name=name)

    @classmethod
    def from_values(
        cls,
        rows,
        columns,
        values,
        *,
        nrows=None,
        ncols=None,
        dup_op=None,
        dtype=None,
        chunks=None,
        name=None,
    ):
        matrix = gb.Matrix.from_values(
            rows, columns, values, nrows=nrows, ncols=ncols, dup_op=dup_op, dtype=dtype
        )
        return cls.from_matrix(matrix, chunks=chunks, name=name)

    @classmethod
    def new(cls, dtype, nrows=0, ncols=0, *, name=None):
        matrix = gb.Matrix.new(dtype, nrows, ncols)
        return cls.from_delayed(
            delayed(matrix), matrix.dtype, matrix.nrows, matrix.ncols, name=name
        )

    def __init__(self, delayed, meta=None):
        assert type(delayed) is da.Array
        assert delayed.ndim == 2
        self._delayed = delayed
        if meta is None:
            meta = gb.Matrix.new(delayed.dtype, *delayed.shape)
        self._meta = meta
        self.dtype = meta.dtype

    @property
    def S(self):
        return StructuralMask(self)

    @property
    def V(self):
        return ValueMask(self)

    @property
    def T(self):
        return TransposedMatrix(self)

    @property
    def nrows(self):
        return self._meta.nrows

    @property
    def ncols(self):
        return self._meta.ncols

    @property
    def shape(self):
        return (self._meta.nrows, self._meta.ncols)

    def resize(self, nrows, ncols):
        self._meta.resize(nrows, ncols)
        raise NotImplementedError()

    def __getitem__(self, index):
        return AmbiguousAssignOrExtract(self, index)

    def __delitem__(self, index):
        raise NotImplementedError()

    def __setitem__(self, index, delayed):
        Updater(self)[index] = delayed

    def ewise_add(self, other, op=monoid.plus, *, require_monoid=True):
        assert type(other) is Matrix  # TODO: or TransposedMatrix
        meta = self._meta.ewise_add(other._meta, op=op, require_monoid=require_monoid)
        return GbDelayed(self, "ewise_add", other, op, require_monoid=require_monoid, meta=meta)

    def ewise_mult(self, other, op=binary.times):
        assert type(other) is Matrix  # TODO: or TransposedMatrix
        meta = self._meta.ewise_mult(other._meta, op=op)
        return GbDelayed(self, "ewise_mult", other, op, meta=meta)

    def mxv(self, other, op=semiring.plus_times):
        from .vector import Vector

        assert type(other) is Vector
        meta = self._meta.mxv(other._meta, op=op)
        return GbDelayed(self, "mxv", other, op, meta=meta)

    def mxm(self, other, op=semiring.plus_times):
        assert type(other) in (Matrix, TransposedMatrix)
        meta = self._meta.mxm(other._meta, op=op)
        return GbDelayed(self, "mxm", other, op, meta=meta)

    def kronecker(self, other, op=binary.times):
        assert type(other) is Matrix  # TODO: or TransposedMatrix
        meta = self._meta.kronecker(other._meta, op=op)
        return GbDelayed(self, "kronecker", other, op, meta=meta)

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

    def reduce_rowwise(self, op=monoid.plus):
        meta = self._meta.reduce_rowwise(op)
        return GbDelayed(self, "reduce_rowwise", op, meta=meta)

    def reduce_columnwise(self, op=monoid.plus):
        meta = self._meta.reduce_columnwise(op)
        return GbDelayed(self, "reduce_columnwise", op, meta=meta)

    def reduce_scalar(self, op=monoid.plus):
        meta = self._meta.reduce_scalar(op)
        return GbDelayed(self, "reduce_scalar", op, meta=meta)

    def build(self, rows, columns, values, *, dup_op=None, clear=False, nrows=None, ncols=None):
        # This doesn't do anything special yet.  Should we have name= and chunks= keywords?
        # TODO: raise if output is not empty
        # This operation could, perhaps, partition rows, columns, and values if there are chunks
        if nrows is None:
            nrows = self.nrows
        if ncols is None:
            ncols = self.ncols
        matrix = gb.Matrix.new(self.dtype, nrows, ncols)
        matrix.build(rows, columns, values, dup_op=dup_op)
        m = Matrix.from_matrix(matrix)
        self._delayed = m._delayed
        self._meta = m._meta

    def to_values(self, dtype=None):
        # TODO: make this lazy; can we do something smart with this?
        return self.compute().to_values(dtype=dtype)

    def isequal(self, other, *, check_dtype=False):
        other = self._expect_type(
            other, (Matrix, TransposedMatrix), within="isequal", argname="other"
        )
        return super().isequal(other, check_dtype=check_dtype)

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        other = self._expect_type(
            other, (Matrix, TransposedMatrix), within="isclose", argname="other"
        )
        return super().isclose(other, rel_tol=rel_tol, abs_tol=abs_tol, check_dtype=check_dtype)


class TransposedMatrix:
    ndim = 2
    _is_transposed = True

    def __init__(self, matrix):
        assert type(matrix) is Matrix
        self._matrix = matrix
        self._meta = matrix._meta.T

    def new(self, *, dtype=None, mask=None):
        raise NotImplementedError("A.T.new()")

    @property
    def T(self):
        return self._matrix

    @property
    def dtype(self):
        return self._meta.dtype

    def to_values(self):
        # TODO: make this lazy; can we do something smart with this?
        rows, cols, vals = self._matrix.to_values()
        return cols, rows, vals

    # Properties
    nrows = Matrix.ncols
    ncols = Matrix.nrows
    shape = Matrix.shape
    nvals = Matrix.nvals

    # Delayed methods
    ewise_add = Matrix.ewise_add
    ewise_mult = Matrix.ewise_mult
    mxv = Matrix.mxv
    mxm = Matrix.mxm
    kronecker = Matrix.kronecker
    apply = Matrix.apply
    reduce_rowwise = Matrix.reduce_rowwise
    reduce_columnwise = Matrix.reduce_columnwise
    reduce_scalar = Matrix.reduce_scalar

    # Misc.
    isequal = Matrix.isequal
    isclose = Matrix.isclose
    __getitem__ = Matrix.__getitem__
    __array__ = Matrix.__array__
    name = Matrix.name


@da.core.concatenate_lookup.register(InnerMatrix)
def _concat_matrix(seq, axis=0):
    if axis not in {0, 1}:
        raise ValueError(f"Can only concatenate for axis 0 or 1.  Got {axis}")
    if axis == 0:
        value = gb.ss.concat([[item.value] for item in seq])
    else:
        value = gb.ss.concat([[item.value for item in seq]])
    return InnerMatrix(value)
