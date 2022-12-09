import dask.array as da
import graphblas as gb
import numpy as np
from dask.delayed import Delayed, delayed

from .base import BaseType, InnerBaseType
from .expr import AmbiguousAssignOrExtract, GbDelayed
from .utils import get_meta, np_dtype


def from_delayed(cls, scalar, dtype, *, name=None):
    if not isinstance(scalar, Delayed):
        raise TypeError(
            "Value is not a dask delayed object.  "
            "Please use dask.delayed to create a graphblas.Scalar"
        )
    inner = delayed(InnerScalar)(scalar)
    value = da.from_delayed(inner, (), dtype=np_dtype(dtype), name=name)
    return cls(value)


def from_value(cls, scalar, dtype=None, *, name=None):
    if type(scalar) is Scalar:
        if cls is Scalar:
            return scalar.dup(dtype, name=name)
        raise NotImplementedError()
    if type(scalar) is PythonScalar:
        scalar = cls(scalar._delayed, scalar._meta)
        if dtype is not None and scalar.dtype != gb.dtypes.lookup_dtype(dtype):
            scalar = scalar.dup(dtype=dtype)
        return scalar
    if type(scalar) is not gb.Scalar:
        scalar = gb.Scalar.from_value(scalar, dtype=dtype)
    elif dtype is not None and scalar.dtype != gb.dtypes.lookup_dtype(dtype):
        scalar = scalar.dup(dtype=dtype)
    return cls.from_delayed(delayed(scalar), scalar.dtype, name=name)


def new(cls, dtype, *, name=None):
    scalar = gb.Scalar(dtype)
    return cls.from_delayed(delayed(scalar), scalar.dtype, name=name)


class InnerScalar(InnerBaseType):
    ndim = 0
    shape = ()

    def __init__(self, graphblas_scalar):
        self.value = graphblas_scalar
        self.dtype = np_dtype(graphblas_scalar.dtype)


class Scalar(BaseType):
    ndim = 0
    shape = ()
    _is_scalar = True

    @classmethod
    def from_delayed(cls, scalar, dtype, *, name=None):
        return from_delayed(cls, scalar, dtype, name=name)

    @classmethod
    def from_value(cls, value, dtype=None, *, name=None):
        return from_value(cls, value, dtype=dtype, name=name)

    @classmethod
    def new(cls, dtype, *, name=None):
        return new(cls, dtype, name=name)

    def __init__(self, delayed, meta=None):
        assert type(delayed) is da.Array, type(delayed)
        assert delayed.ndim == 0
        self._delayed = delayed
        if meta is None:
            meta = gb.Scalar(delayed.dtype)
        self._meta = meta
        self.dtype = meta.dtype

    def update(self, expr):
        self._meta.update(get_meta(expr))
        self._meta.clear()
        typ = type(expr)
        if typ is AmbiguousAssignOrExtract:
            # Extract (s << v[index])
            self.value = expr.new(dtype=self.dtype).value
        elif typ is Scalar:
            # Simple assignment (s << t)
            self.value = expr.value
        elif typ is GbDelayed:
            # s << v.reduce()
            expr._update(self)
        else:
            # Try simple assignment (s << 1)
            self.value = expr

    def _update(self, delayed, *, accum):
        # s(accum=accum) << v.reduce()
        assert type(delayed) is GbDelayed
        delayed._update(self, accum=accum)

    def dup(self, dtype=None, *, name=None):
        if dtype is None:
            meta = self._meta
        else:
            meta = self._meta.dup(dtype=dtype)
        delayed = da.core.elemwise(
            _scalar_dup,
            self._delayed,
            meta.dtype,
            dtype=np_dtype(meta.dtype),
        )
        return Scalar(delayed)

    def _persist(self, *args, **kwargs):
        """Since scalars are small, persist them if they need to be computed"""
        self._delayed = self._delayed.persist(*args, **kwargs)

    def __eq__(self, other):
        return self.isequal(other).compute()

    def __bool__(self):
        self._persist()
        return bool(self.compute())

    def __float__(self):
        self._persist()
        return float(self.compute())

    def __int__(self):
        self._persist()
        return int(self.compute())

    def __complex__(self):
        self._persist()
        return complex(self.compute())

    __index__ = __int__

    def __neg__(self):
        meta = -self._meta
        delayed = da.core.elemwise(_neg, self._delayed, dtype=self._delayed.dtype)
        return Scalar(delayed, meta=meta)

    def __invert__(self):
        meta = ~self._meta
        delayed = da.core.elemwise(_invert, self._delayed, dtype=bool)
        return Scalar(delayed, meta=meta)

    def __array__(self, dtype=None):
        self._persist()
        if dtype is None:
            dtype = self.dtype.np_type
        return np.array(self.value.compute(), dtype=dtype)

    def isequal(self, other, *, check_dtype=False):
        if other is None:
            return self.is_empty
        if type(other) is not Scalar:
            self._meta.isequal(get_meta(other))
            other = Scalar.from_value(other)
            check_dtype = False
        return super().isequal(other, check_dtype=check_dtype)

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        if other is None:
            return self.is_empty
        if type(other) is not Scalar:
            self._meta.isclose(get_meta(other))
            other = Scalar.from_value(other)
            check_dtype = False
        return super().isclose(other, rel_tol=rel_tol, abs_tol=abs_tol, check_dtype=check_dtype)

    @property
    def is_empty(self):
        delayed = da.core.elemwise(
            _is_empty,
            self._delayed,
            dtype=bool,
        )
        return PythonScalar(delayed)

    def clear(self):
        self._delayed = self.new(self.dtype)._delayed

    def _as_vector(self):
        """Copy this Scalar to a Vector
        In the future, we may _cast_ instead of _copy_ when using SuiteSparse.
        """
        from .vector import Vector

        rv = Vector(self.dtype, size=1)
        if not self.is_empty:
            rv[0] = self
        return rv

    @property
    def value(self):
        return PythonScalar(self._delayed, self._meta)

    @value.setter
    def value(self, val):
        scalar = Scalar.from_value(val, dtype=self.dtype)
        self._delayed = scalar._delayed


class PythonScalar:
    __init__ = Scalar.__init__
    __bool__ = Scalar.__bool__
    __int__ = Scalar.__int__
    __float__ = Scalar.__float__
    __complex__ = Scalar.__complex__
    __index__ = Scalar.__index__
    _persist = Scalar._persist

    @classmethod
    def from_delayed(cls, scalar, dtype, *, name=None):
        return from_delayed(cls, scalar, dtype, name=name)

    @classmethod
    def from_value(cls, scalar, dtype=None, *, name=None):
        return from_value(cls, scalar, dtype=dtype, name=name)

    @classmethod
    def new(cls, dtype, *, name=None):
        return new(cls, dtype, name=None)

    def __eq__(self, other):
        return self.compute() == other

    def compute(self, *args, **kwargs):
        innerval = self._delayed.compute(*args, **kwargs)
        return innerval.value.value


# Dask task functions
def _scalar_dup(x, dtype):
    return InnerScalar(x.value.dup(dtype=dtype))


def _is_empty(x):
    return InnerScalar(gb.Scalar.from_value(x.value.is_empty))


def _neg(x):
    return InnerScalar(-x.value)


def _invert(x):
    return InnerScalar(~x.value)


gb.core.utils._output_types[Scalar] = gb.Scalar
gb.core.utils._output_types[PythonScalar] = gb.Scalar
