import dask.array as da
import grblas as gb
from dask.delayed import delayed, Delayed
from .base import BaseType, InnerBaseType
from .expr import AmbiguousAssignOrExtract, GbDelayed
from .utils import np_dtype, get_meta


def from_delayed(cls, scalar, dtype, *, name=None):
    if not isinstance(scalar, Delayed):
        raise TypeError(
            "Value is not a dask delayed object.  Please use dask.delayed to create a grblas.Scalar"
        )
    inner = delayed(InnerScalar)(scalar)
    value = da.from_delayed(inner, (), dtype=np_dtype(dtype), name=name)
    return cls(value)


def from_value(cls, scalar, dtype=None, *, name=None):
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
    scalar = gb.Scalar.new(dtype)
    return cls.from_delayed(delayed(scalar), scalar.dtype, name=name)


class InnerScalar(InnerBaseType):
    ndim = 0
    shape = ()

    def __init__(self, grblas_scalar):
        self.value = grblas_scalar
        self.dtype = np_dtype(grblas_scalar.dtype)


class Scalar(BaseType):
    @classmethod
    def from_delayed(cls, scalar, dtype, *, name=None):
        return from_delayed(cls, scalar, dtype, name=name)

    @classmethod
    def from_value(cls, scalar, dtype=None, *, name=None):
        return from_value(cls, scalar, dtype=dtype, name=name)

    @classmethod
    def new(cls, dtype, *, name=None):
        return new(cls, dtype, name=None)

    def __init__(self, delayed, meta=None):
        assert type(delayed) is da.Array
        assert delayed.ndim == 0
        self._delayed = delayed
        if meta is None:
            meta = gb.Scalar.new(delayed.dtype)
        self._meta = meta
        self.dtype = meta.dtype

    def update(self, delayed):
        self._meta.update(get_meta(delayed))
        self._meta.clear()
        typ = type(delayed)
        if typ is AmbiguousAssignOrExtract:
            # Extract (s << v[index])
            self.value = delayed.new(dtype=self.dtype).value
        elif typ is Scalar:
            # Simple assignment (s << t)
            self.value = delayed.value
        elif typ is GbDelayed:
            # s << v.reduce()
            delayed._update(self)
        else:
            # Try simple assignment (s << 1)
            self.value = delayed

    def _update(self, delayed, *, accum):
        # s(accum=accum) << v.reduce()
        assert type(delayed) is GbDelayed
        delayed._update(self, accum=accum)

    def dup(self, dtype=None):
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

    def __eq__(self, other):
        return self.isequal(other).compute()

    def __bool__(self):
        return bool(self.compute())

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
        return super().isclose(
            other, rel_tol=rel_tol, abs_tol=abs_tol, check_dtype=check_dtype
        )

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
    # __int__?
    # __float__?

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
        return Scalar.from_value(self) == other

    def compute(self, *args, **kwargs):
        innerval = self._delayed.compute(*args, **kwargs)
        return innerval.value.value


# Dask task functions
def _scalar_dup(x, dtype):
    return InnerScalar(x.value.dup(dtype=dtype))


def _is_empty(x):
    return InnerScalar(gb.Scalar.from_value(x.value.is_empty))
