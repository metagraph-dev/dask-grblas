import dask.array as da
import grblas as gb
import numpy as np
from dask.delayed import Delayed, delayed

from .base import BaseType, InnerBaseType, DOnion, Box
from .expr import AmbiguousAssignOrExtract, GbDelayed
from .utils import get_meta, np_dtype


def from_delayed(cls, scalar, dtype, *, name=None):
    if not isinstance(scalar, Delayed):
        raise TypeError(
            "Value is not a dask delayed object.  Please use dask.delayed to create a grblas.Scalar"
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
    scalar = gb.Scalar.new(dtype)
    return cls.from_delayed(delayed(scalar), scalar.dtype, name=name)


class InnerScalar(InnerBaseType):
    ndim = 0
    shape = ()

    def __init__(self, grblas_scalar):
        self.value = grblas_scalar
        self.dtype = np_dtype(grblas_scalar.dtype)


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
        assert type(delayed) in {da.Array, DOnion}, type(delayed)
        self._delayed = delayed
        if type(delayed) is da.Array:
            assert delayed.ndim == 0
        if meta is None:
            meta = gb.Scalar.new(delayed.dtype)
        self._meta = meta
        self.dtype = meta.dtype

    def update(self, expr, in_DOnion=False):
        typ = type(expr)
        if (
            self.is_dOnion
            or typ is AmbiguousAssignOrExtract
            and expr.has_dOnion
            or typ is GbDelayed
            and expr.has_dOnion
            or typ is Scalar
            and expr.is_dOnion
        ):
            self_copy = self.__class__(self._delayed, meta=self._meta)
            expr_ = expr
            if typ is AmbiguousAssignOrExtract and expr.has_dOnion:

                def update_by_aae(c, p, t, k_0, k_1):
                    p = p.T if t else p
                    keys = k_0 if k_1 is None else (k_0, k_1)
                    aae = AmbiguousAssignOrExtract(p, keys)
                    return c.update(aae, in_DOnion=True)

                aae_parent = expr_.parent.dOnion_if
                aae_parent_is_T = expr_.parent.is_dOnion and getattr(
                    expr_.parent, "_is_transposed", False
                )
                if type(expr_.index) is tuple and len(expr_.index) == 2:
                    keys_0, keys_1 = expr_.index[0], expr_.index[1]
                else:
                    keys_0, keys_1 = expr_.index, None

                donion = DOnion.multiple_access(
                    self._meta,
                    update_by_aae,
                    self_copy.dOnion_if,
                    aae_parent,
                    aae_parent_is_T,
                    *(keys_0, keys_1),
                )
                self.__init__(donion, self._meta)
                return

            if typ is GbDelayed and expr.has_dOnion:

                def update_by_gbd(c, t, *args, **kwargs):
                    args = tuple(a.T if xt else a for (xt, a) in zip(t, args))
                    gbd = getattr(args[0], args[1])(*args[2:], **kwargs)
                    return c.update(gbd, in_DOnion=True)

                gbd_parent = expr_.parent.dOnion_if
                gbd_method = expr.method_name
                gbd_args = (gbd_parent, gbd_method) + tuple(
                    getattr(x, "dOnion_if", x) for x in expr.args
                )
                gbd_parent_is_T = expr_.parent.is_dOnion and getattr(
                    expr_.parent, "_is_transposed", False
                )
                is_T = (gbd_parent_is_T, False) + tuple(
                    getattr(x, "is_dOnion", False) and getattr(x, "_is_transposed", False)
                    for x in expr.args
                )
                gbd_kwargs = {k: getattr(v, "dOnion_if", v) for k, v in expr.kwargs.items()}
                donion = DOnion.multiple_access(
                    self._meta,
                    update_by_gbd,
                    self_copy.dOnion_if,
                    is_T,
                    *gbd_args,
                    **gbd_kwargs,
                )
                self.__init__(donion, self._meta)
                return

            elif typ is Scalar and expr.is_dOnion:
                expr_ = expr._delayed
            donion = DOnion.multiple_access(
                self._meta, Scalar.update, self_copy.dOnion_if, expr_, in_DOnion=True
            )
            self.__init__(donion, self._meta)
            return

        if typ is Box:
            expr = expr.content
            typ = type(expr)

        self._meta.update(get_meta(expr))
        self._meta.clear()
        if typ is AmbiguousAssignOrExtract:
            # Extract (s << v[index])
            expr_new = expr.new(dtype=self.dtype)
            self.value = expr_new.value
        elif typ is Scalar:
            # Simple assignment (s << t)
            self.value = expr.value
        elif typ is GbDelayed:
            # s << v.reduce()
            expr._update(self)
        else:
            # Try simple assignment (s << 1)
            self.value = expr
        if in_DOnion:
            return self.__class__(self._delayed, meta=self._meta)

    def _update(self, rhs, *, accum, in_DOnion=False):
        # s(accum=accum) << v.reduce()
        typ = type(rhs)
        if typ is Box:
            rhs = rhs.content

        assert type(rhs) is GbDelayed

        if self.is_dOnion or rhs.parent.is_dOnion:
            self_copy = self.__class__(self._delayed, meta=self._meta)
            self_ = self_copy.dOnion_if
            rhs_ = rhs
            if typ is GbDelayed and rhs.has_dOnion:

                def _update_by_gbd(c, t, *args, accum=None, **kwargs):
                    args = tuple(a.T if xt else a for (xt, a) in zip(t, args))
                    gbd = getattr(args[0], args[1])(*args[2:], **kwargs)
                    return c._update(gbd, accum=accum, in_DOnion=True)

                gbd_parent = rhs_.parent.dOnion_if
                gbd_method = rhs.method_name
                gbd_args = (gbd_parent, gbd_method) + tuple(
                    getattr(x, "dOnion_if", x) for x in rhs.args
                )
                gbd_parent_is_T = rhs_.parent.is_dOnion and getattr(
                    rhs_.parent, "_is_transposed", False
                )
                is_T = (gbd_parent_is_T, False) + tuple(
                    getattr(x, "is_dOnion", False) and getattr(x, "_is_transposed", False)
                    for x in rhs.args
                )
                gbd_kwargs = {k: getattr(v, "dOnion_if", v) for k, v in rhs.kwargs.items()}
                donion = DOnion.multiple_access(
                    self._meta,
                    _update_by_gbd,
                    self_copy.dOnion_if,
                    is_T,
                    *gbd_args,
                    accum=accum,
                    **gbd_kwargs,
                )
                self.__init__(donion, self._meta)
                return

            rhs_ = rhs.parent.dOnion_if
            donion = DOnion.multiple_access(
                self._meta, Scalar._update, self_, rhs_, accum=accum, in_DOnion=True
            )
            self.__init__(donion, self._meta)
            return

        rhs._update(self, accum=accum)
        if in_DOnion:
            return self.__class__(self._delayed, meta=self._meta)

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
        if type(other) is Box:
            other = other.content
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

        rv = Vector.new(self.dtype, size=1)
        if not self.is_empty:
            rv[0] = self
        return rv

    @property
    def value(self):
        return PythonScalar(self._delayed, self._meta)

    @value.setter
    def value(self, val):
        if type(self._delayed) is DOnion:
            donion = DOnion.multiple_access(self._meta, Scalar.from_value, val)
            self.__init__(donion, meta=self._meta)
            return

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
        if type(self._delayed) is DOnion:
            return innerval.value if hasattr(innerval, "value") else innerval

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


gb.utils._output_types[Scalar] = gb.Scalar
gb.utils._output_types[PythonScalar] = gb.Scalar
