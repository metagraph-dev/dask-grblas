import dask.array as da
import grblas as gb
import numpy as np
from dask.delayed import Delayed, delayed

from . import _automethods
from .base import BaseType, InnerBaseType, DOnion, Box, any_dOnions
from .expr import AmbiguousAssignOrExtract, GbDelayed, _is_pair
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
            # meta = gb.Scalar.from_value(1, dtype=delayed.dtype)
        self._meta = meta
        self.dtype = meta.dtype

    def update(self, expr, in_dOnion=False):
        typ = type(expr)
        if any_dOnions(self, expr):
            self_copy = self.__class__(self._optional_dup(), meta=self._meta)
            expr_ = expr
            if isinstance(expr, AmbiguousAssignOrExtract) and expr.has_dOnion:

                def update_by_aae(c, p, k_0, k_1):
                    keys = k_0 if k_1 is None else (k_0, k_1)
                    return c.update(p[keys], in_dOnion=True)

                if _is_pair(expr_.index):
                    keys_0, keys_1 = expr_.index[0], expr_.index[1]
                else:
                    keys_0, keys_1 = expr_.index, None

                donion = DOnion.multi_access(
                    self._meta,
                    update_by_aae,
                    self_copy,
                    expr_.parent,
                    *(keys_0, keys_1),
                )
                self.__init__(donion, self._meta)
                return

            if isinstance(expr, GbDelayed) and expr.has_dOnion:

                def update_by_gbd(c, *args, **kwargs):
                    gbd = getattr(args[0], args[1])(*args[2:], **kwargs)
                    return c.update(gbd, in_dOnion=True)

                donion = DOnion.multi_access(
                    self._meta,
                    update_by_gbd,
                    self_copy,
                    expr_.parent,
                    expr_.method_name,
                    *expr_.args,
                    **expr_.kwargs,
                )
                self.__init__(donion, self._meta)
                return

            donion = DOnion.multi_access(
                self._meta, Scalar.update, self_copy, expr_, in_dOnion=True
            )
            self.__init__(donion, self._meta)
            return

        if typ is Box:
            expr = expr.content
            typ = type(expr)

        self._meta.update(get_meta(expr))
        self._meta.clear()
        if isinstance(expr, AmbiguousAssignOrExtract):
            # Extract (s << v[index])
            expr_new = expr.new(dtype=self.dtype)
            self.value = expr_new.value
        elif typ is Scalar:
            # Simple assignment (s << t)
            self.value = expr.value
        elif isinstance(expr, GbDelayed):
            # s << v.reduce()
            expr._update(self)
        else:
            # Try simple assignment (s << 1)
            self.value = expr
        if in_dOnion:
            return self.__class__(self._delayed, meta=self._meta)

    def _update(self, expr, *, accum, in_dOnion=False):
        # s(accum=accum) << v.reduce()
        typ = type(expr)
        if typ is Box:
            expr = expr.content

        assert isinstance(expr, GbDelayed)

        if any_dOnions(self, expr):
            self_copy = self.__class__(self._optional_dup(), meta=self._meta)
            expr_ = expr
            if isinstance(expr, GbDelayed) and expr.has_dOnion:

                def _update_by_gbd(c, *args, accum=None, **kwargs):
                    gbd = getattr(args[0], args[1])(*args[2:], **kwargs)
                    return c._update(gbd, accum=accum, in_dOnion=True)

                donion = DOnion.multi_access(
                    self._meta,
                    _update_by_gbd,
                    self_copy,
                    expr_.parent,
                    expr_.method_name,
                    *expr_.args,
                    accum=accum,
                    **expr_.kwargs,
                )
                self.__init__(donion, self._meta)
                return

            expr_ = expr.parent.dOnion_if
            donion = DOnion.mult_access(
                self._meta, Scalar._update, self_copy, expr_, accum=accum, in_dOnion=True
            )
            self.__init__(donion, self._meta)
            return

        expr._update(self, accum=accum)
        if in_dOnion:
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
        if self.is_dOnion:
            self._delayed = self._delayed._persist(*args, **kwargs)
        else:
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
            if other is None:
                return self.is_empty
            self._meta.isequal(get_meta(other))
            try:
                other = Scalar.from_value(other)
            except TypeError:
                other = self._expect_type(
                    other,
                    (Scalar, gb.Scalar),
                    within="isequal",
                    argname="other",
                    extra_message="Literal scalars also accepted.",
                )
            # Don't check dtype if we had to infer dtype of `other`
            check_dtype = False
        if check_dtype and self.dtype != other.dtype:
            return False
        else:
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
        if self.is_dOnion:
            donion = DOnion.multi_access(gb.Scalar.new(bool), getattr, self, "is_empty")
            return PythonScalar(donion)

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
        if any_dOnions(self, val):
            donion = DOnion.multi_access(self._meta, Scalar.from_value, val)
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
    is_dOnion = Scalar.is_dOnion

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
        if self.is_dOnion:
            return innerval.value if hasattr(innerval, "value") else innerval

        return innerval.value.value


class ScalarExpression(GbDelayed):
    __slots__ = ()
    output_type = gb.Scalar
    ndim = 0
    shape = ()
    _is_scalar = True
    _is_cscalar = False
    __and__ = gb.scalar.ScalarExpression.__and__
    __bool__ = gb.scalar.ScalarExpression.__bool__
    __eq__ = gb.scalar.ScalarExpression.__eq__
    __float__ = gb.scalar.ScalarExpression.__float__
    __index__ = gb.scalar.ScalarExpression.__index__
    __int__ = gb.scalar.ScalarExpression.__int__
    __or__ = gb.scalar.ScalarExpression.__or__
    _get_value = _automethods._get_value
    isclose = gb.scalar.ScalarExpression.isclose
    isequal = gb.scalar.ScalarExpression.isequal
    value = gb.scalar.ScalarExpression.value

    # def __getattr__(self, item):
    #     return getattr(gb.scalar.ScalarExpression, item)


class ScalarIndexExpr(AmbiguousAssignOrExtract):
    output_type = gb.Scalar
    ndim = 0
    shape = ()
    _is_scalar = True
    _is_cscalar = False

    dup = new

    @property
    def is_cscalar(self):
        return self._is_cscalar

    @property
    def is_grbscalar(self):
        return not self._is_cscalar

    # Begin auto-generated code: Scalar
    __and__ = gb.scalar.ScalarIndexExpr.__and__
    __bool__ = gb.scalar.ScalarIndexExpr.__bool__
    __eq__ = gb.scalar.ScalarIndexExpr.__eq__
    __float__ = gb.scalar.ScalarIndexExpr.__float__
    __index__ = gb.scalar.ScalarIndexExpr.__index__
    __int__ = gb.scalar.ScalarIndexExpr.__int__
    __or__ = gb.scalar.ScalarIndexExpr.__or__
    _get_value = _automethods._get_value
    isclose = gb.scalar.ScalarIndexExpr.isclose
    isequal = gb.scalar.ScalarIndexExpr.isequal
    value = gb.scalar.ScalarIndexExpr.value
    # End auto-generated code: Scalar


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
gb.utils._output_types[ScalarExpression] = gb.Scalar
gb.utils._output_types[ScalarIndexExpr] = gb.Scalar
