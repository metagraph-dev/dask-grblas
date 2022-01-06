import dask.array as da
import grblas as gb
import numpy as np
from grblas.operator import UNKNOWN_OPCLASS, find_opclass, get_typed_op

from . import replace as replace_singleton
from .mask import Mask
from .utils import get_grblas_type, get_meta, np_dtype, wrap_inner

_expect_type = gb.base._expect_type


class InnerBaseType:
    def astype(self, dtype):
        return wrap_inner(self.value.dup(dtype))


class BaseType:
    _expect_type = _expect_type
    _is_scalar = False

    def isequal(self, other, *, check_dtype=False):
        from .scalar import PythonScalar

        # if type(other) is not type(self):
        #     raise TypeError(f'Argument of isequal must be of type {type(self).__name__}')
        if not self._meta.isequal(other._meta):
            return PythonScalar.from_value(False)
        """
        # Alternative to using `elemwise` that has correct metadata
        index = tuple(range(self._delayed.ndim))
        comparisons = da.blockwise(
            _isequal, index,
            self._delayed, index,
            other._delayed, index,
            check_dtype, None,
            dtype=bool,
            token='isequal',
            adjust_chunks={i: 1 for i in range(self._delayed.ndim)},
        )
        """
        delayed = da.core.elemwise(
            _isequal,
            self._delayed,
            other._delayed,
            check_dtype,
            dtype=bool,
        )
        if self._delayed.ndim > 0:
            delayed = da.core.elemwise(
                _to_scalar,
                delayed.all(),
                bool,
            )
        return PythonScalar(delayed)

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        from .scalar import PythonScalar

        # if type(other) is not type(self):
        #     raise TypeError(f'Argument of isclose must be of type {type(self).__name__}')
        if not self._meta.isequal(other._meta):
            return PythonScalar.from_value(False)
        delayed = da.core.elemwise(
            _isclose,
            self._delayed,
            other._delayed,
            rel_tol,
            abs_tol,
            check_dtype,
            dtype=bool,
        )
        if self._delayed.ndim > 0:
            delayed = da.core.elemwise(
                _to_scalar,
                delayed.all(),
                bool,
            )
        return PythonScalar(delayed)

    def clear(self):
        # Should we copy and mutate or simply create new chunks?
        delayed = self._optional_dup()
        # for a function like this, what's the difference between `map_blocks` and `elemwise`?
        self._delayed = delayed.map_blocks(
            _clear,
            dtype=np_dtype(self.dtype),
        )

    def dup(self, dtype=None, *, mask=None, name=None):
        if mask is not None:
            if not isinstance(mask, Mask):
                self._meta.dup(dtype=dtype, mask=mask, name=name)  # should raise
                raise TypeError("Use dask_grblas mask, not a mask from grblas")
            meta = self._meta.dup(dtype=dtype, mask=mask._meta, name=name)
        else:
            meta = self._meta.dup(dtype=dtype, name=name)
        delayed = da.core.elemwise(
            _dup,
            self._delayed,
            mask.mask._delayed if mask is not None else None,
            meta.dtype,
            get_grblas_type(mask) if mask is not None else None,
            dtype=np_dtype(meta.dtype),
        )
        return type(self)(delayed)

    def __lshift__(self, expr):
        self.update(expr)

    def __call__(
        self, *optional_mask_accum_replace, mask=None, accum=None, replace=False, input_mask=None
    ):
        # Pick out mask and accum from positional arguments
        mask_arg = None
        accum_arg = None
        for arg in optional_mask_accum_replace:
            if arg is replace_singleton:
                replace = True
            elif isinstance(arg, (BaseType, Mask)):
                if self._is_scalar:
                    raise TypeError("Mask not allowed for Scalars")
                if mask_arg is not None:
                    raise TypeError("Got multiple values for argument 'mask'")
                mask_arg = arg
            else:
                if accum_arg is not None:
                    raise TypeError("Got multiple values for argument 'accum'")
                accum_arg, opclass = find_opclass(arg)
                if opclass == UNKNOWN_OPCLASS:
                    raise TypeError(f"Invalid item found in output params: {type(arg)}")
        # Merge positional and keyword arguments
        if mask_arg is not None and mask is not None:
            raise TypeError("Got multiple values for argument 'mask'")
        if mask_arg is not None:
            mask = mask_arg
        if accum_arg is not None:
            if accum is not None:
                raise TypeError("Got multiple values for argument 'accum'")
            accum = accum_arg
        if accum is not None:
            # Normalize accumulator
            accum = get_typed_op(accum, self.dtype, kind="binary")
            if accum.opclass == "Monoid":
                accum = accum.binaryop
        return Updater(self, mask=mask, accum=accum, replace=replace, input_mask=input_mask)

    __array__ = gb.base.BaseType.__array__

    def _optional_dup(self):
        # TODO: maybe try to create an optimization pass that remove these if they are unnecessary
        return da.core.elemwise(
            _optional_dup,
            self._delayed,
            dtype=self._delayed.dtype,
        )

    @property
    def nvals(self):
        from .scalar import PythonScalar

        delayed = da.core.elemwise(
            _nvals,
            self._delayed,
            dtype=int,
        )
        if self._delayed.ndim > 0:
            delayed = da.core.elemwise(
                _to_scalar,
                delayed.sum(),
                int,
            )
        return PythonScalar(delayed)

    @property
    def name(self):
        return self._meta.name

    @name.setter
    def name(self, value):
        self._meta.name = value

    def update(self, expr):
        self._meta.update(expr._meta)
        self._meta.clear()
        typ = type(expr)
        if typ is AmbiguousAssignOrExtract:
            # Extract (w << v[index])
            # Is it safe/reasonable to simply replace `_delayed`?
            # Should we try to preserve e.g. format or partitions?
            self._delayed = expr.new(dtype=self.dtype)._delayed
        elif typ is type(self):
            # Simple assignment (w << v)
            if self.dtype == expr.dtype:
                self._delayed = expr._optional_dup()
            else:
                self._delayed = expr.dup(dtype=self.dtype)._delayed
        elif typ is GbDelayed:
            expr._update(self)
        elif typ is TransposedMatrix:
            raise NotImplementedError("C << A.T")
        else:
            # Anything else we need to handle?
            raise TypeError()

    def _update(self, expr, *, mask=None, accum=None, replace=None):
        if mask is None and accum is None:
            self.update(expr)
            return
        typ = type(expr)
        if typ is AmbiguousAssignOrExtract:
            # Extract (w(mask=mask, accum=accum) << v[index])
            delayed = self._optional_dup()
            expr_delayed = expr.new(dtype=self.dtype)._delayed
            self._meta(mask=get_meta(mask), accum=accum, replace=replace)
            if mask is not None:
                delayed_mask = mask.mask._delayed
                grblas_mask_type = get_grblas_type(mask)
            else:
                delayed_mask = None
                grblas_mask_type = None
            self._delayed = da.core.elemwise(
                _update_assign,
                delayed,
                accum,
                delayed_mask,
                grblas_mask_type,
                replace,
                expr_delayed,
                dtype=np_dtype(self._meta.dtype),
            )
        elif typ is GbDelayed:
            # v(mask=mask) << left.ewise_mult(right)
            # Meta check handled in Updater
            expr._update(self, mask=mask, accum=accum, replace=replace)
        elif typ is type(self):
            # v(mask=mask) << other
            delayed = self._optional_dup()
            self._meta(mask=get_meta(mask), accum=accum, replace=replace)
            if mask is not None:
                delayed_mask = mask.mask._delayed
                grblas_mask_type = get_grblas_type(mask)
            else:
                delayed_mask = None
                grblas_mask_type = None
            self._delayed = da.core.elemwise(
                _update_assign,
                delayed,
                accum,
                delayed_mask,
                grblas_mask_type,
                replace,
                expr._delayed,
                dtype=np_dtype(self._meta.dtype),
            )
        else:
            raise NotImplementedError(f"{typ}")

    def wait(self):
        # TODO: What should this do?
        self._meta.wait()

    def compute(self, *args, **kwargs):
        # kwargs['scheduler'] = 'synchronous'
        val = self._delayed.compute(*args, **kwargs)
        return val.value

    def persist(self, *args, **kwargs):
        return type(self)(self._delayed.persist(*args, **kwargs))

    def visualize(self, *args, **kwargs):
        return self._delayed.visualize(*args, **kwargs)


# Dask task functions
def _clear(x):
    x.value.clear()
    return x


def _dup(x, mask, dtype, mask_type):
    if mask is not None:
        mask = mask_type(mask.value)
    return wrap_inner(x.value.dup(dtype=dtype, mask=mask))


def _isclose(x, y, rel_tol, abs_tol, check_dtype):
    val = x.value.isclose(y.value, rel_tol=rel_tol, abs_tol=abs_tol, check_dtype=check_dtype)
    return _reduction_value(x, val)


def _isequal(x, y, check_dtype):
    val = x.value.isequal(y.value, check_dtype=check_dtype)
    return _reduction_value(x, val)


def _nvals(x):
    val = x.value.nvals
    return _reduction_value(x, val)


def _optional_dup(x):
    return wrap_inner(x.value.dup())


def _reduction_value(x, val):
    """Helper function used when reducing objects to scalars such as for `isclose`"""
    if x.ndim == 0:
        return wrap_inner(gb.Scalar.from_value(val))
    elif x.ndim == 1:
        return np.array([val])
    else:
        return np.array([[val]])


def _to_scalar(x, dtype):
    return wrap_inner(gb.Scalar.from_value(x, dtype))


# This mutates the value in `updating`
def _update_assign(updating, accum, mask, mask_type, replace, x):
    if mask is not None:
        mask = mask_type(mask.value)
    updating.value(accum=accum, mask=mask, replace=replace) << x.value
    return updating


from .expr import AmbiguousAssignOrExtract, GbDelayed, Updater  # noqa isort: skip
from .matrix import TransposedMatrix  # noqa isort: skip
