from numbers import Number
from collections.abc import Iterable
from tlz import compose
from functools import partial
import dask.array as da
import grblas as gb
import numpy as np
from grblas.operator import UNKNOWN_OPCLASS, find_opclass, get_typed_op

from . import replace as replace_singleton
from .mask import Mask
from .functools import flexible_partial, skip
from .utils import get_grblas_type, get_meta, np_dtype, wrap_inner
from dask.base import is_dask_collection

_expect_type = gb.base._expect_type


def is_type(arg_type, a):
    return type(a) is arg_type


def _check_mask(mask, output=None):
    if not isinstance(mask, Mask):
        if isinstance(mask, BaseType):
            raise TypeError("Mask must indicate values (M.V) or structure (M.S)")
        raise TypeError(f"Invalid mask: {type(mask)}")
    if output is not None:
        from .vector import Vector

        if type(output) is Vector and type(mask.mask) is not Vector:
            raise TypeError(f"Mask object must be type Vector; got {type(mask.mask)}")


class InnerBaseType:
    def astype(self, dtype):
        return wrap_inner(self.value.dup(dtype))


class BaseType:
    _expect_type = _expect_type
    _is_scalar = False

    @property
    def is_dOnion(self):
        return is_DOnion(self._delayed)

    @property
    def dOnion_if(self):
        return self._delayed if self.is_dOnion else self

    def isequal(self, other, *, check_dtype=False):
        from .scalar import PythonScalar

        args = [self, other]
        if np.any([type(arg._delayed) is DOnion for arg in args]):
            args = [arg._delayed if type(arg._delayed) is DOnion else arg for arg in args]
            meta = gb.Scalar.new(bool)
            delayed = DOnion.multiple_access(
                meta, self.__class__.isequal, *args, check_dtype=check_dtype
            )
            return PythonScalar(delayed, meta=meta)

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

        args = [self, other]
        if np.any([type(arg._delayed) is DOnion for arg in args]):
            args = [arg._delayed if type(arg._delayed) is DOnion else arg for arg in args]
            meta = gb.Scalar.new(bool)
            delayed = DOnion.multiple_access(
                meta,
                self.__class__.isclose,
                *args,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                check_dtype=check_dtype,
            )
            return PythonScalar(delayed, meta=meta)

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

    def _clear(self):
        delayed = self._optional_dup()
        # for a function like this, what's the difference between `map_blocks` and `elemwise`?
        if self.ndim == 0:
            return self.__class__(
                delayed.map_blocks(
                    _clear,
                    dtype=np_dtype(self.dtype),
                )
            )
        else:
            return self.__class__(
                delayed.map_blocks(
                    _clear,
                    dtype=np_dtype(self.dtype),
                ),
                nvals=0,
            )

    def clear(self):
        if is_DOnion(self._delayed):
            self.__init__(self._delayed.getattr(self._meta, "_clear"), meta=self._meta, nvals=0)
            return

        # Should we copy and mutate or simply create new chunks?
        delayed = self._optional_dup()
        # for a function like this, what's the difference between `map_blocks` and `elemwise`?
        if self.ndim == 0:
            self._delayed = delayed.map_blocks(
                _clear,
                dtype=np_dtype(self.dtype),
            )
        else:
            self.__init__(
                delayed.map_blocks(
                    _clear,
                    dtype=np_dtype(self.dtype),
                ),
                nvals=0,
            )

    def dup(self, dtype=None, *, mask=None, name=None):
        if is_DOnion(self._delayed):
            mask_meta = mask._meta if mask else None
            meta = self._meta.dup(dtype=dtype, mask=mask_meta, name=name)
            donion = DOnion.multiple_access(
                meta, self.__class__.dup, self._delayed, dtype=dtype, mask=mask, name=name
            )
            return self.__class__(donion, meta=meta)

        if mask and is_DOnion(mask.mask):
            meta = self._meta.dup(dtype=dtype, name=name)
            donion = DOnion.multiple_access(meta, self.dup, dtype=dtype, mask=mask.mask, name=name)
            return self.__class__(donion, meta=meta)

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
        if self.ndim > 0:
            if mask is None or self._nvals == 0:
                nvals = self._nvals
            else:
                nvals = None

            return type(self)(delayed, nvals=nvals)
        else:
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
            _check_mask(mask)
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
    __bool__ = gb.base.BaseType.__bool__
    # TODO: get these to work so we can do things like `gb.op.plus(v | w)`
    __or__ = gb.base.BaseType.__or__
    __ror__ = gb.base.BaseType.__ror__
    __and__ = gb.base.BaseType.__and__
    __rand__ = gb.base.BaseType.__rand__
    __matmul__ = gb.base.BaseType.__matmul__
    __rmatmul__ = gb.base.BaseType.__rmatmul__
    __imatmul__ = gb.base.BaseType.__imatmul__

    def _optional_dup(self):
        # TODO: maybe try to create an optimization pass that remove these if they are unnecessary
        return da.core.elemwise(
            _optional_dup,
            self._delayed,
            dtype=self._delayed.dtype,
        )

    def compute_and_store_nvals(self):
        """
        compute and store the number of values of this Vector/Matrix

        This could be useful to increase the performance of Aggregators
        which inspect ._nvals to determine if a fast path can be taken
        to compute the aggregation result.
        """
        self._nvals = self.nvals.compute()
        return self._nvals

    @property
    def nvals(self):
        from .scalar import PythonScalar

        if type(self._delayed) is DOnion:
            return PythonScalar(self._delayed.nvals)

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

    @property
    def _name_html(self):
        """Treat characters after _ as subscript"""
        split = self.name.split("_", 1)
        if len(split) == 1:
            return self.name
        return f"{split[0]}<sub>{split[1]}</sub>"

    def update(self, expr, in_DOnion=False):
        if isinstance(expr, Number):
            if self.ndim == 2:
                raise TypeError(
                    "Warning: updating a Matrix with a scalar without a mask will "
                    "make the Matrix dense.  This may use a lot of memory and probably "
                    "isn't what you want.  Perhaps you meant:"
                    "\n\n    M(M.S) << s\n\n"
                    "If you do wish to make a dense matrix, then please be explicit:"
                    "\n\n    M[:, :] = s"
                )
        typ = type(expr)
        if (
            self.is_dOnion
            or typ is AmbiguousAssignOrExtract
            and expr.has_dOnion
            or typ is GbDelayed
            and expr.has_dOnion
            or typ is type(self)
            and expr.is_dOnion
            or typ is TransposedMatrix
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
            elif typ is type(self) and expr.is_dOnion:
                expr_ = expr._delayed
            elif typ is TransposedMatrix and expr.is_dOnion:

                def update_T(lhs, rhs):
                    return BaseType.update(lhs, rhs.T, in_DOnion=True)

                donion = DOnion.multiple_access(
                    self._meta, update_T, self_copy.dOnion_if, expr_.dOnion_if
                )
                self.__init__(donion, self._meta)
                return

            donion = DOnion.multiple_access(
                self._meta, BaseType.update, self_copy.dOnion_if, expr_, in_DOnion=True
            )
            self.__init__(donion, self._meta)
            return

        if typ is Box:
            expr = expr.content
            typ = type(expr)

        if isinstance(expr, Number):
            Updater(self)[...] << expr
            if in_DOnion:
                return self.__class__(self._delayed, meta=self._meta)
            return

        self._meta.clear()
        if typ is AmbiguousAssignOrExtract:
            # Extract (w << v[index])
            # Is it safe/reasonable to simply replace `_delayed`?
            # Should we try to preserve e.g. format or partitions?
            self.__init__(expr.new(dtype=self.dtype)._delayed)
        elif typ is type(self):
            # Simple assignment (w << v)
            if self.dtype == expr.dtype:
                self.__init__(expr._optional_dup())
            else:
                self.__init__(expr.dup(dtype=self.dtype)._delayed)
        elif typ is GbDelayed:
            expr._update(self)
        elif typ is TransposedMatrix:
            # "C << A.T"
            C = expr.new(dtype=self.dtype)
            self.__init__(C._delayed)
        else:
            # Anything else we need to handle?
            raise TypeError()
        if in_DOnion:
            return self.__class__(self._delayed, meta=self._meta)

    def _update(self, expr, *, mask=None, accum=None, replace=None, in_DOnion=False):
        typ = type(expr)
        if (
            self.is_dOnion
            or mask is not None
            and mask.is_dOnion
            or typ is AmbiguousAssignOrExtract
            and expr.has_dOnion
            or typ is GbDelayed
            and expr.has_dOnion
            or typ is type(self)
            and expr.is_dOnion
        ):
            self_copy = self.__class__(self._delayed, meta=self._meta)
            mask_ = mask.dOnion_if if mask is not None else None
            expr_ = expr
            if typ is AmbiguousAssignOrExtract and expr.has_dOnion:

                def _update_by_aae(c, p, t, k_0, k_1, mask=None, accum=None, replace=None):
                    p = p.T if t else p
                    keys = k_0 if k_1 is None else (k_0, k_1)
                    aae = AmbiguousAssignOrExtract(p, keys)
                    return c.update(aae, mask=mask, accum=accum, replace=replace, in_DOnion=True)

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
                    _update_by_aae,
                    self_copy.dOnion_if,
                    aae_parent,
                    aae_parent_is_T,
                    *(keys_0, keys_1),
                    mask=mask_,
                    accum=accum,
                    replace=replace,
                )
                self.__init__(donion, self._meta)
                return

            if typ is GbDelayed and expr.has_dOnion:

                def _update_by_gbd(c, t, *args, mask=None, accum=None, replace=None, **kwargs):
                    args = tuple(a.T if xt else a for (xt, a) in zip(t, args))
                    gbd = getattr(args[0], args[1])(*args[2:], **kwargs)
                    return c._update(gbd, mask=mask, accum=accum, replace=replace, in_DOnion=True)

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
                    _update_by_gbd,
                    self_copy.dOnion_if,
                    is_T,
                    *gbd_args,
                    mask=mask_,
                    accum=accum,
                    replace=replace,
                    **gbd_kwargs,
                )
                self.__init__(donion, self._meta)
                return

            elif typ is type(self) and expr.is_dOnion:
                expr_ = expr._delayed

            donion = DOnion.multiple_access(
                self._meta,
                BaseType._update,
                self_copy.dOnion_if,
                expr_,
                mask=mask_,
                accum=accum,
                replace=replace,
                in_DOnion=True,
            )
            self.__init__(donion, meta=self._meta)
            return

        if typ is Box:
            expr = expr.content
            typ = type(expr)

        if mask is None and accum is None:
            self.update(expr)
            if in_DOnion:
                return self
            return
        if typ is AmbiguousAssignOrExtract:
            # Extract (w(mask=mask, accum=accum) << v[index])
            expr_new = expr.new(dtype=self.dtype)
            if expr_new.is_dOnion:
                self_ = self.__class__(self._delayed, meta=self._meta)
                donion = DOnion.multiple_access(
                    self._meta,
                    BaseType._update,
                    self_,
                    expr_new,
                    mask=mask,
                    accum=accum,
                    replace=replace,
                    in_DOnion=True,
                )
                self.__init__(donion, meta=self._meta)
            else:
                expr_delayed = expr_new._delayed
                delayed = self._optional_dup()
                self._meta(mask=get_meta(mask), accum=accum, replace=replace)
                if mask is not None:
                    delayed_mask = mask.mask._delayed
                    grblas_mask_type = get_grblas_type(mask)
                else:
                    delayed_mask = None
                    grblas_mask_type = None
                self.__init__(
                    da.core.elemwise(
                        _update_assign,
                        delayed,
                        accum,
                        delayed_mask,
                        grblas_mask_type,
                        replace,
                        expr_delayed,
                        dtype=np_dtype(self._meta.dtype),
                    )
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
            self.__init__(
                da.core.elemwise(
                    _update_assign,
                    delayed,
                    accum,
                    delayed_mask,
                    grblas_mask_type,
                    replace,
                    expr._delayed,
                    dtype=np_dtype(self._meta.dtype),
                )
            )
        else:
            raise NotImplementedError(f"{typ}")

        if in_DOnion:
            return self.__class__(self._delayed, meta=self._meta)

    def wait(self):
        # TODO: What should this do?
        self._meta.wait()

    def compute(self, *args, **kwargs):
        # kwargs['scheduler'] = 'synchronous'
        val = self._delayed.compute(*args, **kwargs)
        if type(self._delayed) is DOnion:
            return val
        return val.value

    def persist(self, *args, **kwargs):
        return type(self)(self._delayed.persist(*args, **kwargs))

    def visualize(self, *args, **kwargs):
        return self._delayed.visualize(*args, **kwargs)


class Box:
    """
    An arbitrary wrapper to wrap around the inner values of
    an Array object to prevent dask from post-processing the
    Array at the end of compute()
    """

    def __init__(self, content):
        self.content = content

    def __getattr__(self, item):
        return getattr(self.content, item)


const_obj = object()
_const0_DOnion = {"dtype": np.object_, "meta": np.array(const_obj, dtype=np.object_)}


class DOnion:
    """
    Dask (or Delayed) Onion (DOnion):

    Encapsulates a dask array whose inner value is also a dask array.
    Intended to be used in cases where the size of the inner dask
    array (the seed) depends on the inner value of another dask array
    (the shroud)
    """

    is_dOnion = True

    @classmethod
    def sprout(cls, shroud, seed_meta, seed_func, *args, **kwargs):
        """
        Develop a DOnion from dask arrays listed in `shroud` and using function `seed_func`

        Return dask.array.map_blocks(seed_func, shroud) as a DOnion.

        :shroud: a dask array; or an iterable of multiple such dask arrays; or a tuple (x, y)
            where x and y are respectively a list of dask arrays and a dict of named dask arrays.
            The inner values of these arrays determine the (size of) seed dask array
        :seed_meta: empty instance of the inner value type of the seed
        :seed_func: the function that takes as input the inner value of `shroud` and returns
            another dask array (the seed)
        :args: tuple of arguments to `seed_func`.  May contain one or more `skip` sentinels
            denoting a vacant positions to be taken up by the inner values of dask arrays in
            shroud.
        :kwargs: dict of keyword arguments to `seed_func`
        """
        named_shrouds = {}
        if is_dask_collection(shroud):
            shroud = [shroud]
        else:
            if isinstance(shroud, Iterable):
                if len(shroud) > 0:
                    if (
                        len(shroud) == 2
                        and isinstance(shroud[0], Iterable)
                        and isinstance(shroud[1], dict)
                    ):
                        shroud = shroud[0]
                        named_shrouds = shroud[1]
                else:
                    raise ValueError("`shroud` must contain at least one dask array!")
            else:
                raise ValueError(
                    "`shroud` must be a dask array; a list x of dask arrays or"
                    "a dict y of named dask arrays; or a tuple of both: (x, y)"
                )

        seed_func = flexible_partial(seed_func, *args, **kwargs)
        kernel = da.map_blocks(seed_func, *shroud, **named_shrouds, **_const0_DOnion)
        return DOnion(kernel, meta=seed_meta)

    def __init__(self, kernel, meta=None):
        self.kernel = kernel
        # Why ._meta and .dtype? B'cos Scalar, Vector & Matrix need them
        self._meta = meta
        try:
            self.dtype = meta.dtype
        except AttributeError:
            self.dtype = type(meta)

    def __eq__(self, other):
        if type(other) is DOnion:
            other = other.compute()
        return self.compute() == other

    def compute(self, *args, **kwargs):
        value = self.kernel.compute(*args, **kwargs)
        while hasattr(value, "compute"):
            value = value.compute(*args, **kwargs)
        if type(value) is Box:
            value = value.content
        return value

    def compute_once(self, *args, **kwargs):
        value = self.kernel.compute(*args, **kwargs)
        if type(value) is Box:
            value = value.content
        return value

    def persist(self, *args, **kwargs):
        value = self.compute_once(*args, **kwargs)
        while type(value) is DOnion or (
            hasattr(value, "_delayed") and type(value._delayed) is DOnion
        ):
            if type(value) is DOnion:
                value = value.compute_once(*args, **kwargs)
            else:
                value = value._delayed.compute_once(*args, **kwargs)

        if hasattr(value, "persist"):
            return value.persist(*args, **kwargs)
        elif hasattr(value, "_persist") and hasattr(value, "_delayed"):
            value._persist(*args, **kwargs)
            return value._delayed
        else:
            raise TypeError(f'Something went wrong: {self} cannot be "persisted".')

    @classmethod
    def multiple_access(cls, out_meta, func, *args, **kwargs):
        """
        Pass inner values of any DOnions in `args` and/or `kwargs` into `func`.

        :func: Callable that can accept the contents of `args` and/or `kwargs`
            as parameters
        :args: a list of positional arguments to `func`
        :kwargs: a dict of named arguments to `func`
        """
        # First, pass non-DOnion args and kwargs to func:
        skip_Donions = [arg if not is_DOnion(arg) else skip for arg in args]
        non_DOnion_kwargs = {k: v for (k, v) in kwargs.items() if not is_DOnion(v)}
        func = flexible_partial(func, *skip_Donions, **non_DOnion_kwargs)

        # Next, pass func and DOnion args and kwargs to map_blocks:
        donion_args = tuple(arg.kernel for arg in args if is_DOnion(arg))
        donion_kwargs = {k: v.kernel for (k, v) in kwargs.items() if is_DOnion(v)}
        kernel = da.map_blocks(func, *donion_args, **donion_kwargs, **_const0_DOnion)
        return DOnion(kernel, meta=out_meta)

    def deep_extract(self, out_meta, func, *args, **kwargs):
        func = flexible_partial(func, *args, **kwargs)
        if not isinstance(
            out_meta, (np.ndarray, gb.base.BaseType, gb.mask.Mask, gb.matrix.TransposedMatrix)
        ):
            func = compose(Box, func)
        kernel = self.kernel.map_blocks(func, **_const0_DOnion)
        return DOnion(kernel, meta=out_meta)

    def __call__(self, *args, **kwargs):
        meta = self._meta(*args, **kwargs)
        return self.getattr(meta, "__call__", *args, **kwargs)

    def __getattr__(self, item):
        # TODO: how to compute meta of attribute?!!!
        meta = getattr(self._meta, item)
        _getattr = flexible_partial(getattr, skip, item)
        return self.deep_extract(meta, _getattr)

    def getattr(self, meta, attr_name, *args, **kwargs):
        _getattr = flexible_partial(DOnion._getattr, skip, attr_name, *args, **kwargs)
        return self.deep_extract(meta, _getattr)

    @classmethod
    def _getattr(cls, x, attr_name, *args, **kwargs):
        return getattr(x, attr_name)(*args, **kwargs)


is_DOnion = partial(is_type, DOnion)


def like_DOnion(what):
    return (
        is_DOnion(what)
        or isinstance(what, BaseType)
        and is_DOnion(what._delayed)
        or hasattr(what, "_matrix")
        and is_DOnion(what._matrix)
        or hasattr(what, "parent")
        and is_DOnion(what.parent)
        or hasattr(what, "mask")
        and is_DOnion(what.mask)
        or hasattr(what, "_donion")
        and is_DOnion(what._donion)
    )


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
