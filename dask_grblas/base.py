from numbers import Number
from functools import partial
from reprlib import recursive_repr
import dask.array as da
import grblas as gb
import numpy as np
from scipy.sparse import csr_matrix
from grblas.operator import UNKNOWN_OPCLASS, find_opclass, get_typed_op
from grblas.dtypes import lookup_dtype

from . import replace as replace_singleton
from .mask import Mask
from .utils import pack_args, pack_kwargs, get_grblas_type, get_meta, np_dtype, wrap_inner

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

    def update(self, expr):
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
            Updater(self)[...] << expr
            return
        self._meta.update(expr._meta)
        self._meta.clear()
        typ = type(expr)
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
            C = expr.new()
            self.__init__(C._delayed)
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


class DOnion:
    """
    Dask (or Delayed) Onion (DOnion):

    Encapsulates a dask array whose inner value is also a dask array.
    Intended to be used in cases where the size of the inner dask
    array (the seed) depends on the inner value of another dask array
    (the shroud)
    """

    @classmethod
    def sprout(cls, shroud, seed_func, seed_meta, packed_args, packed_kwargs, *args, **kwargs):
        """
        Develop a DOnion from dask array `shroud`

        Shroud a dask array (the seed) returned by `seed_func` using another dask array (the
        shroud)
        :shroud: dask array whose inner value determines the (size of) seed dask array
        :seed_func: the function that takes as input the inner value of `shroud` and returns
            another dask array (the seed)
        :seed_meta: empty instance of the inner value type of the seed
        :packed_args: tuple of arguments to `seed_func`
        :packed_kwargs: dict of keyword arguments to `seed_func`
        :args: other dask arrays that together with `shroud` determine the (size of) `seed`
        :kwargs: other named dask arrays that together with `shroud` determine the (size of) `seed`
        """
        seed_func = partial(seed_func, *packed_args, **packed_kwargs)
        dtype = np_dtype(lookup_dtype(shroud.dtype))
        _meta = np.array([], dtype=dtype)
        kernel = shroud.map_blocks(seed_func, *args, **kwargs, dtype=dtype, meta=_meta)
        return DOnion(kernel, meta=seed_meta)

    def __init__(self, kernel, meta=None):
        self.kernel = kernel
        self.dtype = kernel.dtype
        self._meta = meta

    def __eq__(self, other):
        return self.compute() == other

    def compute(self, *args, **kwargs):
        value = self.kernel.compute(*args, **kwargs)
        while hasattr(value, "compute"):
            value = value.compute(*args, **kwargs)
        return value

    def persist(self, *args, **kwargs):
        return self.kernel.compute(*args, **kwargs).persist(*args, **kwargs)

    def extract(self, func, packed_args, packed_kwargs, dtype, meta, *args, **kwargs):
        func = partial(func, *packed_args, **packed_kwargs)
        kernel = self.kernel.map_blocks(func, *args, **kwargs, dtype=dtype, meta=meta)
        return DOnion(kernel, meta=meta)

    @classmethod
    def joint_access(
        cls, func, packed_args, packed_kwargs, dtype, meta
    ):
        """
        Pass inner values of any DOnions in `packed_args` and/or `packed_kwargs` into `func`.
        
        :func: Callable that can accept the contents of `packed_args` and/or `packed_kwargs`
            as parameters 
        :packed_args: a list of positional arguments to `func`
        :packed_kwargs: a dict of named arguments to `func`
        """
        omit_DOnion = is_DOnion
        func = flexible_partial(func, omit_DOnion, *packed_args, **packed_kwargs)
        donion_args = tuple(arg.kernel for arg in packed_args if is_DOnion(arg))
        donion_kwargs = {k: v.kernel for (k, v) in packed_kwargs.items() if is_DOnion(v)}
        kernel = da.map_blocks(func, *donion_args, **donion_kwargs, dtype=dtype, meta=meta)
        return DOnion(kernel, meta=meta)

    def __getattr__(self, item):
        func = partial(getattr, name=item)
        # TODO: lookup dtype and meta of attribute!!!
        dtype = np_dtype(lookup_dtype(self.dtype))
        meta = self._meta
        return self.extract(func, pack_args(), pack_kwargs(), dtype, meta)

    def getattr(self, name, packed_args, packed_kwargs, *args, **kwargs):
        func = partial(DOnion.extractattr, name, packed_args, packed_kwargs)
        # TODO: lookup dtype and meta of attribute!!!
        dtype = np_dtype(lookup_dtype(self.dtype))
        meta = self._meta
        return self.extract(func, pack_args(), pack_kwargs(), dtype, meta, *args, **kwargs)

    @classmethod
    def extractattr(cls, name, packed_args, packed_kwargs, x):
        return getattr(x, name)(*packed_args, **packed_kwargs)


is_DOnion = partial(is_type, DOnion)


class skip:
    def __repr__(self):
        return "skip"
    __str__ = __repr__
    __reduce__ = __repr__  # This makes it pickle well!

skip = skip()


def normalize_occupancies(specs):
    # Converts any valid `specs` to the form: [True, False, True, ...]
    if isinstance(specs, Iterable):
        try:
            a = np.asarray(specs)
        except Exception as e:
            raise e

        if a.ndim == 1:
            if a.dtype is np.bool_:
                return a, None
            
            if a.dtype.kind in np.typecodes["AllInteger"]:
                pos = csr_matrix(np.ones_like(a, dtype=np.bool_), a, np.array([0]))
                return pos.toarray(), None

            occupancy = np.array([False if x is skip else True for x in a])
            args = [x for x in a if x is not skip]
            return occupancy, args

        if a.ndim == 2 and a.shape[1] == 2:
            pos = a[:, 0]
            pos = csr_matrix(np.ones_like(pos, dtype=np.bool_), pos, np.array([0]))
            return pos.toarray(), a[:, 1]

    raise ValueError(
        'specs should be an iterable of any of the following forms:\n'
        '[True, False, True, ...]\n'
        '[0, 2, 3, ...]\n'
        '(skip, b, skip, skip, d, ...)'
        '((1, b), (3, d), ...)'
    )


################################################################################
### flexible_partial() argument application
################################################################################

# Purely functional, no descriptor behaviour
class flexible_partial:
    """New function with flexible partial application of the given
    arguments and keywords.
    """

    __slots__ = "func", "args", "vacancies", "kwargs", "__dict__", "__weakref__"

    def __new__(cls, func, specs, /, *args, **kwargs):
        # Validate input parameters:
        if not callable(func):
            raise TypeError("the first argument must be callable")

        occupancies, args_ = normalize_occupancies(specs)
        args = args if args_ is None else args_
        new_arg = iter(args)

        nfilled = np.count_nonzero(occupancies)
        nargs = len(args)
        if nargs != nfilled:
            raise ValueError(
                f"Number ({nargs}) of given arguments does not match "
                f"number ({nfilled}) of argument slots to be occupied."
            )

        self = super(flexible_partial, cls).__new__(cls)
        
        if hasattr(func, "func") and hasattr(func, "vacancies") and hasattr(func, "kwargs"):
            func = func.func
            self.args = list(func.args)
            kwargs = {**func.kwargs, **kwargs}

            old_vacancy = iter(func.vacancies)
            slot_status = iter(occupancies)

            # step through old vacancies:
            for occupy in occupancies:
                try:
                    pos = next(old_vacancy)
                except StopIteration:
                    # old vacancies now exhausted => reset occupancies and continue elsewhere:
                    occupancies = list(slot_status)
                    break
                else:
                    next(slot_status)

                if occupy:
                    # fill the vacancy:
                    self.args[pos] = next(new_arg)
                else:
                    # record the vacancy:
                    self.vacancies.append(pos)
        else:
            self.args = []
            self.vacancies = []

        start = len(self.args)
        for pos, occupied in enumerate(occupancies, start=start):
            if occupied:
                self.args.append(next(new_arg))
            else:
                # create a vacancy:
                self.args.append(None)
                self.vacancies.append(pos)

        self.func = func
        self.args = tuple(self.args)
        self.vacancies = tuple(self.vacancies)
        self.kwargs = kwargs
        return self

    def __call__(self, /, *args, **kwargs):
        if len(args) < len(self.vacancies):
            raise ValueError(f"Expected at least {len(self.vacancies)} positional arguments. "
                             f"Got {len(args)}.")

        new_arg = iter(args)
        self_args = list(self.args)
        for pos in self.vacancies:
            # fill all the vacancies
            self_args[pos] = next(new_arg)

        # append the remaining arguments and make the call:
        self_args.extend(new_arg)
        kwargs = {**self.kwargs, **kwargs}
        return self.func(*self_args, **kwargs)

    @recursive_repr()
    def __repr__(self):
        qualname = type(self).__qualname__
        args = [repr(self.func)]
        c = count()
        args.extend("_" if next(c) in self.vacancies else repr(x) for x in self.args)
        args.extend(f"{k}={v!r}" for (k, v) in self.kwargs.items())
        if type(self).__module__ == "functools":
            return f"functools.{qualname}({', '.join(args)})"
        return f"{qualname}({', '.join(args)})"

    def __reduce__(self):
        return type(self), (self.func,), (self.func, self.args, self.vacancies,
               self.kwargs or None, self.__dict__ or None)

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 5:
            raise TypeError(f"expected 5 items in state, got {len(state)}")
        func, args, vacs, kwds, namespace = state
        if (not callable(func) or not isinstance(args, tuple) or not isinstance(vacs, tuple) or
           (kwds is not None and not isinstance(kwds, dict)) or
           (namespace is not None and not isinstance(namespace, dict))):
            raise TypeError("invalid flexible_partial state")

        args = tuple(args) # just in case it's a subclass
        vacs = tuple(vacs)
        if kwds is None:
            kwds = {}
        elif type(kwds) is not dict: # XXX does it need to be *exactly* dict?
            kwds = dict(kwds)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self.func = func
        self.args = args
        self.vacancies = vacs
        self.keywords = kwds


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
