import numpy as np

from reprlib import recursive_repr
from collections.abc import Iterable
from itertools import count
from scipy.sparse import csr_matrix


class skip:
    def __repr__(self):
        return "skip"

    __str__ = __repr__
    __reduce__ = __repr__  # This makes it pickle well!


skip = skip()


def normalize_occupancies(specs):
    """
    Convert any valid `specs` into the form: [True, False, True, ...]
    """
    error_msg = (
        "`specs` should be array-like with structure matching any of the following forms:\n"
        "[False, True, False, False, True, ...]\n"
        "[1, 4, ...]\n"
        "(skip, b, skip, skip, d, ...)\n"
        "((1, b), (4, d), ...)"
    )
    if isinstance(specs, Iterable):
        try:
            a = np.asarray(specs)
        except Exception:
            raise ValueError(error_msg)

        if a.ndim == 1:
            if a.dtype == np.bool_:
                # specs = [False, True, False, False, True, ...]
                return a, None

            if a.dtype.kind in np.typecodes["AllInteger"]:
                # specs = (1, 4, ...)
                data = np.ones_like(a, dtype=np.bool_)
                indices = a
                indptr = np.array([0, data.size])
                occupancy = csr_matrix((data, indices, indptr)).toarray().squeeze()
                return occupancy, None

            # specs = (skip, b, skip, skip, d, ...)
            occupancy = np.array([False if x is skip else True for x in a])
            args = [x for x in a if x is not skip]
            return occupancy, args

        if a.ndim == 2 and a.shape[1] == 2:
            # specs = [(1, b), (4, d), ...]
            indices = np.asarray(a[:, 0], dtype=int)
            args = a[:, 1]

            data = np.ones_like(indices, dtype=np.bool_)
            indptr = np.array([0, data.size])
            occupancy = csr_matrix((data, indices, indptr)).toarray().squeeze()
            return occupancy, args

        raise ValueError(error_msg)

    raise TypeError(error_msg)


################################################################################
### flexible_partial() argument application  # noqa
################################################################################

# Purely functional, no descriptor behaviour
class flexible_partial:
    """New function with flexible partial application of the given
    arguments and keywords. (Any argument slot of the given function
    may be occupied.)
    """

    __slots__ = "base_func", "args", "vacancies", "kwargs", "__dict__", "__weakref__"

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

        _func = None
        if hasattr(func, "base_func"):
            # `func` is `flexible_partial`
            _func = func.base_func
        elif hasattr(func, "func"):
            # `func` is `partial`
            _func = func.func
        if _func and hasattr(func, "args") and hasattr(func, "kwargs"):
            self.args = list(func.args)
            kwargs = {**func.kwargs, **kwargs}

            if hasattr(func, "vacancies"):
                old_vacancy = iter(func.vacancies)
            else:
                old_vacancy = iter([])

            func = _func

            # step through old vacancies:
            occupancy = iter(occupancies)
            self.vacancies = []
            for occupy in occupancies:
                try:
                    pos = next(old_vacancy)
                except StopIteration:
                    # inner vacancies now exhausted => continue elsewhere:
                    break
                else:
                    next(occupancy)

                if occupy:
                    # fill the vacancy:
                    self.args[pos] = next(new_arg)
                else:
                    # record the vacancy:
                    self.vacancies.append(pos)

            # reset to remaining occupancies
            occupancies = list(occupancy)
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

        self.base_func = func
        self.args = tuple(self.args)
        self.vacancies = tuple(self.vacancies)
        self.kwargs = kwargs
        return self

    def __call__(self, /, *args, **kwargs):
        if len(args) < len(self.vacancies):
            raise ValueError(
                f"Expected at least {len(self.vacancies)} positional arguments. "
                f"Got {len(args)}."
            )

        new_arg = iter(args)
        self_args = list(self.args)
        for pos in self.vacancies:
            # fill all the vacancies
            self_args[pos] = next(new_arg)

        # append the remaining arguments and make the call:
        self_args.extend(new_arg)
        kwargs = {**self.kwargs, **kwargs}
        return self.base_func(*self_args, **kwargs)

    @recursive_repr()
    def __repr__(self):
        qualname = type(self).__qualname__
        args = [repr(self.base_func)]
        c = count()
        args.extend("_" if next(c) in self.vacancies else repr(x) for x in self.args)
        args.extend(f"{k}={v!r}" for (k, v) in self.kwargs.items())
        if type(self).__module__ == "functools":
            return f"functools.{qualname}({', '.join(args)})"
        return f"{qualname}({', '.join(args)})"

    def __reduce__(self):
        return (
            type(self),
            (self.base_func,),
            (self.base_func, self.args, self.vacancies, self.kwargs or None, self.__dict__ or None),
        )

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 5:
            raise TypeError(f"expected 5 items in state, got {len(state)}")
        func, args, vacs, kwds, namespace = state
        if (
            not callable(func)
            or not isinstance(args, tuple)
            or not isinstance(vacs, tuple)
            or (kwds is not None and not isinstance(kwds, dict))
            or (namespace is not None and not isinstance(namespace, dict))
        ):
            raise TypeError("invalid flexible_partial state")

        args = tuple(args)  # just in case it's a subclass
        vacs = tuple(vacs)
        if kwds is None:
            kwds = {}
        elif type(kwds) is not dict:  # XXX does it need to be *exactly* dict?
            kwds = dict(kwds)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self.base_func = func
        self.args = args
        self.vacancies = vacs
        self.keywords = kwds
