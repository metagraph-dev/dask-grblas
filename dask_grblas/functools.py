from reprlib import recursive_repr


class skip:
    def __repr__(self):
        return "skip"

    __str__ = __repr__
    __reduce__ = __repr__  # This makes it pickle well!


skip = skip()


class flexible_partial:
    """New function with flexible partial application of the given
    arguments and keywords. Any argument slot of the given function
    may be occupied (not just the leading slots).  Use the sentinel
    `skip` to denote vacant argument slots.
    """

    __slots__ = "base_func", "args", "keywords", "__dict__", "__weakref__"

    def __new__(cls, func, /, *args, **keywords):
        if not callable(func):
            raise TypeError("the first argument must be callable")

        if hasattr(func, "base_func"):
            func_ = func.base_func
            func_is_partial = True
        elif hasattr(func, "func"):
            func_ = func.func
            func_is_partial = True
        else:
            func_is_partial = False

        if func_is_partial:
            new_arg = iter(args)
            args = tuple(next(new_arg) if arg is skip else arg for arg in func.args)
            args += tuple(new_arg)
            keywords = {**func.keywords, **keywords}
            func = func_

        self = super(flexible_partial, cls).__new__(cls)

        self.base_func = func
        self.args = args
        self.keywords = keywords
        return self

    def __call__(self, /, *args, **keywords):
        new_arg = iter(args)
        args = (next(new_arg) if arg is skip else arg for arg in self.args)

        keywords = {**self.keywords, **keywords}
        return self.base_func(*args, *new_arg, **keywords)

    @recursive_repr()
    def __repr__(self):
        qualname = type(self).__qualname__
        args = [repr(self.base_func)]
        args.extend(repr(x) for x in self.args)
        args.extend(f"{k}={v!r}" for (k, v) in self.keywords.items())
        if type(self).__module__ == "functools":
            return f"functools.{qualname}({', '.join(args)})"
        return f"{qualname}({', '.join(args)})"

    def __reduce__(self):
        return (
            type(self),
            (self.base_func,),
            (self.base_func, self.args, self.keywords or None, self.__dict__ or None),
        )

    def __setstate__(self, state):
        if not isinstance(state, tuple):
            raise TypeError("argument to __setstate__ must be a tuple")
        if len(state) != 4:
            raise TypeError(f"expected 4 items in state, got {len(state)}")
        func, args, kwds, namespace = state
        if (
            not callable(func)
            or not isinstance(args, tuple)
            or (kwds is not None and not isinstance(kwds, dict))
            or (namespace is not None and not isinstance(namespace, dict))
        ):
            raise TypeError("invalid partial state")

        args = tuple(args)  # just in case it's a subclass
        if kwds is None:
            kwds = {}
        elif type(kwds) is not dict:  # XXX does it need to be *exactly* dict?
            kwds = dict(kwds)
        if namespace is None:
            namespace = {}

        self.__dict__ = namespace
        self.base_func = func
        self.args = args
        self.keywords = kwds
