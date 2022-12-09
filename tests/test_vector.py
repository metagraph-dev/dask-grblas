import inspect

import graphblas as gb
import pytest
from graphblas import dtypes
from pytest import raises
import dask.array as da

import dask_grblas as dgb

from .utils import compare


@pytest.fixture
def vs():
    v = gb.Vector.from_values([0, 1, 2, 4, 5], [0, -20, 30, 40, 50])
    dv0 = dgb.Vector.from_vector(v)
    dv1 = dgb.concat_vectors(
        [
            dgb.Vector.from_vector(gb.Vector.from_values([0, 1, 2], [0, -20, 30])),
            dgb.Vector.from_vector(gb.Vector.from_values([1, 2], [40, 50])),
        ]
    )
    dv2 = dgb.concat_vectors(
        [
            dgb.concat_vectors(
                [
                    dgb.Vector.from_vector(gb.Vector.from_values([0], [0])),
                    dgb.Vector.from_vector(gb.Vector.from_values([0, 1], [-20, 30])),
                ]
            ),
            dgb.Vector.from_vector(gb.Vector.from_values([1, 2], [40, 50])),
        ]
    )
    return v, (dv0, dv1, dv2)


@pytest.fixture
def ws():
    w = gb.Vector.from_values([0, 1, 3, 4, 5], [1.0, 2.0, 3.0, -4.0, 0.0])
    dw0 = dgb.Vector.from_vector(w)
    dw1 = dgb.concat_vectors(
        [
            dgb.Vector.from_vector(gb.Vector.from_values([0, 1], [1.0, 2.0])),
            dgb.Vector.from_vector(gb.Vector.from_values([1, 2, 3], [3.0, -4.0, 0.0])),
        ]
    )
    return w, (dw0, dw1)


@pytest.fixture
def vms():
    val_mask = gb.Vector.from_values(
        [0, 1, 2, 3, 4, 5], [True, False, False, True, True, False], size=6
    )
    dvm0 = dgb.Vector.from_vector(val_mask)
    dvm1 = dgb.concat_vectors(
        [
            dgb.Vector.from_vector(gb.Vector.from_values([0, 1], [True, False])),
            dgb.Vector.from_vector(
                gb.Vector.from_values([0, 1, 2, 3], [False, True, True, False], size=4)
            ),
        ]
    )
    return val_mask, (dvm0, dvm1)


@pytest.fixture
def sms():
    struct_mask = gb.Vector.from_values([0, 3, 4], [False, False, False], size=6)

    dsm0 = dgb.Vector.from_vector(struct_mask)
    dsm1 = dgb.concat_vectors(
        [
            dgb.Vector.from_vector(gb.Vector.from_values([0], [False], size=2)),
            dgb.Vector.from_vector(gb.Vector.from_values([1, 2], [False, False], size=4)),
        ]
    )
    return struct_mask, (dsm0, dsm1)


def test_new():
    v = gb.Vector.new(int)
    dv = dgb.Vector.new(int)
    compare(lambda x: x, v, dv)
    compare(lambda x: x.size, v, dv, compute=False)
    compare(lambda x: x.shape, v, dv, compute=False)
    v = gb.Vector.new(float, 3)
    dv = dgb.Vector.new(float, 3)
    compare(lambda x: x, v, dv)
    compare(lambda x: x.size, v, dv, compute=False)
    compare(lambda x: x.shape, v, dv, compute=False)
    o = object()
    compare(lambda x, y: type(x).new(y), (v, o), (dv, o), errors=True)


def test_from_values():
    n = 100
    I = da.arange(n, chunks=10)
    f = lambda module, arr: module.Vector.from_values(arr, arr, name="arange")
    compare(f, (gb, I.compute()), (dgb, I))

    I = [0, 1, 3, 4, 5]
    V = [1.0, 2.0, 3.0, -4.0, 0.0]
    I_da = da.from_array(I, chunks=3)
    V_da = da.from_array(V, chunks=2)
    f = lambda module, i, v, kwargs: module.Vector.from_values(
        i, v, name="test-from-values", **kwargs
    )
    compare(f, (gb, I, V, {}), (dgb, I_da, V_da, {}))
    compare(f, (gb, I, V, {}), (dgb, I_da, V_da, {"chunks": 3}))


def test_to_values(vs, ws):
    (v, dvs) = vs
    (w, dws) = ws

    def f0(v):
        return v.to_values()[0]

    def f1(v):
        return v.to_values()[1]

    for dv in dvs:
        compare(f0, v, dv)
        compare(f1, v, dv)

    for dw in dws:
        compare(f0, w, dw)
        compare(f1, w, dw)


def test_dup(vs):
    v, dvs = vs
    for dv in dvs:
        compare(lambda x: x.dup(), v, dv)
        compare(lambda x: x.dup(dtype=dtypes.FP64), v, dv)
        o = object()
        compare(lambda x, y: x.dup(y), (v, o), (dv, o), errors=True)
        compare(lambda x: x.dup(mask=1), v, dv, errors=True)
    with raises(TypeError):
        dv.dup(mask=v.S)


@pytest.mark.slow
def test_isequal_isclose(vs, ws):
    o = object()
    for method_name in ["isequal", "isclose"]:
        v = vs[0]
        w = ws[0]
        for dv in vs[1]:
            compare(lambda x, y: getattr(x, method_name)(y), (v, v), (dv, dv))
            compare(lambda x: getattr(x, method_name)(o), v, dv, errors=True)
            for dw in ws[1]:
                compare(lambda x, y: getattr(x, method_name)(y), (v, w), (dv, dw))


def test_nvals(vs):
    v, dvs = vs
    for dv in dvs:
        compare(lambda x: x.nvals, v, dv)
    v = gb.Vector.new(int)
    dv = dgb.Vector.new(int)
    compare(lambda x: x.nvals, v, dv)


def test_clear(vs):
    def f(x):
        x.clear()
        return x

    v, dvs = vs
    compare(f, v, dvs[0])


def test_ewise(vs, ws):
    v = vs[0]
    w = ws[0]
    binfunc = lambda x, y: getattr(x, method_name)(y, op, require_monoid=False).new()
    for op in [gb.monoid.plus, gb.binary.plus]:
        for method_name in ["ewise_add", "ewise_mult"]:

            def f(w, x, y):
                w << getattr(x, method_name)(y, op)
                return w

            # errors = method_name == 'ewise_add' and op is gb.binary.plus
            errors = False
            compute = not errors
            funcs = [
                lambda x, y: getattr(x, method_name)(y, op).new(),
                lambda x, y: getattr(x, method_name)(y, op).new(dtype=dtypes.FP64),
                lambda x, y: getattr(x, method_name)(y, op).new(mask=y.S),
                lambda x, y: getattr(x, method_name)(y, op).new(mask=y.V),
                lambda x, y: getattr(x, method_name)(y, op).new(mask=~x.S),
                lambda x, y: getattr(x, method_name)(y, op).new(mask=~x.V),
            ]
            for dv in vs[1]:
                for index, func in enumerate(funcs):
                    compare(func, (v, v), (dv, dv), errors=errors, compute=compute)
                if method_name == "ewise_add":
                    compare(binfunc, (v, v), (dv, dv))
                compare(
                    f,
                    (v.dup(), v, v),
                    (dv.dup(), dv, dv),
                    errors=errors,
                    compute=compute,
                )
                for dw in ws[1]:
                    for func in funcs:
                        compare(func, (v, w), (dv, dw), errors=errors, compute=compute)
                    if method_name == "ewise_add":
                        compare(binfunc, (v, v), (dv, dv))
                    compare(
                        f,
                        (v.dup(), v, w),
                        (dv.dup(), dv, dw),
                        errors=errors,
                        compute=compute,
                    )
                    compare(
                        f,
                        (w.dup(), v, w),
                        (dw.dup(), dv, dw),
                        errors=errors,
                        compute=compute,
                    )


def test_reduce(vs):
    v, dvs = vs

    def f0(x, y):
        x << y.reduce()
        return x

    def f1(x, y):
        x() << y.reduce()
        return x

    def f2(x, y):
        x(accum=gb.binary.plus) << y.reduce()
        return x

    for dv in dvs:
        compare(lambda x: x.reduce().new(), v, dv)
        compare(lambda x: x.reduce(gb.monoid.max).new(), v, dv)
        compare(lambda x: x.reduce().new(dtype=dtypes.FP64), v, dv)
        compare(lambda x: x.reduce(gb.binary.plus).new(), v, dv)
        for i, f in enumerate([f0, f1, f2]):
            s = gb.Scalar.new(int)
            ds = dgb.Scalar.from_value(s.dup())
            compare(f, (s, v), (ds, dv))

            s = gb.Scalar.from_value(100)
            ds = dgb.Scalar.from_value(s.dup())
            compare(f, (s, v), (ds, dv))

            s = gb.Scalar.new(float)
            ds = dgb.Scalar.from_value(s.dup())
            compare(f, (s, v), (ds, dv))

            s = gb.Scalar.from_value(1.23)
            ds = dgb.Scalar.from_value(s.dup())
            compare(f, (s, v), (ds, dv))


def test_apply(vs):
    v, dvs = vs

    def f(x):
        y = type(x).new(x.dtype, x.size)
        y << x.apply(gb.unary.abs)
        return y

    def g(x, scalar=1):
        y = type(x).new(x.dtype, x.size)
        y << x.apply(gb.binary.gt, right=scalar)
        return y

    def h(x, scalar=2):
        y = type(x).new(x.dtype, x.size)
        y << x.apply(gb.binary.minus, left=scalar)
        return y

    def i(x, scalar=1):
        y = type(x).new(x.dtype, x.size)
        y << x.apply(gb.binary.plus, left=scalar)
        return y

    def j(x, scalar=1):
        y = type(x).new(x.dtype, x.size)
        y << x.apply(gb.monoid.plus, left=scalar)
        return y

    def k(x, scalar=1):
        y = type(x).new(x.dtype, x.size)
        y << x.apply(gb.monoid.plus, right=scalar)
        return y

    for dv in dvs:
        compare(lambda x: x.apply(gb.unary.abs).new(), v, dv)
        compare(lambda x: x.apply(gb.unary.abs).new(dtype=float), v, dv)
        compare(lambda x: x.apply(gb.binary.plus).new(), v, dv, errors=True)
        compare(f, v.dup(), dv.dup())

        compare(lambda x: x.apply(gb.binary.gt, right=1).new(), v, dv)
        compare(lambda x: x.apply(gb.binary.gt, right=1).new(dtype=float), v, dv)
        compare(g, v.dup(), dv.dup())
        s = gb.Scalar.from_value(1)
        ds = dgb.Scalar.from_value(s)
        compare(
            lambda x, s: x.apply(gb.binary.gt, right=s).new(dtype=float),
            (v, s),
            (dv, ds),
        )
        compare(g, (v.dup(), s), (dv.dup(), ds))

        compare(lambda x: x.apply(gb.binary.minus, left=2).new(), v, dv)
        compare(lambda x: x.apply(gb.binary.minus, left=2).new(dtype=float), v, dv)
        compare(h, v.dup(), dv.dup())
        s = gb.Scalar.from_value(2)
        ds = dgb.Scalar.from_value(s)
        compare(
            lambda x, s: x.apply(gb.binary.minus, left=s).new(dtype=float),
            (v, s),
            (dv, ds),
        )
        compare(h, (v.dup(), s), (dv.dup(), ds))

        compare(lambda x: x.apply(gb.binary.plus, left=1).new(), v, dv)
        compare(lambda x: x.apply(gb.binary.plus, left=1).new(dtype=float), v, dv)
        compare(i, v.dup(), dv.dup())
        s = gb.Scalar.from_value(1)
        ds = dgb.Scalar.from_value(s)
        compare(
            lambda x, s: x.apply(gb.binary.minus, left=s).new(dtype=float),
            (v, s),
            (dv, ds),
        )
        compare(i, (v.dup(), s), (dv.dup(), ds))

        compare(lambda x: x.apply(gb.monoid.plus, left=1).new(), v, dv)
        compare(lambda x: x.apply(gb.monoid.plus, left=1).new(dtype=float), v, dv)
        compare(j, v.dup(), dv.dup())
        s = gb.Scalar.from_value(1)
        ds = dgb.Scalar.from_value(s)
        compare(
            lambda x, s: x.apply(gb.binary.minus, left=s).new(dtype=float),
            (v, s),
            (dv, ds),
        )
        compare(j, (v.dup(), s), (dv.dup(), ds))

        compare(lambda x: x.apply(gb.monoid.plus, right=1).new(), v, dv)
        compare(lambda x: x.apply(gb.monoid.plus, right=1).new(dtype=float), v, dv)
        compare(k, v.dup(), dv.dup())
        s = gb.Scalar.from_value(1)
        ds = dgb.Scalar.from_value(s)
        compare(
            lambda x, s: x.apply(gb.binary.minus, right=s).new(dtype=float),
            (v, s),
            (dv, ds),
        )
        compare(k, (v.dup(), s), (dv.dup(), ds))


def test_update(vs, ws):
    v, dvs = vs
    w, dws = ws

    def f0(x, y):
        x.update(y)
        return x

    def f1(x, y):
        x << y
        return x

    def f2(x, y):
        x().update(y)
        return x

    def f3(x, y):
        x(y.S) << y
        return x

    def f4(x, y):
        x(y.V) << y
        return x

    def f5(x, y):
        x(accum=gb.binary.plus).update(y)
        return x

    for f in [f0, f1, f2, f3, f4, f5]:
        for dv in dvs:
            for dw in dws:
                compare(f, (v.dup(), w.dup()), (dv.dup(), dw.dup()))
                compare(f, (v.dup(dtype=float), w.dup()), (dv.dup(dtype=float), dw.dup()))


@pytest.mark.slow
def test_extract(vs, ws, vms, sms):
    v, dvs = vs
    w, dws = ws
    vm, dvms = vms
    sm, dsms = sms

    for index in [
        slice(None),
        [0, 3, 1, 4, 2, 5],
        [0, 5, 5, 1, 2, 0],
        slice(None, None, -1),
        [0] * 6,
    ]:

        def f1(x, y):
            x << y[index]
            return x

        def f2(x, y):
            x() << y[index]
            return x

        def g1(m, x, y):
            x(mask=m) << y[index]
            return x

        def g2(x, y):
            x(accum=gb.binary.plus) << y[index]
            return x

        def g3(x, y):
            x(replace=True) << y[index]
            return x

        def g4(x, y):
            x(replace=False) << y[index]
            return x

        def h1(x, y):
            x(accum=gb.binary.plus, replace=True) << y[index]
            return x

        def h2(x, y):
            x(accum=gb.binary.plus, replace=False) << y[index]
            return x

        def h3(m, x, y):
            x(mask=m, replace=True) << y[index]
            return x

        def h4(m, x, y):
            x(mask=m, replace=False) << y[index]
            return x

        def h5(m, x, y):
            x(mask=m, accum=gb.binary.plus) << y[index]
            return x

        def i1(m, x, y):
            x(mask=m, accum=gb.binary.plus, replace=True) << y[index]
            return x

        def i2(m, x, y):
            x(mask=m, accum=gb.binary.plus, replace=False) << y[index]
            return x

        for dv in dvs:
            compare(lambda x: x[index].new(), v, dv)
            compare(lambda x: x[index].new(dtype=float), v, dv)
            for dw in dws:
                compare(f1, (v.dup(), w), (dv.dup(), dw))
                compare(f1, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                compare(f2, (v.dup(), w), (dv.dup(), dw))
                compare(f2, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                compare(g2, (v.dup(), w), (dv.dup(), dw))
                compare(g2, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                compare(g3, (v.dup(), w), (dv.dup(), dw), errors=True)
                compare(g3, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw), errors=True)
                compare(g4, (v.dup(), w), (dv.dup(), dw))
                compare(g4, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                compare(h1, (v.dup(), w), (dv.dup(), dw), errors=True)
                compare(h1, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw), errors=True)
                compare(h2, (v.dup(), w), (dv.dup(), dw))
                compare(h2, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                for dvm in dvms:
                    compare(g1, (vm.V, v.dup(), w), (dvm.V, dv.dup(), dw))
                    compare(
                        g1,
                        (vm.V, v.dup(dtype=float), w),
                        (dvm.V, dv.dup(dtype=float), dw),
                    )
                    compare(lambda m, x: x[index].new(mask=m), (vm.V, w), (dvm.V, dw))
                    compare(
                        lambda m, x: x[index].new(mask=m, dtype=float),
                        (vm.V, w),
                        (dvm.V, dw),
                    )
                    compare(h3, (vm.V, v.dup(), w), (dvm.V, dv.dup(), dw))
                    compare(
                        h3,
                        (vm.V, v.dup(dtype=float), w),
                        (dvm.V, dv.dup(dtype=float), dw),
                    )
                    compare(
                        lambda m, x: x[index].new(mask=m, replace=True),
                        (vm.V, w),
                        (dvm.V, dw),
                        errors=True,
                    )
                    compare(
                        lambda m, x: x[index].new(mask=m, replace=True, dtype=float),
                        (vm.V, w),
                        (dvm.V, dw),
                        errors=True,
                    )
                    compare(h4, (vm.V, v.dup(), w), (dvm.V, dv.dup(), dw))
                    compare(
                        h4,
                        (vm.V, v.dup(dtype=float), w),
                        (dvm.V, dv.dup(dtype=float), dw),
                    )
                    compare(
                        lambda m, x: x[index].new(mask=m, replace=False),
                        (vm.V, w),
                        (dvm.V, dw),
                        errors=True,
                    )
                    compare(
                        lambda m, x: x[index].new(mask=m, replace=False, dtype=float),
                        (vm.V, w),
                        (dvm.V, dw),
                        errors=True,
                    )
                    compare(h5, (vm.V, v.dup(), w), (dvm.V, dv.dup(), dw))
                    compare(
                        h5,
                        (vm.V, v.dup(dtype=float), w),
                        (dvm.V, dv.dup(dtype=float), dw),
                    )
                    compare(
                        lambda m, x: x[index].new(mask=m, accum=gb.binary.plus),
                        (vm.V, w),
                        (dvm.V, dw),
                        errors=True,
                    )
                    compare(
                        lambda m, x: x[index].new(mask=m, accum=gb.binary.plus, dtype=float),
                        (vm.V, w),
                        (dvm.V, dw),
                        errors=True,
                    )
                    compare(i1, (vm.V, v.dup(), w), (dvm.V, dv.dup(), dw))
                    compare(
                        i1,
                        (vm.V, v.dup(dtype=float), w),
                        (dvm.V, dv.dup(dtype=float), dw),
                    )
                    compare(
                        lambda m, x: x[index].new(mask=m, accum=gb.binary.plus, replace=True),
                        (vm.V, w),
                        (dvm.V, dw),
                        errors=True,
                    )
                    compare(
                        lambda m, x: x[index].new(
                            mask=m, accum=gb.binary.plus, replace=True, dtype=float
                        ),
                        (vm.V, w),
                        (dvm.V, dw),
                        errors=True,
                    )
                    compare(i2, (vm.V, v.dup(), w), (dvm.V, dv.dup(), dw))
                    compare(
                        i2,
                        (vm.V, v.dup(dtype=float), w),
                        (dvm.V, dv.dup(dtype=float), dw),
                    )
                    compare(
                        lambda m, x: x[index].new(mask=m, accum=gb.binary.plus, replace=False),
                        (vm.V, w),
                        (dvm.V, dw),
                        errors=True,
                    )
                    compare(
                        lambda m, x: x[index].new(
                            mask=m, accum=gb.binary.plus, replace=False, dtype=float
                        ),
                        (vm.V, w),
                        (dvm.V, dw),
                        errors=True,
                    )
                for dsm in dsms:
                    compare(g1, (sm.S, v.dup(), w), (dsm.S, dv.dup(), dw))
                    compare(
                        g1,
                        (sm.S, v.dup(dtype=float), w),
                        (dsm.S, dv.dup(dtype=float), dw),
                    )
                    compare(lambda m, x: x[index].new(mask=m), (sm.S, w), (dsm.S, dw))
                    compare(
                        lambda m, x: x[index].new(mask=m, dtype=float),
                        (sm.S, w),
                        (dsm.S, dw),
                    )
                    compare(h3, (sm.S, v.dup(), w), (dsm.S, dv.dup(), dw))
                    compare(
                        h3,
                        (sm.S, v.dup(dtype=float), w),
                        (dsm.S, dv.dup(dtype=float), dw),
                    )
                    compare(
                        lambda m, x: x[index].new(mask=m, replace=True),
                        (sm.S, w),
                        (dsm.S, dw),
                        errors=True,
                    )
                    compare(
                        lambda m, x: x[index].new(mask=m, replace=True, dtype=float),
                        (sm.S, w),
                        (dsm.S, dw),
                        errors=True,
                    )
                    compare(h4, (sm.S, v.dup(), w), (dsm.S, dv.dup(), dw))
                    compare(
                        h4,
                        (sm.S, v.dup(dtype=float), w),
                        (dsm.S, dv.dup(dtype=float), dw),
                    )
                    compare(
                        lambda m, x: x[index].new(mask=m, replace=False),
                        (sm.S, w),
                        (dsm.S, dw),
                        errors=True,
                    )
                    compare(
                        lambda m, x: x[index].new(mask=m, replace=False, dtype=float),
                        (sm.S, w),
                        (dsm.S, dw),
                        errors=True,
                    )
                    compare(h5, (sm.S, v.dup(), w), (dsm.S, dv.dup(), dw))
                    compare(
                        h5,
                        (sm.S, v.dup(dtype=float), w),
                        (dsm.S, dv.dup(dtype=float), dw),
                    )
                    compare(
                        lambda m, x: x[index].new(mask=m, accum=gb.binary.plus),
                        (sm.S, w),
                        (dsm.S, dw),
                        errors=True,
                    )
                    compare(
                        lambda m, x: x[index].new(mask=m, accum=gb.binary.plus, dtype=float),
                        (sm.S, w),
                        (dsm.S, dw),
                        errors=True,
                    )
                    compare(i1, (sm.S, v.dup(), w), (dsm.S, dv.dup(), dw))
                    compare(
                        i1,
                        (sm.S, v.dup(dtype=float), w),
                        (dsm.S, dv.dup(dtype=float), dw),
                    )
                    compare(
                        lambda m, x: x[index].new(mask=m, accum=gb.binary.plus, replace=True),
                        (sm.S, w),
                        (dsm.S, dw),
                        errors=True,
                    )
                    compare(
                        lambda m, x: x[index].new(
                            mask=m, accum=gb.binary.plus, replace=True, dtype=float
                        ),
                        (sm.S, w),
                        (dsm.S, dw),
                        errors=True,
                    )
                    compare(i2, (sm.S, v.dup(), w), (dsm.S, dv.dup(), dw))
                    compare(
                        i2,
                        (sm.S, v.dup(dtype=float), w),
                        (dsm.S, dv.dup(dtype=float), dw),
                    )
                    compare(
                        lambda m, x: x[index].new(mask=m, accum=gb.binary.plus, replace=False),
                        (sm.S, w),
                        (dsm.S, dw),
                        errors=True,
                    )
                    compare(
                        lambda m, x: x[index].new(
                            mask=m, accum=gb.binary.plus, replace=False, dtype=float
                        ),
                        (sm.S, w),
                        (dsm.S, dw),
                        errors=True,
                    )


@pytest.mark.slow
def test_extract_da(vs, ws, vms, sms):
    v, dvs = vs
    w, dws = ws
    vm, dvms = vms
    sm, dsms = sms

    index1_da = da.from_array([0, 3, 1, 4, 2, 5], chunks=((2, 2, 2),))
    index2_da = da.from_array([0, 5, 5, 1, 2, 0], chunks=((2, 2, 2),))
    index3_da = da.from_array([0] * 6, chunks=((2, 2, 2),))

    for index in [
        index1_da,
        index2_da,
        index3_da,
    ]:
        ind = index.compute()

        def f1(x, y, index):
            x << y[index]
            return x

        def f2(x, y, index):
            x() << y[index]
            return x

        def g1(m, x, y, index):
            x(mask=m) << y[index]
            return x

        def g2(x, y, index):
            x(accum=gb.binary.plus) << y[index]
            return x

        def g3(x, y, index):
            x(replace=True) << y[index]
            return x

        def g4(x, y, index):
            x(replace=False) << y[index]
            return x

        def h1(x, y, index):
            x(accum=gb.binary.plus, replace=True) << y[index]
            return x

        def h2(x, y, index):
            x(accum=gb.binary.plus, replace=False) << y[index]
            return x

        def h3(m, x, y, index):
            x(mask=m, replace=True) << y[index]
            return x

        def h4(m, x, y, index):
            x(mask=m, replace=False) << y[index]
            return x

        def h5(m, x, y, index):
            x(mask=m, accum=gb.binary.plus) << y[index]
            return x

        def i1(m, x, y, index):
            x(mask=m, accum=gb.binary.plus, replace=True) << y[index]
            return x

        def i2(m, x, y, index):
            x(mask=m, accum=gb.binary.plus, replace=False) << y[index]
            return x

        for dv in dvs:
            compare(lambda x, index: x[index].new(), (v, ind), (dv, index))
            compare(lambda x, index: x[index].new(dtype=float), (v, ind), (dv, index))
            for dw in dws:
                compare(f1, (v.dup(), w, ind), (dv.dup(), dw, index))
                compare(f1, (v.dup(dtype=float), w, ind), (dv.dup(dtype=float), dw, index))
                compare(f2, (v.dup(), w, ind), (dv.dup(), dw, index))
                compare(f2, (v.dup(dtype=float), w, ind), (dv.dup(dtype=float), dw, index))
                compare(g2, (v.dup(), w, ind), (dv.dup(), dw, index))
                compare(g2, (v.dup(dtype=float), w, ind), (dv.dup(dtype=float), dw, index))
                compare(g3, (v.dup(), w, ind), (dv.dup(), dw, index), errors=True)
                compare(
                    g3, (v.dup(dtype=float), w, ind), (dv.dup(dtype=float), dw, index), errors=True
                )
                compare(g4, (v.dup(), w, ind), (dv.dup(), dw, index))
                compare(g4, (v.dup(dtype=float), w, ind), (dv.dup(dtype=float), dw, index))
                compare(h1, (v.dup(), w, ind), (dv.dup(), dw, index), errors=True)
                compare(
                    h1, (v.dup(dtype=float), w, ind), (dv.dup(dtype=float), dw, index), errors=True
                )
                compare(h2, (v.dup(), w, ind), (dv.dup(), dw, index))
                compare(h2, (v.dup(dtype=float), w, ind), (dv.dup(dtype=float), dw, index))
                for dvm in dvms:
                    compare(g1, (vm.V, v.dup(), w, ind), (dvm.V, dv.dup(), dw, index))
                    compare(
                        g1,
                        (vm.V, v.dup(dtype=float), w, ind),
                        (dvm.V, dv.dup(dtype=float), dw, index),
                    )
                    compare(
                        lambda m, x, index: x[index].new(mask=m), (vm.V, w, ind), (dvm.V, dw, index)
                    )
                    compare(
                        lambda m, x, index: x[index].new(mask=m, dtype=float),
                        (vm.V, w, ind),
                        (dvm.V, dw, index),
                    )
                    compare(h3, (vm.V, v.dup(), w, ind), (dvm.V, dv.dup(), dw, index))
                    compare(
                        h3,
                        (vm.V, v.dup(dtype=float), w, ind),
                        (dvm.V, dv.dup(dtype=float), dw, index),
                    )
                    compare(
                        lambda m, x, index: x[index].new(mask=m, replace=True),
                        (vm.V, w, ind),
                        (dvm.V, dw, index),
                        errors=True,
                    )
                    compare(
                        lambda m, x, index: x[index].new(mask=m, replace=True, dtype=float),
                        (vm.V, w, ind),
                        (dvm.V, dw, index),
                        errors=True,
                    )
                    compare(h4, (vm.V, v.dup(), w, ind), (dvm.V, dv.dup(), dw, index))
                    compare(
                        h4,
                        (vm.V, v.dup(dtype=float), w, ind),
                        (dvm.V, dv.dup(dtype=float), dw, index),
                    )
                    compare(
                        lambda m, x, index: x[index].new(mask=m, replace=False),
                        (vm.V, w, ind),
                        (dvm.V, dw, index),
                        errors=True,
                    )
                    compare(
                        lambda m, x, index: x[index].new(mask=m, replace=False, dtype=float),
                        (vm.V, w, ind),
                        (dvm.V, dw, index),
                        errors=True,
                    )
                    compare(h5, (vm.V, v.dup(), w, ind), (dvm.V, dv.dup(), dw, index))
                    compare(
                        h5,
                        (vm.V, v.dup(dtype=float), w, ind),
                        (dvm.V, dv.dup(dtype=float), dw, index),
                    )
                    compare(
                        lambda m, x, index: x[index].new(mask=m, accum=gb.binary.plus),
                        (vm.V, w, ind),
                        (dvm.V, dw, index),
                        errors=True,
                    )
                    compare(
                        lambda m, x, index: x[index].new(mask=m, accum=gb.binary.plus, dtype=float),
                        (vm.V, w, ind),
                        (dvm.V, dw, index),
                        errors=True,
                    )
                    compare(i1, (vm.V, v.dup(), w, ind), (dvm.V, dv.dup(), dw, index))
                    compare(
                        i1,
                        (vm.V, v.dup(dtype=float), w, ind),
                        (dvm.V, dv.dup(dtype=float), dw, index),
                    )
                    compare(
                        lambda m, x, index: x[index].new(
                            mask=m, accum=gb.binary.plus, replace=True
                        ),
                        (vm.V, w, ind),
                        (dvm.V, dw, index),
                        errors=True,
                    )
                    compare(
                        lambda m, x, index: x[index].new(
                            mask=m, accum=gb.binary.plus, replace=True, dtype=float
                        ),
                        (vm.V, w, ind),
                        (dvm.V, dw, index),
                        errors=True,
                    )
                    compare(i2, (vm.V, v.dup(), w, ind), (dvm.V, dv.dup(), dw, index))
                    compare(
                        i2,
                        (vm.V, v.dup(dtype=float), w, ind),
                        (dvm.V, dv.dup(dtype=float), dw, index),
                    )
                    compare(
                        lambda m, x, index: x[index].new(
                            mask=m, accum=gb.binary.plus, replace=False
                        ),
                        (vm.V, w, ind),
                        (dvm.V, dw, index),
                        errors=True,
                    )
                    compare(
                        lambda m, x, index: x[index].new(
                            mask=m, accum=gb.binary.plus, replace=False, dtype=float
                        ),
                        (vm.V, w, ind),
                        (dvm.V, dw, index),
                        errors=True,
                    )
                for dsm in dsms:
                    compare(g1, (sm.S, v.dup(), w, ind), (dsm.S, dv.dup(), dw, index))
                    compare(
                        g1,
                        (sm.S, v.dup(dtype=float), w, ind),
                        (dsm.S, dv.dup(dtype=float), dw, index),
                    )
                    compare(
                        lambda m, x, index: x[index].new(mask=m), (sm.S, w, ind), (dsm.S, dw, index)
                    )
                    compare(
                        lambda m, x, index: x[index].new(mask=m, dtype=float),
                        (sm.S, w, ind),
                        (dsm.S, dw, index),
                    )
                    compare(h3, (sm.S, v.dup(), w, ind), (dsm.S, dv.dup(), dw, index))
                    compare(
                        h3,
                        (sm.S, v.dup(dtype=float), w, ind),
                        (dsm.S, dv.dup(dtype=float), dw, index),
                    )
                    compare(
                        lambda m, x, index: x[index].new(mask=m, replace=True),
                        (sm.S, w, ind),
                        (dsm.S, dw, index),
                        errors=True,
                    )
                    compare(
                        lambda m, x, index: x[index].new(mask=m, replace=True, dtype=float),
                        (sm.S, w, ind),
                        (dsm.S, dw, index),
                        errors=True,
                    )
                    compare(h4, (sm.S, v.dup(), w, ind), (dsm.S, dv.dup(), dw, index))
                    compare(
                        h4,
                        (sm.S, v.dup(dtype=float), w, ind),
                        (dsm.S, dv.dup(dtype=float), dw, index),
                    )
                    compare(
                        lambda m, x, index: x[index].new(mask=m, replace=False),
                        (sm.S, w, ind),
                        (dsm.S, dw, index),
                        errors=True,
                    )
                    compare(
                        lambda m, x, index: x[index].new(mask=m, replace=False, dtype=float),
                        (sm.S, w, ind),
                        (dsm.S, dw, index),
                        errors=True,
                    )
                    compare(h5, (sm.S, v.dup(), w, ind), (dsm.S, dv.dup(), dw, index))
                    compare(
                        h5,
                        (sm.S, v.dup(dtype=float), w, ind),
                        (dsm.S, dv.dup(dtype=float), dw, index),
                    )
                    compare(
                        lambda m, x, index: x[index].new(mask=m, accum=gb.binary.plus),
                        (sm.S, w, ind),
                        (dsm.S, dw, index),
                        errors=True,
                    )
                    compare(
                        lambda m, x, index: x[index].new(mask=m, accum=gb.binary.plus, dtype=float),
                        (sm.S, w, ind),
                        (dsm.S, dw, index),
                        errors=True,
                    )
                    compare(i1, (sm.S, v.dup(), w, ind), (dsm.S, dv.dup(), dw, index))
                    compare(
                        i1,
                        (sm.S, v.dup(dtype=float), w, ind),
                        (dsm.S, dv.dup(dtype=float), dw, index),
                    )
                    compare(
                        lambda m, x, index: x[index].new(
                            mask=m, accum=gb.binary.plus, replace=True
                        ),
                        (sm.S, w, ind),
                        (dsm.S, dw, index),
                        errors=True,
                    )
                    compare(
                        lambda m, x, index: x[index].new(
                            mask=m, accum=gb.binary.plus, replace=True, dtype=float
                        ),
                        (sm.S, w, ind),
                        (dsm.S, dw, index),
                        errors=True,
                    )
                    compare(i2, (sm.S, v.dup(), w, ind), (dsm.S, dv.dup(), dw, index))
                    compare(
                        i2,
                        (sm.S, v.dup(dtype=float), w, ind),
                        (dsm.S, dv.dup(dtype=float), dw, index),
                    )
                    compare(
                        lambda m, x, index: x[index].new(
                            mask=m, accum=gb.binary.plus, replace=False
                        ),
                        (sm.S, w, ind),
                        (dsm.S, dw, index),
                        errors=True,
                    )
                    compare(
                        lambda m, x, index: x[index].new(
                            mask=m, accum=gb.binary.plus, replace=False, dtype=float
                        ),
                        (sm.S, w, ind),
                        (dsm.S, dw, index),
                        errors=True,
                    )


@pytest.mark.slow
def test_subassign(vs, ws, vms, sms):

    test_replace_true = True
    v, dvs = vs
    gw, dws = ws
    vm, dvms = vms
    sm, dsms = sms

    scalars = (1, 1.0)
    gws = (gw,) * len(dws) + scalars
    dws = dws + scalars

    def inv_if(mask, is_inv=False):
        if is_inv:
            return ~mask
        return mask

    index1_da = da.from_array([0, 3, 1, 4, 2, 5], chunks=((2, 2, 2),))
    index2_da = da.from_array([0, 5, 5, 1, 2, 0], chunks=((2, 2, 2),))
    index3_da = da.from_array([0] * 6, chunks=((2, 2, 2),))

    for index in [
        index1_da,
        index2_da,
        index3_da,
        [0, 3, 1, 4, 2, 5],
        [0, 5, 5, 1, 2, 0],
        slice(None),
        slice(None, None, -1),
        [0] * 6,
    ]:

        def f2(x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x[ind]() << y
            return x

        def g1(m, x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x[ind](mask=m) << y
            return x

        def g2(x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x[ind](accum=gb.binary.plus) << y
            return x

        def g3(x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x[ind](replace=True) << y
            return x

        def g4(x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x[ind](replace=False) << y
            return x

        def h1(x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x[ind](accum=gb.binary.plus, replace=True) << y
            return x

        def h2(x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x[ind](accum=gb.binary.plus, replace=False) << y
            return x

        def h3(m, x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x[ind](mask=m, replace=test_replace_true) << y
            return x

        def h4(m, x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x[ind](mask=m, replace=False) << y
            return x

        def h5(m, x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x[ind](mask=m, accum=gb.binary.plus) << y
            return x

        def i1(m, x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x[ind](mask=m, accum=gb.binary.plus, replace=test_replace_true) << y
            return x

        def i2(m, x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x[ind](mask=m, accum=gb.binary.plus, replace=False) << y
            return x

        for dv in dvs:
            for w, dw in zip(gws, dws):
                compare(f2, (v.dup(), w), (dv.dup(), dw))
                compare(f2, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                compare(g2, (v.dup(), w), (dv.dup(), dw))
                compare(g2, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                compare(g3, (v.dup(), w), (dv.dup(), dw), errors=True)
                compare(g3, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw), errors=True)
                compare(g4, (v.dup(), w), (dv.dup(), dw))
                compare(g4, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                compare(h1, (v.dup(), w), (dv.dup(), dw), errors=True)
                compare(h1, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw), errors=True)
                compare(h2, (v.dup(), w), (dv.dup(), dw))
                compare(h2, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                for is_inv in [False, True]:
                    for dvm in dvms:
                        compare(
                            g1,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            g1,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h3,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h3,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h4,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h4,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h5,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h5,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            i1,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            i1,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            i2,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            i2,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                    for dsm in dsms:
                        compare(
                            g1,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            g1,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h3,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h3,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h4,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h4,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h5,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h5,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            i1,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            i1,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            i2,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            i2,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )


@pytest.mark.slow
def test_assign(vs, ws, vms, sms):

    test_replace_true = True
    v, dvs = vs
    gw, dws = ws
    vm, dvms = vms
    sm, dsms = sms

    gb_s = gb.Scalar.from_value(9)
    dgb_s = dgb.Scalar.from_value(9)
    scalars = (1, 1.0)
    gws = (gw,) * len(dws) + scalars + (gb_s,)
    dws = dws + scalars + (dgb_s,)

    def inv_if(mask, is_inv=False):
        if is_inv:
            return ~mask
        return mask

    index1_da = da.from_array([0, 3, 1, 4, 2, 5], chunks=((2, 2, 2),))
    index2_da = da.from_array([0, 5, 5, 1, 2, 0], chunks=((2, 2, 2),))
    index3_da = da.from_array([0] * 6, chunks=((2, 2, 2),))

    for index in [
        index1_da,
        index2_da,
        index3_da,
        [0] * 6,
        slice(None, None, -1),
        slice(None),
        [0, 3, 1, 4, 2, 5],
        [0, 5, 5, 1, 2, 0],
    ]:

        def f1(x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x[ind] << y
            return x

        def f2(x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x()[ind] << y
            return x

        def g1(m, x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x(mask=m)[ind] << y
            return x

        def g2(x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x(accum=gb.binary.plus)[ind] << y
            return x

        def g3(x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x(replace=True)[ind] << y
            return x

        def g4(x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x(replace=False)[ind] << y
            return x

        def h1(x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x(accum=gb.binary.plus, replace=True)[ind] << y
            return x

        def h2(x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x(accum=gb.binary.plus, replace=False)[ind] << y
            return x

        def h3(m, x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x(mask=m, replace=test_replace_true)[ind] << y
            return x

        def h4(m, x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x(mask=m, replace=False)[ind] << y
            return x

        def h5(m, x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x(mask=m, accum=gb.binary.plus)[ind] << y
            return x

        def i1(m, x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x(mask=m, accum=gb.binary.plus, replace=test_replace_true)[ind] << y
            return x

        def i2(m, x, y):
            ind = (
                index.compute()
                if isinstance(x, gb.core.base.BaseType) and type(index) is da.core.Array
                else index
            )
            x(mask=m, accum=gb.binary.plus, replace=False)[ind] << y
            return x

        for dv in dvs:
            for w, dw in zip(gws, dws):
                compare(f1, (v.dup(), w), (dv.dup(), dw))
                compare(f1, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                compare(f2, (v.dup(), w), (dv.dup(), dw))
                compare(f2, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                compare(g2, (v.dup(), w), (dv.dup(), dw))
                compare(g2, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                compare(g3, (v.dup(), w), (dv.dup(), dw), errors=True)
                compare(g3, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw), errors=True)
                compare(g4, (v.dup(), w), (dv.dup(), dw))
                compare(g4, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                compare(h1, (v.dup(), w), (dv.dup(), dw), errors=True)
                compare(h1, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw), errors=True)
                compare(h2, (v.dup(), w), (dv.dup(), dw))
                compare(h2, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                for is_inv in [False, True]:
                    for dvm in dvms:
                        compare(
                            g1,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            g1,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h3,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h3,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h4,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h4,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h5,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h5,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            i1,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            i1,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            i2,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            i2,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                    for dsm in dsms:
                        compare(
                            g1,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            g1,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h3,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h3,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h4,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h4,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h5,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h5,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            i1,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            i1,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            i2,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            i2,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )


def test_reduce_assign(vs, ws, vms, sms):

    test_replace_true = True
    v, dvs = vs
    gw, dws = ws
    vm, dvms = vms
    sm, dsms = sms

    gb_s = gb.Scalar.from_value(9)
    dgb_s = dgb.Scalar.from_value(9)
    scalars = (1, 1.0)
    gws = (gw,) * len(dws) + scalars + (gb_s,)
    dws = dws + scalars + (dgb_s,)

    def inv_if(mask, is_inv=False):
        if is_inv:
            return ~mask
        return mask

    index1_da = da.from_array([0, 3, 1, 4, 2, 5], chunks=((2, 2, 2),))
    index2_da = da.from_array([0, 5, 5, 1, 2, 0], chunks=((2, 2, 2),))
    index3_da = da.from_array([0] * 6, chunks=((2, 2, 2),))

    for index in [
        [0, 3, 1, 4, 2, 5],
        [0, 5, 5, 1, 2, 0],
        [0] * 6,
        slice(None),
        slice(None, None, -1),
        index1_da,
        index2_da,
        index3_da,
    ]:

        def f1(x, y):
            if isinstance(x, gb.core.base.BaseType):
                ind = index.compute() if type(index) is da.core.Array else index
                x[ind] << y
            else:
                ind = index
                dgb.expr.reduce_assign(x, ind, y, dup_op="last")
            return x

        def g1(m, x, y):
            if isinstance(x, gb.core.base.BaseType):
                ind = index.compute() if type(index) is da.core.Array else index
                x(mask=m)[ind] << y
            else:
                ind = index
                dgb.expr.reduce_assign(x, ind, y, mask=m, dup_op="last")
            return x

        def g2(x, y):
            if isinstance(x, gb.core.base.BaseType):
                ind = index.compute() if type(index) is da.core.Array else index
                x(accum=gb.binary.plus)[ind] << y
            else:
                ind = index
                dgb.expr.reduce_assign(x, ind, y, accum=gb.binary.plus, dup_op="last")
            return x

        def g3(x, y):
            if isinstance(x, gb.core.base.BaseType):
                ind = index.compute() if type(index) is da.core.Array else index
                x(replace=True)[ind] << y
            else:
                ind = index
                dgb.expr.reduce_assign(x, ind, y, dup_op="last", replace=True)
            return x

        def g4(x, y):
            if isinstance(x, gb.core.base.BaseType):
                ind = index.compute() if type(index) is da.core.Array else index
                x(replace=False)[ind] << y
            else:
                ind = index
                dgb.expr.reduce_assign(x, ind, y, dup_op="last", replace=False)
            return x

        def h1(x, y):
            if isinstance(x, gb.core.base.BaseType):
                ind = index.compute() if type(index) is da.core.Array else index
                x(accum=gb.binary.plus, replace=True)[ind] << y
            else:
                ind = index
                dgb.expr.reduce_assign(x, ind, y, dup_op="last", accum=gb.binary.plus, replace=True)
            return x

        def h2(x, y):
            if isinstance(x, gb.core.base.BaseType):
                ind = index.compute() if type(index) is da.core.Array else index
                x(accum=gb.binary.plus, replace=False)[ind] << y
            else:
                ind = index
                dgb.expr.reduce_assign(
                    x, ind, y, dup_op="last", accum=gb.binary.plus, replace=False
                )
            return x

        def h3(m, x, y):
            if isinstance(x, gb.core.base.BaseType):
                ind = index.compute() if type(index) is da.core.Array else index
                x(mask=m, replace=True)[ind] << y
            else:
                ind = index
                dgb.expr.reduce_assign(x, ind, y, mask=m, dup_op="last", replace=True)
            return x

        def h4(m, x, y):
            if isinstance(x, gb.core.base.BaseType):
                ind = index.compute() if type(index) is da.core.Array else index
                x(mask=m, replace=False)[ind] << y
            else:
                ind = index
                dgb.expr.reduce_assign(x, ind, y, mask=m, dup_op="last", replace=False)
            return x

        def h5(m, x, y):
            if isinstance(x, gb.core.base.BaseType):
                ind = index.compute() if type(index) is da.core.Array else index
                x(mask=m, accum=gb.binary.plus)[ind] << y
            else:
                ind = index
                dgb.expr.reduce_assign(x, ind, y, mask=m, accum=gb.binary.plus, dup_op="last")
            return x

        def i1(m, x, y):
            if isinstance(x, gb.core.base.BaseType):
                ind = index.compute() if type(index) is da.core.Array else index
                x(mask=m, accum=gb.binary.plus, replace=True)[ind] << y
            else:
                ind = index
                dgb.expr.reduce_assign(
                    x, ind, y, mask=m, accum=gb.binary.plus, replace=True, dup_op="last"
                )
            return x

        def i2(m, x, y):
            if isinstance(x, gb.core.base.BaseType):
                ind = index.compute() if type(index) is da.core.Array else index
                x(mask=m, accum=gb.binary.plus, replace=False)[ind] << y
            else:
                ind = index
                dgb.expr.reduce_assign(
                    x, ind, y, mask=m, accum=gb.binary.plus, replace=False, dup_op="last"
                )
            return x

        for dv in dvs:
            for w, dw in zip(gws, dws):
                compare(f1, (v.dup(), w), (dv.dup(), dw))
                compare(f1, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                compare(g2, (v.dup(), w), (dv.dup(), dw))
                compare(g2, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                compare(g3, (v.dup(), w), (dv.dup(), dw), errors=True)
                compare(g3, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw), errors=True)
                compare(g4, (v.dup(), w), (dv.dup(), dw))
                compare(g4, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                compare(h1, (v.dup(), w), (dv.dup(), dw), errors=True)
                compare(h1, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw), errors=True)
                compare(h2, (v.dup(), w), (dv.dup(), dw))
                compare(h2, (v.dup(dtype=float), w), (dv.dup(dtype=float), dw))
                for is_inv in [False, True]:
                    for dvm in dvms:
                        compare(
                            g1,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            g1,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h3,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h3,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h4,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h4,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h5,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h5,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            i1,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            i1,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            i2,
                            (inv_if(vm.V, is_inv), v.dup(), w),
                            (inv_if(dvm.V, is_inv), dv.dup(), dw),
                        )
                        compare(
                            i2,
                            (inv_if(vm.V, is_inv), v.dup(dtype=float), w),
                            (inv_if(dvm.V, is_inv), dv.dup(dtype=float), dw),
                        )
                    for dsm in dsms:
                        compare(
                            g1,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            g1,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h3,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h3,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h4,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h4,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            h5,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            h5,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            i1,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            i1,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )
                        compare(
                            i2,
                            (inv_if(sm.S, is_inv), v.dup(), w),
                            (inv_if(dsm.S, is_inv), dv.dup(), dw),
                        )
                        compare(
                            i2,
                            (inv_if(sm.S, is_inv), v.dup(dtype=float), w),
                            (inv_if(dsm.S, is_inv), dv.dup(dtype=float), dw),
                        )


@pytest.mark.xfail
def test_attrs(vs):
    v, dvs = vs
    dv = dvs[0]
    assert set(dir(v)) - set(dir(dv)) == {
        "__del__",  # TODO
        "_assign_element",
        "_extract_element",
        "_is_scalar",
        "_prep_for_assign",
        "_prep_for_extract",
        "_delete_element",
        "gb_obj",
        "show",
    }
    assert set(dir(dv)) - set(dir(v)) == {
        "_delayed",
        "_meta",
        "_optional_dup",
        "compute",
        "from_vector",
        "from_delayed",
        "persist",
        "visualize",
    }


def test_signatures_match_grblas():
    def has_signature(x):
        try:
            inspect.signature(x)
        except Exception:
            return False
        else:
            return True

    v1 = gb.Vector.from_values(1, 2)
    v2 = dgb.Vector.from_values(1, 2)
    skip = {
        "from_pygraphblas",
        "to_pygraphblas",
        "__class__",
        "__init__",
        "from_values",
        "inner",  # TODO
        "outer",  # TODO
    }
    d1 = {
        key: inspect.signature(val)
        for key, val in inspect.getmembers(v1)
        if has_signature(val) and key not in skip
    }
    d2 = {
        key: inspect.signature(val)
        for key, val in inspect.getmembers(v2)
        if has_signature(val) and key not in skip
    }
    for key, val in d1.items():
        # if not key.startswith("_") or key.startswith("__") and key in d2:
        if not key.startswith("_"):
            assert val == d2[key], (key, str(val), str(d2[key]))
