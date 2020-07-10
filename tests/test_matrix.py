"""
import pytest
import grblas as gb
import dask_grblas as dgb
from grblas import dtypes
from pytest import raises
from .utils import compare


@pytest.fixture
def As():
    v = gb.Vector.from_values([0, 1, 2, 4, 5], [0, -20, 30, 40, 50])
    dv0 = dgb.Vector.from_vector(v)
    dv1 = dgb.concat_vectors([
        dgb.Vector.from_vector(gb.Vector.from_values([0, 1, 2], [0, -20, 30])),
        dgb.Vector.from_vector(gb.Vector.from_values([1, 2], [40, 50])),
    ])
    dv2 = dgb.concat_vectors([
        dgb.concat_vectors([
            dgb.Vector.from_vector(gb.Vector.from_values([0], [0])),
            dgb.Vector.from_vector(gb.Vector.from_values([0, 1], [-20, 30])),
        ]),
        dgb.Vector.from_vector(gb.Vector.from_values([1, 2], [40, 50])),
    ])
    return v, (dv0, dv1, dv2)


@pytest.fixture
def Bs():
    v = gb.Vector.from_values([0, 1, 3, 4, 5], [1.0, 2.0, 3.0, -4.0, 0.0])
    dv0 = dgb.Vector.from_vector(v)
    dv1 = dgb.concat_vectors([
        dgb.Vector.from_vector(gb.Vector.from_values([0, 1], [1.0, 2.0])),
        dgb.Vector.from_vector(gb.Vector.from_values([1, 2, 3], [3.0, -4.0, 0.0])),
    ])
    return v, (dv0, dv1)


def test_new():
    v = gb.Vector.new(int)
    dA = dgb.Vector.new(int)
    compare(lambda x: x, v, dA)
    compare(lambda x: x.size, v, dA, compute=False)
    compare(lambda x: x.shape, v, dA, compute=False)
    v = gb.Vector.new(float, 3)
    dA = dgb.Vector.new(float, 3)
    compare(lambda x: x, v, dA)
    compare(lambda x: x.size, v, dA, compute=False)
    compare(lambda x: x.shape, v, dA, compute=False)
    o = object()
    compare(lambda x, y: type(x).new(y), (v, o), (dA, o), errors=True)


def test_dup(As):
    A, dAs = As
    for dA in dAs:
        compare(lambda x: x.dup(), v, dA)
        compare(lambda x: x.dup(dtype=dtypes.FP64), v, dA)
        o = object()
        compare(lambda x, y: x.dup(y), (v, o), (dA, o), errors=True)


@pytest.mark.slow
def test_isequal_isclose(As, Bs):
    o = object()
    for method_name in ['isequal', 'isclose']:
        v = As[0]
        w = Bs[0]
        for dA in As[1]:
            compare(lambda x, y: getattr(x, method_name)(y), (v, v), (dA, dA))
            compare(lambda x: getattr(x, method_name)(o), v, dA, errors=True)
            for dB in Bs[1]:
                compare(lambda x, y: getattr(x, method_name)(y), (v, w), (dA, dB))


def test_nvals(As):
    A, dAs = As
    for dA in dAs:
        compare(lambda x: x.nvals, v, dA)
    v = gb.Vector.new(int)
    dA = dgb.Vector.new(int)
    compare(lambda x: x.nvals, v, dA)


def test_clear(As):

    def f(x):
        x.clear()
        return x

    A, dAs = As
    compare(f, v, dAs[0])


def test_ewise(As, Bs):
    v = As[0]
    w = Bs[0]
    binfunc = lambda x, y: getattr(x, method_name)(y, op, require_monoid=False).new()
    for op in [gb.monoid.plus, gb.binary.plus]:
        for method_name in ['ewise_add', 'ewise_mult']:

            def f(w, x, y):
                w << getattr(x, method_name)(y, op)
                return w

            errors = method_name == 'ewise_add' and op is gb.binary.plus
            compute = not errors
            funcs = [
                lambda x, y: getattr(x, method_name)(y, op).new(),
                lambda x, y: getattr(x, method_name)(y, op).new(dtype=dtypes.FP64),
                lambda x, y: getattr(x, method_name)(y, op).new(mask=y.S),
                lambda x, y: getattr(x, method_name)(y, op).new(mask=y.V),
                lambda x, y: getattr(x, method_name)(y, op).new(mask=~x.S),
                lambda x, y: getattr(x, method_name)(y, op).new(mask=~x.V),
            ]
            for dA in As[1]:
                for func in funcs:
                    compare(func, (v, v), (dA, dA), errors=errors, compute=compute)
                if method_name == 'ewise_add':
                    compare(binfunc, (v, v), (dA, dA))
                compare(f, (v.dup(), v, v), (dA.dup(), dA, dA), errors=errors, compute=compute)
                for dB in Bs[1]:
                    for func in funcs:
                        compare(func, (v, w), (dA, dB), errors=errors, compute=compute)
                    if method_name == 'ewise_add':
                        compare(binfunc, (v, v), (dA, dA))
                    compare(f, (v.dup(), v, w), (dA.dup(), dA, dB), errors=errors, compute=compute)
                    compare(f, (w.dup(), v, w), (dB.dup(), dA, dB), errors=errors, compute=compute)


def test_reduce(As):
    A, dAs = As

    def f0(x, y):
        x << y.reduce()
        return x

    def f1(x, y):
        x() << y.reduce()
        return x

    def f2(x, y):
        x(accum=gb.binary.plus) << y.reduce()
        return x

    for dA in dAs:
        compare(lambda x: x.reduce().new(), v, dA)
        compare(lambda x: x.reduce(gb.monoid.max).new(), v, dA)
        compare(lambda x: x.reduce().new(dtype=dtypes.FP64), v, dA)
        compare(lambda x: x.reduce(gb.binary.plus).new(), v, dA, errors=True)
        for i, f in enumerate([f0, f1, f2]):
            s = gb.Scalar.new(int)
            ds = dgb.Scalar.from_value(s.dup())
            compare(f, (s, v), (ds, dA))

            s = gb.Scalar.from_value(100)
            ds = dgb.Scalar.from_value(s.dup())
            compare(f, (s, v), (ds, dA))

            s = gb.Scalar.new(float)
            ds = dgb.Scalar.from_value(s.dup())
            compare(f, (s, v), (ds, dA))

            if f is not f2:  # XXX: uncomment when updated to SS 3.3.1
                s = gb.Scalar.from_value(1.23)
                ds = dgb.Scalar.from_value(s.dup())
                compare(f, (s, v), (ds, dA))


def test_apply(As):
    A, dAs = As

    def f(x):
        y = type(x).new(x.dtype, x.size)
        y << x.apply(gb.unary.abs)
        return y

    for dA in dAs:
        compare(lambda x: x.apply(gb.unary.abs).new(), v, dA)
        compare(lambda x: x.apply(gb.unary.abs).new(dtype=float), v, dA)
        compare(lambda x: x.apply(gb.binary.plus).new(), v, dA, errors=True)
        compare(f, v.dup(), dA.dup())


def test_update(As, Bs):
    A, dAs = As
    B, dBs = Bs

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

    # TODO: add f5 when the next version of SuiteSparse:GraphBLAS is released
    # It's not tested now, because there is a bug: https://github.com/DrTimothyAldenDavis/GraphBLAS/issues/7
    for f in [f0, f1, f2, f3, f4]:
        for dA in dAs:
            for dB in dBs:
                print(f.__name__, v.dtype, w.dtype)
                compare(f, (v.dup(), w.dup()), (dA.dup(), dB.dup()))
                compare(f, (v.dup(dtype=float), w.dup()), (dA.dup(dtype=float), dB.dup()))


def test_extract(As, Bs):
    A, dAs = As
    B, dBs = Bs

    for index in [
        [0, 3, 1, 4, 2, 5],
        [0, 5, 5, 1, 2, 0],
        slice(None),
        slice(None, None, -1),
        [0] * 6,
    ]:
        def f(x, y):
            x << y[index]
            return x

        for dA in dAs:
            compare(lambda x: x[index].new(), v, dA)
            compare(lambda x: x[index].new(dtype=float), v, dA)
            for dB in dBs:
                compare(f, (v.dup(), w), (dA.dup(), dB))
                compare(f, (v.dup(dtype=float), w), (dA.dup(dtype=float), dB))


def test_attrs(As):
    A, dAs = As
    dA = dAs[0]
    assert set(dir(v)) - set(dir(dA)) == {
        '__del__',  # TODO
        '_assign_element', '_extract_element', '_is_scalar', '_prep_for_assign',
        '_prep_for_extract', 'gb_obj', 'show',
    }
    assert set(dir(dA)) - set(dir(v)) == {
        '_delayed', '_meta', '_optional_dup',
        'compute', 'from_vector', 'from_delayed', 'persist', 'visualize',
    }
"""
