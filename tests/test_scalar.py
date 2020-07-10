import pytest
import grblas as gb
import dask_grblas as dgb
from grblas import dtypes
from pytest import raises
from .utils import compare


def test_new():
    s = gb.Scalar.new(int)
    ds = dgb.Scalar.new(int)
    compare(lambda x: x, s, ds)
    s = gb.Scalar.new(float)
    ds = dgb.Scalar.new(float)
    compare(lambda x: x, s, ds)
    o = object()
    compare(lambda x, y: type(x).new(y), (s, o), (ds, o), errors=True)


def test_dup():
    s = gb.Scalar.from_value(5)
    ds = dgb.Scalar.from_value(5)
    ds2 = dgb.Scalar.from_value(s)
    compare(lambda x: x, s, ds)
    compare(lambda x: x, s, ds2)
    compare(lambda x: x.dup(), s, ds)
    compare(lambda x: x.dup(), s, ds2)
    compare(lambda x: x.dup(dtype=dtypes.FP64), s, ds)
    compare(lambda x: x.dup(dtype=dtypes.FP64), s, ds2)
    o = object()
    compare(lambda x, y: x.dup(y), (s, o), (ds, o), errors=True)
    # testing compare
    with raises(AssertionError):
        compare(lambda x: x, s, dgb.Scalar.from_value(6))
    with raises(AssertionError):
        compare(lambda x: x, s, dgb.Scalar.from_value(5, dtype=dtypes.FP64))


@pytest.mark.slow
def test_isequal_isclose():
    values = [
        (gb.Scalar.from_value(5), gb.Scalar.from_value(5)),
        (gb.Scalar.from_value(5), gb.Scalar.from_value(6)),
        (gb.Scalar.from_value(5), gb.Scalar.from_value(5.0)),
        (gb.Scalar.from_value(None, dtype=int), gb.Scalar.from_value(5)),
        (gb.Scalar.from_value(None, dtype=int), gb.Scalar.from_value(None, dtype=int)),
        (gb.Scalar.from_value(None, dtype=int), gb.Scalar.from_value(None, dtype=float)),
    ]
    o = object()
    for s, t in values:
        for method_name in ['isequal', 'isclose']:
            ds = dgb.Scalar.from_value(s)
            dt = dgb.Scalar.from_value(t)
            compare(
                lambda x, y: getattr(x, method_name)(y),
                (s, t),
                (ds, dt),
            )
            compare(
                lambda x, y: getattr(x, method_name)(y, check_dtype=True),
                (s, t),
                (ds, dt),
            )
            compare(lambda x, y: x == y, (s, t), (ds, dt), compute=False)
            compare(lambda x: getattr(x, method_name)(o), s, ds, errors=True)

    s = gb.Scalar.from_value(5.0)
    t = gb.Scalar.from_value(5.000000001)
    ds = dgb.Scalar.from_value(s)
    dt = dgb.Scalar.from_value(t)
    assert s.isclose(t)
    compare(lambda x, y: x.isclose(y), (s, t), (ds, dt))
    assert not s.isclose(None)
    compare(lambda x, y: x.isclose(y), (s, None), (ds, None))
    assert not s.isequal(None)
    compare(lambda x, y: x.isequal(y), (s, None), (ds, None))
    assert not s.isclose(t, rel_tol=1e-10)
    compare(lambda x, y: x.isclose(y, rel_tol=1e-10), (s, t), (ds, dt))
    assert s.isclose(t, rel_tol=1e-10, abs_tol=1e-8)
    compare(lambda x, y: x.isclose(y, rel_tol=1e-10, abs_tol=1e-8), (s, t), (ds, dt))
    compare(lambda x, y: x.isequal(y, check_dtype=True), (s, 5), (ds, 5))
    compare(lambda x, y: x.isclose(y, check_dtype=True), (s, 5), (ds, 5))


def test_nvals():
    s = gb.Scalar.from_value(1)
    ds = dgb.Scalar.from_value(s)
    compare(lambda x: x.nvals, s, ds)
    s = gb.Scalar.from_value(None, dtype=int)
    ds = dgb.Scalar.from_value(s)
    compare(lambda x: x.nvals, s, ds)
    # Test creation with PythonScalar
    compare(lambda x: type(x).from_value(x.nvals), s, ds)


def test_value():
    s = gb.Scalar.from_value(3)
    ds = dgb.Scalar.from_value(s)
    compare(lambda x: x.value, s, ds)

    def f(x, y):
        x.value = y
        return x

    compare(f, (s, 4), (ds, 4))
    s2 = gb.Scalar.from_value(5)
    ds2 = dgb.Scalar.from_value(s)
    # compare(f, (s, s2), (ds, ds2))  # not yet supported in grblas
    compare(f, (s, s2.value), (ds, ds2.value))
    compare(f, (s, s.nvals), (ds, ds.nvals))
    compare(f, (s, None), (ds, None))
    o = object()
    compare(f, (s, o), (ds, o), errors=True)


def test_bool():
    values = [
        gb.Scalar.from_value(0),
        gb.Scalar.from_value(10.1),
        gb.Scalar.from_value(True),
        gb.Scalar.from_value(False),
        gb.Scalar.from_value(None, dtype=int),
    ]
    for s in values:
        ds = dgb.Scalar.from_value(s)
        compare(lambda x: bool(x), s, ds, compute=False)


def test_clear():
    s = gb.Scalar.from_value(4)
    ds = dgb.Scalar.from_value(s)

    def f(x):
        x.clear()
        return x

    compare(f, s, ds)


def test_is_empty():
    s = gb.Scalar.from_value(4)
    ds = dgb.Scalar.from_value(s)
    compare(lambda x: x.is_empty, s, ds)
    s.clear()
    ds.clear()
    compare(lambda x: x.is_empty, s, ds)
    s = gb.Scalar.from_value(None, dtype=float)
    ds = dgb.Scalar.from_value(s)
    compare(lambda x: x.is_empty, s, ds)


def test_update():

    def f1(x, y):
        x.update(y)
        return x

    def f2(x, y):
        x << y
        return x

    for f in [f1, f2]:
        s = gb.Scalar.from_value(6)
        ds = dgb.Scalar.from_value(s)
        s2 = gb.Scalar.from_value(7)
        ds2 = dgb.Scalar.from_value(s)

        compare(f, (s, s2), (ds, ds2))
        compare(f, (s, 1), (ds, 1))
        compare(f, (s, None), (ds, None))

        v = gb.Vector.from_values([0, 2], [0, 2])
        dv = dgb.Vector.from_vector(v)
        compare(f, (s, v[0]), (ds, dv[0]))


def test_attrs():
    s = gb.Scalar.from_value(3)
    ds = dgb.Scalar.from_value(s)
    assert set(dir(s)) - set(dir(ds)) == {
        '_is_empty', '_assign_element', '_extract_element', '_is_scalar', '_prep_for_assign',
        '_prep_for_extract', 'gb_obj', 'show',
    }
    assert set(dir(ds)) - set(dir(s)) == {
        '_delayed', '_meta', '_optional_dup',
        'compute', 'from_delayed', 'persist', 'visualize',
    }
