import pytest
from dask_grblas.functools import flexible_partial, skip


def func(a, b, c, d, e, f):
    return a, b, c, d, e, f


def funk(a, b, c, d, e, f, ka="a", kb="b", kc="c"):
    return a, b, c, d, e, f, ka, kb, kc


@pytest.mark.parametrize(
    "specs",
    [
        [[False, True, False, False, True], [True, False, True]],
        [[1, 4], [0, 2]],
    ],
)
def test_flexible_partial_specs_sans_args(specs):
    specs0, specs1 = specs
    with pytest.raises(ValueError):
        _ = flexible_partial(func, specs0, 1, 2, 3, 4, 5)
    with pytest.raises(ValueError):
        _ = flexible_partial(func, specs0, 1)

    # without keyword arguments:
    part_func = flexible_partial(func, specs0, 2, 5)
    result = part_func(1, 3, 4, 6)
    assert result == (1, 2, 3, 4, 5, 6)

    # with keyword arguments:
    part_funk = flexible_partial(funk, specs0, 2, 5, kb="B")
    result = part_funk(1, 3, 4, 6, kc="C")
    assert result == (1, 2, 3, 4, 5, 6, "a", "B", "C")

    # apply a 2nd flexible_partial on first flexible_partial:
    part_funk2 = flexible_partial(part_funk, specs1, 1, 4, ka="A")
    result = part_funk2(3, 6, kc="C")
    assert result == (1, 2, 3, 4, 5, 6, "A", "B", "C")


@pytest.mark.parametrize(
    "specs",
    [
        (skip, 2, skip, skip, 5),
        [(1, 2), (4, 5)],
    ],
)
def test_flexible_partial_specs_with_args(specs):
    part_func = flexible_partial(func, specs)
    result = part_func(1, 3, 4, 6)
    assert result == (1, 2, 3, 4, 5, 6)

    part_funk = flexible_partial(funk, specs, 2, 5, kb="B")
    result = part_funk(1, 3, 4, 6, kc="C")
    assert result == (1, 2, 3, 4, 5, 6, "a", "B", "C")
