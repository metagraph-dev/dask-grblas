import pytest
from functools import partial
from dask_grblas.functools import flexible_partial, skip


def func(a, b, c, d, e, f):
    return a, b, c, d, e, f


def funk(a, b, c, d, e, f, ka="a", kb="b", kc="c"):
    return a, b, c, d, e, f, ka, kb, kc


def test_flexible_partial():
    # without keywords
    part_func = flexible_partial(func, skip, 2, skip, skip, 5)
    result = part_func(1, 3, 4, 6)
    assert result == (1, 2, 3, 4, 5, 6)

    # with keywords
    part_funk = flexible_partial(funk, skip, 2, skip, skip, 5, kb="B")
    result = part_funk(1, 3, 4, 6, kc="C")
    assert result == (1, 2, 3, 4, 5, 6, "a", "B", "C")

    # apply a 2nd `flexible_partial` on first `flexible_partial`:
    part_funk2 = flexible_partial(part_funk, 1, skip, 4, ka="A")
    result = part_funk2(3, 6, kc="C")
    assert result == (1, 2, 3, 4, 5, 6, "A", "B", "C")

    # or apply a `partial` on first `flexible_partial`:
    part_funk2 = partial(part_funk, 1, 3, ka="A")
    result = part_funk2(4, 6, kc="C")
    assert result == (1, 2, 3, 4, 5, 6, "A", "B", "C")

    # or apply a `flexible_partial` on a `partial`:
    part_funk = partial(funk, 1, 2, kb="B")
    part_funk2 = flexible_partial(part_funk, skip, 4, ka="A")
    result = part_funk2(3, 5, 6, kc="C")
    assert result == (1, 2, 3, 4, 5, 6, "A", "B", "C")
