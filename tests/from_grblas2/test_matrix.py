import inspect
import itertools
import pickle
import sys
import weakref

import dask.array as da
import dask_grblas
import grblas
import numpy as np
import pytest
from grblas import agg, binary, dtypes, monoid, semiring, unary
from grblas.exceptions import (
    DimensionMismatch,
    EmptyObject,
    IndexOutOfBound,
    InvalidValue,
    NotImplementedException,
    OutputNotEmpty,
)
from numpy.testing import assert_array_equal

from .conftest import autocompute, compute

from dask_grblas import Matrix, Scalar, Vector  # isort:skip
from dask_grblas.base import is_DOnion, like_dOnion


@pytest.fixture
def A():
    #    0 1 2 3 4 5 6
    # 0 [- 2 - 3 - - -]
    # 1 [- - - - 8 - 4]
    # 2 [- - - - - 1 -]
    # 3 [3 - 3 - - - -]
    # 4 [- - - - - 7 -]
    # 5 [- - 1 - - - -]
    # 6 [- - 5 7 3 - -]
    data = [
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4],
    ]
    return Matrix.from_values(*data)


@pytest.fixture
def A_dask():
    #    0 1 2 3 4 5 6
    # 0 [- 2 - 3 - - -]
    # 1 [- - - - 8 - 4]
    # 2 [- - - - - 1 -]
    # 3 [3 - 3 - - - -]
    # 4 [- - - - - 7 -]
    # 5 [- - 1 - - - -]
    # 6 [- - 5 7 3 - -]
    data = [
        [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
        [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
        [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4],
    ]
    data = [da.from_array(np.array(a, dtype=np.int64)) for a in data]
    return Matrix.from_values(*data)


@pytest.fixture
def As(A, A_dask):
    return [A, A_dask]


@pytest.fixture
def A_chunks():
    return [7, 3]


@pytest.fixture
def v():
    data = [[1, 3, 4, 6], [1, 1, 2, 0]]
    return Vector.from_values(*data)


@pytest.fixture
def v_dask():
    data = [[1, 3, 4, 6], [1, 1, 2, 0]]
    data = [da.from_array(a) for a in data]
    return Vector.from_values(*data)


@pytest.fixture
def vs(v, v_dask):
    return [v, v_dask]


def test_new():
    C = Matrix.new(dtypes.INT8, 17, 12)
    assert C.dtype == "INT8"
    assert C.nvals == 0
    assert C.nrows == 17
    assert C.ncols == 12


def test_dup(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            C = A.dup()
            assert C is not A
            assert C.dtype == A.dtype
            assert C.nvals == A.nvals
            assert C.nrows == A.nrows
            assert C.ncols == A.ncols
            # Ensure they are not the same backend object
            A[0, 0] = 1000
            assert A[0, 0].new() == 1000
            assert C[0, 0].new() != 1000

    # extended functionality
    Ds = [Matrix.from_values([0, 1], [0, 1], [0, 2.5], dtype=dtypes.FP64)]
    Ds.append(
        Matrix.from_values(
            da.from_array([0, 1]), da.from_array([0, 1]), da.from_array([0, 2.5]), dtype=dtypes.FP64
        )
    )
    for D_ in Ds:
        for chunks in A_chunks:
            D = D_.dup()
            D.rechunk(chunks=chunks, inplace=True)
            E = D.dup(dtype=dtypes.INT64)
            assert E.isequal(
                Matrix.from_values([0, 1], [0, 1], [0, 2], dtype=dtypes.INT64), check_dtype=True
            )
            E = D.dup(mask=D.V)
            assert E.isequal(
                Matrix.from_values([1], [1], [2.5], dtype=dtypes.FP64), check_dtype=True
            )
            E = D.dup(dtype=dtypes.INT64, mask=D.V)
            assert E.isequal(
                Matrix.from_values([1], [1], [2], dtype=dtypes.INT64), check_dtype=True
            )


def test_from_values():
    C = Matrix.from_values([0, 1, 3], [1, 1, 2], [True, False, True])
    assert C.nrows == 4
    assert C.ncols == 3
    assert C.nvals == 3
    assert C.dtype == bool
    C2 = Matrix.from_values([0, 1, 3], [1, 1, 2], [12.3, 12.4, 12.5], nrows=17, ncols=3)
    assert C2.nrows == 17
    assert C2.ncols == 3
    assert C2.nvals == 3
    assert C2.dtype == float
    C3 = Matrix.from_values([0, 1, 1], [2, 1, 1], [1, 2, 3], nrows=10, dup_op=binary.times)
    assert C3.nrows == 10
    assert C3.ncols == 3
    assert C3.nvals == 2  # duplicates were combined
    assert C3.dtype == int
    assert C3[1, 1].new() == 6  # 2*3
    C3monoid = Matrix.from_values([0, 1, 1], [2, 1, 1], [1, 2, 3], nrows=10, dup_op=monoid.times)
    assert C3.isequal(C3monoid)

    with pytest.raises(ValueError, match="Duplicate indices found"):
        # Duplicate indices requires a dup_op
        Matrix.from_values([0, 1, 1], [2, 1, 1], [True, True, True])
    with pytest.raises(IndexOutOfBound):
        # Specified ncols can't hold provided indexes
        Matrix.from_values([0, 1, 3], [1, 1, 2], [12.3, 12.4, 12.5], nrows=17, ncols=2)
    with pytest.raises(ValueError, match="No row indices provided. Unable to infer nrows."):
        Matrix.from_values([], [], [])

    # Changed: Assume empty value is float64 (like numpy)
    # with pytest.raises(ValueError, match="No values provided. Unable to determine type"):
    empty1 = Matrix.from_values([], [], [], nrows=3, ncols=4)
    assert empty1.dtype == dtypes.FP64
    assert empty1.nrows == 3
    assert empty1.ncols == 4
    assert empty1.nvals == 0

    with pytest.raises(ValueError, match="Unable to infer"):
        Matrix.from_values([], [], [], dtype=dtypes.INT64)
    with pytest.raises(ValueError, match="Unable to infer"):
        # could also raise b/c rows and columns are different sizes
        Matrix.from_values([0], [], [0], dtype=dtypes.INT64)
    C4 = Matrix.from_values([], [], [], nrows=3, ncols=4, dtype=dtypes.INT64)
    C5 = Matrix.new(dtypes.INT64, nrows=3, ncols=4)
    assert C4.isequal(C5, check_dtype=True)

    with pytest.raises(
        ValueError, match="`rows` and `columns` and `values` lengths must match: 1, 2, 1"
    ):
        Matrix.from_values([0], [1, 2], [0])


def test_from_values_dask():
    rows = da.from_array(np.array([0, 1, 3]))
    cols = da.from_array(np.array([1, 1, 2]))
    vals = da.from_array(np.array([True, False, True]))
    C = Matrix.from_values(rows, cols, vals)
    assert C.nrows == 4
    assert C.ncols == 3
    assert C.nvals == 3
    assert C.dtype == bool

    vals = da.from_array(np.array([12.3, 12.4, 12.5]))
    C2 = Matrix.from_values(rows, cols, vals, nrows=17, ncols=3)
    assert C2.nrows == 17
    assert C2.ncols == 3
    assert C2.nvals == 3
    assert C2.dtype == float

    rows = da.from_array(np.array([0, 1, 1]))
    cols = da.from_array(np.array([2, 1, 1]))
    vals = da.from_array(np.array([1, 2, 3], dtype=np.int64))
    C3 = Matrix.from_values(rows, cols, vals, nrows=10, dup_op=binary.times)
    assert C3.nrows == 10
    assert C3.ncols == 3
    assert C3.nvals == 2  # duplicates were combined
    assert C3.dtype == int
    assert C3[1, 1].new() == 6  # 2*3
    C3monoid = Matrix.from_values(rows, cols, vals, nrows=10, dup_op=monoid.times)
    assert C3.isequal(C3monoid)

    vals = da.from_array(np.array([True, True, True]))
    with pytest.raises(ValueError, match="Duplicate indices found"):
        # Duplicate indices requires a dup_op
        Matrix.from_values(rows, cols, vals).compute()

    rows = da.from_array(np.array([0, 1, 3]))
    cols = da.from_array(np.array([1, 1, 2]))
    vals = da.from_array(np.array([12.3, 12.4, 12.5]))
    with pytest.raises(IndexOutOfBound):
        # Specified ncols can't hold provided indexes
        Matrix.from_values(rows, cols, vals, nrows=17, ncols=2).compute()

    empty_da = da.from_array(np.array([]))
    with pytest.raises(ValueError, match="No row indices provided. Unable to infer nrows."):
        Matrix.from_values(empty_da, empty_da, empty_da)

    # Changed: Assume empty value is float64 (like numpy)
    # with pytest.raises(ValueError, match="No vals provided. Unable to determine type"):
    empty1 = Matrix.from_values(empty_da, empty_da, empty_da, nrows=3, ncols=4)
    assert empty1.dtype == dtypes.FP64
    assert empty1.nrows == 3
    assert empty1.ncols == 4
    assert empty1.nvals == 0

    with pytest.raises(ValueError, match="Unable to infer"):
        Matrix.from_values(empty_da, empty_da, empty_da, dtype=dtypes.INT64)

    zero_da = da.from_array(np.array([0]))
    with pytest.raises(ValueError, match="Unable to infer"):
        # could also raise b/c rows and columns are different sizes
        Matrix.from_values(zero_da, empty_da, zero_da, dtype=dtypes.INT64)

    C4 = Matrix.from_values(empty_da, empty_da, empty_da, nrows=3, ncols=4, dtype=dtypes.INT64)
    C5 = Matrix.new(dtypes.INT64, nrows=3, ncols=4)
    assert C4.isequal(C5, check_dtype=True)

    cols = da.from_array(np.array([1, 2]))
    with pytest.raises(
        ValueError, match="`rows` and `columns` and `values` lengths must match: 1, 2, 1"
    ):
        Matrix.from_values(zero_da, cols, zero_da)


def test_from_values_scalar():
    Cs = [Matrix.from_values([0, 1, 3], [1, 1, 2], 7)]
    Cs.append(
        Matrix.from_values(
            da.from_array([0, 1, 3]),
            da.from_array([1, 1, 2]),
            7,
        )
    )
    for C in Cs:
        assert C.nrows == 4
        assert C.ncols == 3
        assert C.nvals == 3
        assert C.dtype == dtypes.INT64
        # assert C.ss.is_iso
        assert C.reduce_scalar(monoid.any).new() == 7

    # iso drumps duplicates
    C = Matrix.from_values([0, 1, 3, 0], [1, 1, 2, 1], 7)
    Cs.append(
        Matrix.from_values(
            da.from_array([0, 1, 3, 0]),
            da.from_array([1, 1, 2, 1]),
            7,
        )
    )
    for C in Cs:
        assert C.nrows == 4
        assert C.ncols == 3
        assert C.nvals == 3
        assert C.dtype == dtypes.INT64
        # assert C.ss.is_iso
        assert C.reduce_scalar(monoid.any).new() == 7
        with pytest.raises(ValueError, match="dup_op must be None"):
            Matrix.from_values([0, 1, 3, 0], [1, 1, 2, 1], 7, dup_op=binary.plus)


def test_clear(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            A.clear()
            assert A.nvals == 0
            assert A.nrows == 7
            assert A.ncols == 7


def test_resize(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            assert A.nrows == 7
            assert A.ncols == 7
            assert A.nvals.compute() == 12
            A.resize(10, 11)
            assert A.nrows == 10
            assert A.ncols == 11
            assert A.nvals == 12
            assert compute(A[9, 9].new().value) is None
            A.resize(4, 1)
            assert A.nrows == 4
            assert A.ncols == 1
            assert A.nvals == 1

            A = A_.dup()
            assert A.nrows == 7
            assert A.ncols == 7
            assert A.nvals == 12
            A.resize(6, 11, chunks=4)
            assert A.nrows == 6
            assert A.ncols == 11
            assert A.nvals == 9
            if not A.is_dOnion:
                assert A._delayed.chunks == ((4, 2), (4, 4, 3))
            else:
                assert A._delayed.deep_extract(None, lambda x: x._delayed.chunks) == (
                    (4, 2),
                    (4, 4, 3),
                )
            assert compute(A[3, 2].new().value) == 3
            assert compute(A[5, 7].new().value) is None

            A = A_.dup()
            A.resize(11, 3, chunks=4)
            assert A.nrows == 11
            assert A.ncols == 3
            assert A.nvals == 5
            if type(A._delayed) is da.Array:
                assert A._delayed.chunks == ((4, 4, 3), (3,))
            else:
                assert A._delayed.deep_extract(None, lambda x: x._delayed.chunks) == (
                    (4, 4, 3),
                    (3,),
                )
            assert compute(A[3, 2].new().value) == 3
            assert compute(A[7, 2].new().value) is None


def test_rechunk(As, A_chunks):
    for A_ in As:
        A = A_.dup()
        for chunks in A_chunks + A_chunks[::-1]:
            A.rechunk(chunks=chunks, inplace=True)
            assert A.nrows == 7
            assert A.ncols == 7
            assert A.nvals == 12


def test_nrows(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            assert A.nrows == 7


def test_ncols(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            assert A.ncols == 7


def test_nvals(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            assert A.nvals == 12


def test_build(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            assert A.nvals == 12
            A.clear()
            A.build([0, 6], [0, 1], [1, 2])
            assert A.nvals == 2
            with pytest.raises(OutputNotEmpty):
                A.build([1, 5], [2, 3], [3, 4])
            assert A.nvals == 2  # nothing should be modified
            # We can clear though
            A.build([1, 2, 5], [1, 2, 3], [2, 3, 4], clear=True)
            assert A.nvals == 3
            A.clear()
            if is_DOnion(A._delayed):
                A.build([0, 11], [0, 0], [1, 1])
                with pytest.raises(IndexOutOfBound):
                    A.compute()
            else:
                with pytest.raises(IndexOutOfBound):
                    A.build([0, 11], [0, 0], [1, 1])

            B = Matrix.new(int, nrows=2, ncols=2)
            B.build([0, 11], [0, 0], [1, 1], nrows=12)
            assert B.isequal(Matrix.from_values([0, 11], [0, 0], [1, 1], ncols=2))
            C = Matrix.new(int, nrows=2, ncols=2)
            C.build([0, 0], [0, 11], [1, 1], ncols=12)
            assert C.isequal(Matrix.from_values([0, 0], [0, 11], [1, 1], nrows=2))


def test_build_scalar(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        assert A.nvals == 12
        with pytest.raises(OutputNotEmpty):
            A.ss.build_scalar([1, 5], [2, 3], 3)
        A.clear()
        A.ss.build_scalar([0, 6], [0, 1], 1)
        assert A.nvals == 2
        # assert A.ss.is_iso
        A.clear()
        with pytest.raises(ValueError, match="lengths must match"):
            A.ss.build_scalar([0, 6], [0, 1, 2], 1)
        with pytest.raises(EmptyObject):
            A.ss.build_scalar([0, 5], [0, 1], None)


def test_extract_values(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            rows, cols, vals = A.to_values(dtype=int)
            rcv = set(
                zip(
                    rows.compute(),
                    cols.compute(),
                    vals.compute(),
                )
            )
            expected = set(
                zip(
                    (0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6),
                    (1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4),
                    (2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3),
                )
            )
            assert rcv == expected
            assert rows.dtype == np.uint64
            assert cols.dtype == np.uint64
            assert vals.dtype == np.int64
            Trows, Tcols, Tvals = A.T.to_values(dtype=float)
            np.testing.assert_array_equal(rows.compute(), Tcols.compute())
            np.testing.assert_array_equal(cols.compute(), Trows.compute())
            np.testing.assert_array_equal(vals.compute(), Tvals.compute())
            assert Trows.dtype == np.uint64
            assert Tcols.dtype == np.uint64
            assert Tvals.dtype == np.float64


def test_extract_element(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            assert A[3, 0].new() == 3
            assert A[1, 6].new() == 4
            with pytest.raises(TypeError, match="enable automatic"):
                A[1, 6].value
            assert A.T[6, 1].new() == 4
            s = A[0, 0].new()
            assert compute(s.value) is None
            assert s.dtype == "INT64"
            s = A[1, 6].new(dtype=float)
            assert s.value == 4.0
            assert s.dtype == "FP64"


def test_set_element(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            assert compute(A[1, 1].new().value) is None
            assert A[3, 0].new() == 3
            A[1, 1].update(21)
            A[3, 0] << -5
            assert A[1, 1].new() == 21
            assert A[3, 0].new() == -5


def test_remove_element(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            assert A[3, 0].new() == 3
            del A[3, 0]
            assert compute(A[3, 0].new().value) is None
            assert A[6, 3].new() == 7
            with pytest.raises(TypeError, match="Remove Element only supports"):
                del A[3:5, 3]


def test_mxm(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            C = A.mxm(A, semiring.plus_times).new()
            result = Matrix.from_values(
                [0, 0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 3, 4, 5, 6, 6, 6],
                [0, 2, 4, 6, 2, 3, 4, 5, 2, 1, 3, 5, 2, 5, 0, 2, 5],
                [9, 9, 16, 8, 20, 28, 12, 56, 1, 6, 9, 3, 7, 1, 21, 21, 26],
            )
            assert C.isequal(result)


def test_mxm_transpose(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            C = A.dup()
            C << A.mxm(A.T, semiring.plus_times)
            result = Matrix.from_values(
                [0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6],
                [0, 6, 1, 6, 2, 4, 3, 5, 6, 2, 4, 3, 5, 6, 0, 1, 3, 5, 6],
                [13, 21, 80, 24, 1, 7, 18, 3, 15, 7, 49, 3, 1, 5, 21, 24, 15, 5, 83],
            )
            assert C.isequal(result)
            C << A.T.mxm(A, semiring.plus_times)
            result2 = Matrix.from_values(
                [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 6, 6],
                [0, 2, 1, 3, 0, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 6, 5, 4, 6],
                [9, 9, 4, 6, 9, 35, 35, 15, 6, 35, 58, 21, 15, 21, 73, 32, 50, 32, 16],
            )
            assert C.isequal(result2)


def test_mxm_nonsquare():
    A = Matrix.from_values([0, 0, 0], [0, 2, 4], [1, 2, 3], nrows=1, ncols=5)
    B = Matrix.from_values([0, 2, 4], [0, 0, 0], [10, 20, 30], nrows=5, ncols=1)
    C = Matrix.new(A.dtype, nrows=1, ncols=1)
    C << A.mxm(B, semiring.max_plus)
    assert C[0, 0].new() == 33
    C1 = A.mxm(B, semiring.max_plus).new()
    assert C1.isequal(C)
    C2 = A.T.mxm(B.T, semiring.max_plus).new()
    assert C2.nrows == 5
    assert C2.ncols == 5

    A = Matrix.from_values(
        da.from_array([0, 0, 0]),
        da.from_array([0, 2, 4]),
        da.from_array([1, 2, 3]),
        nrows=1,
        ncols=5,
    )
    B = Matrix.from_values([0, 2, 4], [0, 0, 0], [10, 20, 30], nrows=5, ncols=1)
    C = Matrix.new(A.dtype, nrows=1, ncols=1)
    C << A.mxm(B, semiring.max_plus)
    assert C[0, 0].new() == 33
    C1 = A.mxm(B, semiring.max_plus).new()
    assert C1.isequal(C)
    C2 = A.T.mxm(B.T, semiring.max_plus).new()
    assert C2.nrows == 5
    assert C2.ncols == 5


def test_mxm_mask(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            val_mask = Matrix.from_values(
                [0, 3, 4], [2, 3, 2], [True, True, True], nrows=7, ncols=7
            )
            struct_mask = Matrix.from_values([0, 3, 4], [2, 3, 2], [1, 0, 0], nrows=7, ncols=7)
            C = A.dup()
            C(val_mask.V) << A.mxm(A, semiring.plus_times)
            result = Matrix.from_values(
                [0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 4, 5, 6, 6, 6],
                [1, 2, 3, 4, 6, 5, 0, 2, 3, 2, 5, 2, 2, 3, 4],
                [2, 9, 3, 8, 4, 1, 3, 3, 9, 7, 7, 1, 5, 7, 3],
            )
            assert C.isequal(result)
            C = A.dup()
            C(~val_mask.V) << A.mxm(A, semiring.plus_times)
            result2 = Matrix.from_values(
                [0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 5, 6, 6, 6],
                [0, 4, 6, 2, 3, 4, 5, 2, 1, 5, 5, 0, 2, 5],
                [9, 16, 8, 20, 28, 12, 56, 1, 6, 3, 1, 21, 21, 26],
            )
            assert C.isequal(result2)
            C = A.dup()
            C(struct_mask.S, replace=True).update(A.mxm(A, semiring.plus_times))
            result3 = Matrix.from_values([0, 3, 4], [2, 3, 2], [9, 9, 7], nrows=7, ncols=7)
            assert C.isequal(result3)
            C2 = A.mxm(A, semiring.plus_times).new(mask=struct_mask.S)
            assert C2.isequal(result3)
            with pytest.raises(TypeError, match="Mask must indicate"):
                A.mxm(A).new(mask=struct_mask)


def test_mxm_accum(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            A(binary.plus) << A.mxm(A, semiring.plus_times)
            # fmt: off
            result = Matrix.from_values(
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6],
                [0, 1, 2, 3, 4, 6, 2, 3, 4, 5, 6, 2, 5, 0, 1, 2, 3, 5, 2, 5, 2, 5, 0, 2, 3, 4, 5],
                [9, 2, 9, 3, 16, 8, 20, 28, 20, 56, 4, 1, 1, 3, 6, 3, 9, 3, 7, 7, 1, 1, 21, 26, 7, 3, 26],
            )
            # fmt: on
            assert A.isequal(result)


def test_mxv(As, A_chunks, v):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            w = A.mxv(v, semiring.plus_times).new()
            result = Vector.from_values([0, 1, 6], [5, 16, 13])
            assert w.isequal(result)


def test_ewise_mult(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            # Binary, Monoid, and Semiring
            B = Matrix.from_values([0, 0, 5], [1, 2, 2], [5, 4, 8], nrows=7, ncols=7)
            result = Matrix.from_values([0, 5], [1, 2], [10, 8], nrows=7, ncols=7)
            C = A.ewise_mult(B, binary.times).new()
            assert C.isequal(result)
            C() << A.ewise_mult(B, monoid.times)
            assert C.isequal(result)
            with pytest.raises(TypeError, match="Expected type: BinaryOp, Monoid"):
                A.ewise_mult(B, semiring.plus_times)


def test_ewise_add(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            # Binary, Monoid, and Semiring
            B = Matrix.from_values([0, 0, 5], [1, 2, 2], [5, 4, 8], nrows=7, ncols=7)
            result = Matrix.from_values(
                [0, 3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
                [2, 0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                [4, 3, 5, 3, 8, 5, 3, 7, 8, 3, 1, 7, 4],
            )
            with pytest.raises(TypeError, match="require_monoid"):
                A.ewise_add(B, binary.second)
            # surprising that SECOND(x, empty) == x, which is why user
            # must opt-in to using binary ops in ewise_add
            C = A.ewise_add(B, binary.second, require_monoid=False).new()
            assert C.isequal(result)
            C << A.ewise_add(B, monoid.max)
            assert C.isequal(result)
            C << A.ewise_add(B, binary.max)
            assert C.isequal(result)
            with pytest.raises(TypeError, match="Expected type: Monoid"):
                A.ewise_add(B, semiring.max_minus)


def test_extract(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            C = Matrix.new(A.dtype, 3, 4)
            result = Matrix.from_values(
                [0, 0, 1, 2, 2, 2], [0, 2, 1, 1, 2, 3], [2, 3, 3, 5, 7, 3], nrows=3, ncols=4
            )
            C << A[[0, 3, 6], [1, 2, 3, 4]]
            assert C.isequal(result)
            C << A[0::3, 1:5]
            assert C.isequal(result)
            C << A[[0, 3, 6], 1:5:1]
            assert C.isequal(result)
            C2 = A[[0, 3, 6], [1, 2, 3, 4]].new()
            assert C2.isequal(result)


def test_extract_row(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            w = Vector.new(A.dtype, 3)
            result = Vector.from_values([1, 2], [5, 3], size=3)
            w << A[6, [0, 2, 4]]
            assert w.isequal(result)
            w << A[6, :5:2]
            assert w.isequal(result)
            w << A.T[[0, 2, 4], 6]
            assert w.isequal(result)
            w2 = A[6, [0, 2, 4]].new()
            assert w2.isequal(result)
            with pytest.raises(TypeError):
                # Should be list, not tuple (although tuple isn't so bad)
                A[6, (0, 2, 4)]
            w3 = A[6, np.array([0, 2, 4])].new()
            assert w3.isequal(result)
            with pytest.raises(TypeError, match="Invalid dtype"):
                A[6, np.array([0, 2, 4], dtype=float)]
            with pytest.raises(TypeError, match="Invalid number of dimensions"):
                A[6, np.array([[0, 2, 4]])]


def test_extract_column(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            w = Vector.new(A.dtype, 3)
            result = Vector.from_values([1, 2], [3, 1], size=3)
            w << A[[1, 3, 5], 2]
            assert w.isequal(result)
            w << A[1:6:2, 2]
            assert w.isequal(result)
            w << A.T[2, [1, 3, 5]]
            assert w.isequal(result)
            w2 = A[1:6:2, 2].new()
            assert w2.isequal(result)


def test_extract_input_mask():
    # A       M
    # 0 1 2   _ 0 1
    # 3 4 5   2 3 _
    As = [
        Matrix.from_values(
            [0, 0, 0, 1, 1, 1],
            [0, 1, 2, 0, 1, 2],
            [0, 1, 2, 3, 4, 5],
        )
    ]
    As += [
        Matrix.from_values(
            da.from_array([0, 0, 0, 1, 1, 1]),
            da.from_array([0, 1, 2, 0, 1, 2]),
            da.from_array([0, 1, 2, 3, 4, 5]),
        )
    ]
    Ms = [
        Matrix.from_values(
            [0, 0, 1, 1],
            [1, 2, 0, 1],
            [0, 1, 2, 3],
        )
    ]
    Ms += [
        Matrix.from_values(
            da.from_array([0, 0, 1, 1]),
            da.from_array([1, 2, 0, 1]),
            da.from_array([0, 1, 2, 3]),
        )
    ]
    for A_ in As:
        for M_ in Ms:
            A = A_.dup()
            M = M_.dup()
            m = M[0, :].new()
            MT = M.T.new()
            # Matrix structure mask
            result = A[0, [0, 1]].new(input_mask=M.S)
            expected = Vector.from_values([1], [1])
            assert result.isequal(expected)
            # again
            result.clear()
            result(input_mask=M.S) << A[0, [0, 1]]
            assert result.isequal(expected)

            # Vector mask
            result = A[0, [0, 1]].new(input_mask=m.S)
            assert result.isequal(expected)
            # again
            result.clear()
            result(input_mask=m.S) << A[0, [0, 1]]
            assert result.isequal(expected)

            # Matrix value mask
            result = A[0, [1, 2]].new(input_mask=M.V)
            expected = Vector.from_values([1], [2], size=2)
            assert result.isequal(expected)
            # again
            result.clear()
            result(input_mask=M.V) << A[0, [1, 2]]
            assert result.isequal(expected)

            with pytest.raises(
                ValueError, match="Shape of `input_mask` does not match shape of input"
            ):
                A[0, [0, 1]].new(input_mask=MT.S).compute()

            with pytest.raises(
                ValueError, match="Shape of `input_mask` does not match shape of input"
            ):
                m(input_mask=MT.S) << A[0, [0, 1]]
                m.compute()

            with pytest.raises(
                ValueError, match="Size of `input_mask` Vector does not match ncols of Matrix"
            ):
                A[0, [0]].new(input_mask=expected.S).compute()

            m = M[0, :].new()
            with pytest.raises(
                ValueError, match="Size of `input_mask` Vector does not match ncols of Matrix"
            ):
                m(input_mask=expected.S) << A[0, [0]]
                m.compute()

            m = M[0, :].new()
            with pytest.raises(
                ValueError, match="Size of `input_mask` Vector does not match nrows of Matrix"
            ):
                A[[0], 0].new(input_mask=m.S).compute()

            m = M[0, :].new()
            with pytest.raises(
                ValueError, match="Size of `input_mask` Vector does not match nrows of Matrix"
            ):
                m(input_mask=m.S) << A[[0], 0]
                m.compute()

            with pytest.raises(
                TypeError, match="Got Vector `input_mask` when extracting a submatrix from a Matrix"
            ):
                A[[0], [0]].new(input_mask=expected.S).compute()

            with pytest.raises(
                TypeError, match="Got Vector `input_mask` when extracting a submatrix from a Matrix"
            ):
                A(input_mask=expected.S) << A[[0], [0]]
                A.compute()

            A = A_.dup()
            with pytest.raises(
                TypeError, match="mask is not allowed for single element extraction"
            ):
                A[0, 0].new(input_mask=M.S).compute()

            with pytest.raises(
                TypeError, match="mask and input_mask arguments cannot both be given"
            ):
                A[0, [0, 1]].new(input_mask=M.S, mask=expected.S).compute()

            with pytest.raises(
                TypeError, match="mask and input_mask arguments cannot both be given"
            ):
                A(input_mask=M.S, mask=expected.S).compute()

            with pytest.raises(
                TypeError, match=r"Mask must indicate values \(M.V\) or structure \(M.S\)"
            ):
                A[0, [0, 1]].new(input_mask=M).compute()

            with pytest.raises(
                TypeError, match=r"Mask must indicate values \(M.V\) or structure \(M.S\)"
            ):
                A(input_mask=M).compute()

            with pytest.raises(TypeError, match="Mask object must be type Vector"):
                expected[[0, 1]].new(input_mask=M.S).compute()

            with pytest.raises(TypeError, match="Mask object must be type Vector"):
                expected(input_mask=M.S) << expected[[0, 1]]
                expected.compute()

            with pytest.raises(
                TypeError, match=r"new\(\) got an unexpected keyword argument 'input_mask'"
            ):
                A.new(input_mask=M.S).compute()

            with pytest.raises(
                TypeError, match="`input_mask` argument may only be used for extract"
            ):
                A(input_mask=M.S) << A.apply(unary.ainv)
                A.compute()

            A = A_.dup()
            with pytest.raises(
                TypeError, match="`input_mask` argument may only be used for extract"
            ):
                A(input_mask=M.S)[[0], [0]] = 1
                A.compute()

            A = A_.dup()
            with pytest.raises(
                TypeError, match="`input_mask` argument may only be used for extract"
            ):
                A(input_mask=M.S)[[0], [0]]
                A.compute()

            A = A_.dup()
            m = M[0, :].new()
            # With transpose input value
            # Matrix structure mask
            result = A.T[[0, 1], 0].new(input_mask=MT.S)
            expected = Vector.from_values([1], [1])
            assert result.isequal(expected)
            # again
            result.clear()
            result(input_mask=MT.S) << A.T[[0, 1], 0]
            assert result.isequal(expected)

            # Vector mask
            result = A.T[[0, 1], 0].new(input_mask=m.S)
            assert result.isequal(expected)
            # again
            result.clear()
            result(input_mask=m.S) << A.T[[0, 1], 0]
            assert result.isequal(expected)

            # Matrix value mask
            result = A.T[[1, 2], 0].new(input_mask=MT.V)
            expected = Vector.from_values([1], [2], size=2)
            assert result.isequal(expected)
            # again
            result.clear()
            result(input_mask=MT.V) << A.T[[1, 2], 0]
            assert result.isequal(expected)

    # With transpose input value
    # Matrix structure mask
    As = [
        Matrix.from_values(
            [0, 0, 0, 1, 1, 1],
            [0, 1, 2, 0, 1, 2],
            [0, 1, 2, 3, 4, 5],
        )
    ]
    As += [
        Matrix.from_values(
            da.from_array([0, 0, 0, 1, 1, 1]),
            da.from_array([0, 1, 2, 0, 1, 2]),
            da.from_array([0, 1, 2, 3, 4, 5]),
        )
    ]
    Ms = [
        Matrix.from_values(
            [0, 0, 1, 1],
            [1, 2, 0, 1],
            [0, 1, 2, 3],
        )
    ]
    Ms += [
        Matrix.from_values(
            da.from_array([0, 0, 1, 1]),
            da.from_array([1, 2, 0, 1]),
            da.from_array([0, 1, 2, 3]),
        )
    ]
    for A_ in As:
        for M_ in Ms:
            A = A_.dup()
            M = M_.dup()
            A.rechunk(chunks=((1, 1), (2, 1)), inplace=True)
            result = A.T[[0, 1], 0].new(input_mask=MT.S)
            expected = Vector.from_values([1], [1])
            assert result.isequal(expected)
            # again
            result.clear()
            result(input_mask=MT.S) << A.T[[0, 1], 0]
            assert result.isequal(expected)

            # Vector mask
            result = A.T[[0, 1], 0].new(input_mask=m.S)
            assert result.isequal(expected)
            # again
            result.clear()
            result(input_mask=m.S) << A.T[[0, 1], 0]
            assert result.isequal(expected)

            # Matrix value mask
            result = A.T[[1, 2], 0].new(input_mask=MT.V)
            expected = Vector.from_values([1], [2], size=2)
            assert result.isequal(expected)
            # again
            result.clear()
            result(input_mask=MT.V) << A.T[[1, 2], 0]
            assert result.isequal(expected)


def test_extract_with_matrix(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            with pytest.raises(TypeError, match="Invalid type for index"):
                A[A.T, 1].new()
            with pytest.raises(TypeError, match="Invalid type for index"):
                A[A, [1]].new()
            with pytest.raises(TypeError, match="Invalid type for index"):
                A[[0], A.V].new()


def test_assign(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            B = Matrix.from_values([0, 0, 1], [0, 1, 0], [9, 8, 7])
            result = Matrix.from_values(
                [0, 0, 2, 3, 0, 3, 5, 6, 0, 6, 1, 6, 4, 1],
                [0, 5, 0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6],
                [9, 8, 7, 3, 2, 3, 1, 5, 3, 7, 8, 3, 7, 4],
            )
            C = A.dup()
            C()[[0, 2], [0, 5]] = B
            assert C.isequal(result)
            C = A.dup()
            C[:3:2, :6:5]() << B
            assert C.isequal(result)
            with pytest.raises(TypeError, match="will make the Matrix dense"):
                C << 1
            nvals = C.nvals
            C(C.S) << 1
            assert C.nvals == nvals
            assert C.reduce_scalar().new() == nvals
            with pytest.raises(TypeError, match="Invalid type for index"):
                C[C, [1]] = C
            B = B.T.new()
            C = A.dup()
            C()[[0, 2], [0, 5]] = B.T
            assert C.isequal(result)
            C = A.dup()
            C[:3:2, :6:5]() << B.T
            assert C.isequal(result)

            B.rechunk(chunks=1)
            C = A.dup()
            C()[[0, 2], [0, 5]] = B.T
            assert C.isequal(result)
            C = A.dup()
            C[:3:2, :6:5]() << B.T
            assert C.isequal(result)


def test_assign_wrong_dims(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            B = Matrix.from_values([0, 0, 1], [0, 1, 0], [9, 8, 7])
            with pytest.raises(DimensionMismatch):
                A[[0, 2, 4], [0, 5]] = B
                A.compute()


def test_assign_row(As, A_chunks, v):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            result = Matrix.from_values(
                [3, 3, 5, 6, 6, 1, 6, 2, 4, 1, 0, 0, 0, 0],
                [0, 2, 2, 2, 3, 4, 4, 5, 5, 6, 1, 3, 4, 6],
                [3, 3, 1, 5, 7, 8, 3, 1, 7, 4, 1, 1, 2, 0],
            )
            C = A.dup()
            C[0, :] = v
            assert C.isequal(result)


def test_subassign_row_col(A_chunks):
    A_0 = Matrix.from_values(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
    )
    A_1 = Matrix.from_values(
        da.from_array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        da.from_array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
        da.from_array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    )
    As = [A_0, A_1]
    for A_ in As:
        for chunks in [3, 2, 1]:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            m = Vector.from_values([1], [True])
            v = Vector.from_values([0, 1], [10, 20])

            A[[0, 1], 0](m.S) << v
            result1 = Matrix.from_values(
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
                [0, 1, 2, 0, 1, 2, 0, 1, 2],
                [0, 1, 2, 20, 4, 5, 6, 7, 8],
            )
            assert A.isequal(result1)

            A[1, [1, 2]](m.V, accum=binary.plus).update(v)
            result2 = Matrix.from_values(
                [0, 0, 0, 1, 1, 1, 2, 2, 2],
                [0, 1, 2, 0, 1, 2, 0, 1, 2],
                [0, 1, 2, 20, 4, 25, 6, 7, 8],
            )
            assert A.isequal(result2)

            A[[0, 1], 0](m.S, binary.plus, replace=True) << v
            result3 = Matrix.from_values(
                [0, 0, 1, 1, 1, 2, 2, 2],
                [1, 2, 0, 1, 2, 0, 1, 2],
                [1, 2, 40, 4, 25, 6, 7, 8],
            )
            assert A.isequal(result3)

            _A = A.dup()
            with pytest.raises(DimensionMismatch):
                A(m.S)[[0, 1], 0] << v
                A.compute()

            A = _A
            A[[0, 1], 0](m.S) << 99
            result4 = Matrix.from_values(
                [0, 0, 1, 1, 1, 2, 2, 2],
                [1, 2, 0, 1, 2, 0, 1, 2],
                [1, 2, 99, 4, 25, 6, 7, 8],
            )
            assert A.isequal(result4)

            A[[1, 2], 0](m.S, binary.plus, replace=True) << 100
            result5 = Matrix.from_values(
                [0, 0, 1, 1, 2, 2, 2],
                [1, 2, 1, 2, 0, 1, 2],
                [1, 2, 4, 25, 106, 7, 8],
            )
            assert A.isequal(result5)

            A[2, [0, 1]](m.S) << -1
            result6 = Matrix.from_values(
                [0, 0, 1, 1, 2, 2, 2],
                [1, 2, 1, 2, 0, 1, 2],
                [1, 2, 4, 25, 106, -1, 8],
            )
            assert A.isequal(result6)


def test_subassign_matrix():
    A_0 = Matrix.from_values(
        [0, 0, 0, 1, 1, 1, 2, 2, 2],
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
    )
    A_1 = Matrix.from_values(
        da.from_array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        da.from_array([0, 1, 2, 0, 1, 2, 0, 1, 2]),
        da.from_array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    )
    As = [A_0, A_1]
    for A_i in As:
        A = A_i.dup()
        m = Matrix.from_values([1], [0], [True])
        v = Matrix.from_values([0, 1], [0, 0], [10, 20])
        mT = m.T.new()

        A[[0, 1], [0]](m.S) << v
        result1 = Matrix.from_values(
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 1, 2, 20, 4, 5, 6, 7, 8],
        )
        assert A.isequal(result1)

        A_ = A.dup()
        _A = A.dup()
        _A_ = A.dup()

        A[[1], [1, 2]](mT.V, accum=binary.plus) << v.T
        result2 = Matrix.from_values(
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [0, 1, 2, 20, 4, 25, 6, 7, 8],
        )
        assert A.isequal(result2)

        A_[[1], 1:3](mT.V, accum=binary.plus) << v.T
        assert A_.isequal(result2)

        _A[1:2, [1, 2]](mT.V, accum=binary.plus) << v.T
        assert _A.isequal(result2)

        _A_[1:2, 1:3](mT.V, accum=binary.plus) << v.T
        assert _A_.isequal(result2)

        A[[0, 1], [0]](m.S, binary.plus, replace=True) << v
        result3 = Matrix.from_values(
            [0, 0, 1, 1, 1, 2, 2, 2],
            [1, 2, 0, 1, 2, 0, 1, 2],
            [1, 2, 40, 4, 25, 6, 7, 8],
        )
        assert A.isequal(result3)

        A__ = A.dup()
        with pytest.raises(DimensionMismatch):
            A(m.S)[[0, 1], [0]] << v
            A.compute()

        A = A__
        A[[0, 1], [0]](m.S) << 99
        result4 = Matrix.from_values(
            [0, 0, 1, 1, 1, 2, 2, 2],
            [1, 2, 0, 1, 2, 0, 1, 2],
            [1, 2, 99, 4, 25, 6, 7, 8],
        )
        assert A.isequal(result4)

        A[[1, 2], [0]](m.S, binary.plus, replace=True) << 100
        result5 = Matrix.from_values(
            [0, 0, 1, 1, 2, 2, 2],
            [1, 2, 1, 2, 0, 1, 2],
            [1, 2, 4, 25, 106, 7, 8],
        )
        assert A.isequal(result5)

        A[[2], [0, 1]](mT.S) << -1
        result6 = Matrix.from_values(
            [0, 0, 1, 1, 2, 2, 2],
            [1, 2, 1, 2, 0, 1, 2],
            [1, 2, 4, 25, 106, -1, 8],
        )
        assert A.isequal(result6)


def test_assign_column(As, A_chunks, v):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            result = Matrix.from_values(
                [3, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1, 1, 3, 4, 6],
                [0, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 1, 1, 1, 1],
                [3, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4, 1, 1, 2, 0],
            )
            C = A.dup()
            C[:, 1] = v
            assert C.isequal(result)


def test_assign_row_scalar(As, A_chunks, v):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            C = A.dup()
            C[0, :](v.S) << v
            D = A.dup()
            D(v.S)[0, :] << v
            assert C.isequal(D)

            C[:, :](C.S) << 1

            C_ = C.dup()
            with pytest.raises(
                TypeError, match="Unable to use Vector mask on Matrix assignment to a Matrix"
            ):
                C[:, :](v.S) << 1
                C.compute()

            C = C_.dup()
            with pytest.raises(
                TypeError,
                match="Unable to use Vector mask on single element assignment to a Matrix",
            ):
                C[0, 0](v.S) << 1
                C.compute()

            C = C_.dup()
            with pytest.raises(TypeError):
                C[0, 0](v.S) << v
                C.compute()

            C = C_.dup()
            with pytest.raises(TypeError):
                C(v.S)[0, 0] << v
                C.compute()

            C = C_.dup()
            with pytest.raises(TypeError):
                C[0, 0](C.S) << v
                C.compute()

            C = C_.dup()
            with pytest.raises(TypeError):
                C(C.S)[0, 0] << v
                C.compute()

            C = C_.dup()
            with pytest.raises(TypeError):
                C[0, 0](v.S) << C
                C.compute()

            C = C_.dup()
            with pytest.raises(TypeError):
                C[0, 0](C.S) << C
                C.compute()

            C = A.dup()
            C(v.S)[0, :] = 10
            result = Matrix.from_values(
                [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1, 0, 0],
                [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 4, 6],
                [3, 10, 3, 1, 5, 10, 7, 8, 3, 1, 7, 4, 10, 10],
            )
            assert C.isequal(result)


def test_assign_row_col_matrix_mask():
    # A         B       v1      v2
    # 0 1       4 _     100     10
    # 2 _       0 5             20
    A_0 = Matrix.from_values([0, 0, 1], [0, 1, 0], [0, 1, 2])
    B_0 = Matrix.from_values([0, 1, 1], [0, 0, 1], [4, 0, 5])
    v1_0 = Vector.from_values([0], [100])
    v2_0 = Vector.from_values([0, 1], [10, 20])

    A_1 = Matrix.from_values(
        da.from_array([0, 0, 1]), da.from_array([0, 1, 0]), da.from_array([0, 1, 2])
    )
    B_1 = Matrix.from_values(
        da.from_array([0, 1, 1]), da.from_array([0, 0, 1]), da.from_array([4, 0, 5])
    )
    v1_1 = Vector.from_values(da.from_array([0]), da.from_array([100]))
    v2_1 = Vector.from_values(da.from_array([0, 1]), da.from_array([10, 20]))

    As = [A_0, A_1]
    Bs = [B_0, B_1]
    v1s = [v1_0, v1_1]
    v2s = [v2_0, v2_1]

    for A in As:
        for B in Bs:
            for v1 in v1s:
                for v2 in v2s:
                    # row assign
                    C = A.dup()
                    C(B.S)[0, :] << v2
                    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [10, 1, 2])
                    assert C.isequal(result)

                    C = A.dup()
                    C(B.S, accum=binary.plus)[1, :] = v2
                    result = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 12, 20])
                    assert C.isequal(result)

                    C = A.dup()
                    C(B.S, replace=True)[1, :] << v2
                    result = Matrix.from_values([0, 1, 1], [0, 0, 1], [0, 10, 20])
                    assert C.isequal(result)

                    # col assign
                    C = A.dup()
                    C(B.S)[:, 0] = v2
                    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [10, 1, 20])
                    assert C.isequal(result)

                    C = A.dup()
                    C(B.S, accum=binary.plus)[:, 1] << v2
                    result = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 2, 20])
                    assert C.isequal(result)

                    C = A.dup()
                    C(B.S, replace=True)[:, 1] = v2
                    result = Matrix.from_values([0, 1, 1], [0, 0, 1], [0, 2, 20])
                    assert C.isequal(result)

                    # row assign scalar (as a sanity check)
                    C = A.dup()
                    C(B.S)[0, :] = 100
                    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [100, 1, 2])
                    assert C.isequal(result)

                    C = A.dup()
                    C(B.S, accum=binary.plus)[1, :] << 100
                    result = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 102, 100])
                    assert C.isequal(result)

                    C = A.dup()
                    C(B.S, replace=True)[1, :] = 100
                    result = Matrix.from_values([0, 1, 1], [0, 0, 1], [0, 100, 100])
                    assert C.isequal(result)

                    # col assign scalar (as a sanity check)
                    C = A.dup()
                    C(B.S)[:, 0] << 100
                    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [100, 1, 100])
                    assert C.isequal(result)

                    C = A.dup()
                    C(B.S, accum=binary.plus)[:, 1] = 100
                    result = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 2, 100])
                    assert C.isequal(result)

                    C = A.dup()
                    C(B.S, replace=True)[:, 1] << 100
                    result = Matrix.from_values([0, 1, 1], [0, 0, 1], [0, 2, 100])
                    assert C.isequal(result)

                    # row subassign
                    C = A.dup()
                    C[0, :](v2.S) << v2
                    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [10, 20, 2])
                    assert C.isequal(result)

                    C = A.dup()
                    C[0, [0]](v1.S) << v1
                    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [100, 1, 2])
                    assert C.isequal(result)

                    with pytest.raises(
                        TypeError,
                        match="Indices for subassign imply Vector submask, but got Matrix mask instead",
                    ):
                        C[0, :](B.S) << v2
                        C.compute()

                    # col subassign
                    C = A.dup()
                    C[:, 0](v2.S) << v2
                    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [10, 1, 20])
                    assert C.isequal(result)

                    C = A.dup()
                    C[[0], 0](v1.S) << v1
                    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [100, 1, 2])
                    assert C.isequal(result)

                    with pytest.raises(
                        TypeError,
                        match="Indices for subassign imply Vector submask, but got Matrix mask instead",
                    ):
                        C[:, 0](B.S) << v2
                        C.compute()

                    # row subassign scalar
                    C = A.dup()
                    C[0, :](v2.S) << 100
                    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [100, 100, 2])
                    assert C.isequal(result)

                    C = A.dup()
                    C[0, [0]](v1.S) << 100
                    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [100, 1, 2])
                    assert C.isequal(result)

                    with pytest.raises(
                        TypeError,
                        match="Indices for subassign imply Vector submask, but got Matrix mask instead",
                    ):
                        C[:, 0](B.S) << 100
                        C.compute()

                    # col subassign scalar
                    C = A.dup()
                    C[:, 0](v2.S) << 100
                    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [100, 1, 100])
                    assert C.isequal(result)

                    C = A.dup()
                    C[[0], 0](v1.S) << 100
                    result = Matrix.from_values([0, 0, 1], [0, 1, 0], [100, 1, 2])
                    assert C.isequal(result)

                    with pytest.raises(
                        TypeError,
                        match="Indices for subassign imply Vector submask, but got Matrix mask instead",
                    ):
                        C[:, 0](B.S) << 100
                        C.compute()

                    # Bad subassign
                    C = A.dup()
                    with pytest.raises(
                        TypeError, match="Single element assign does not accept a submask"
                    ):
                        C[0, 0](B.S) << 100
                        C.compute()


def test_assign_column_scalar(As, A_chunks, v):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            C = A.dup()
            C[:, 0](v.S) << v
            D = A.dup()
            D(v.S)[:, 0] << v
            assert C.isequal(D)

            C = A.dup()
            C[:, 1] = v
            C(v.S)[:, 1] = 10
            result = Matrix.from_values(
                [3, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1, 1, 3, 4, 6],
                [0, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 1, 1, 1, 1],
                [3, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4, 10, 10, 10, 10],
            )
            assert C.isequal(result)

            C(v.V, replace=True, accum=binary.plus)[:, 1] = 20
            result = Matrix.from_values(
                [3, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1, 1, 3, 4],
                [0, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 1, 1, 1],
                [3, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4, 30, 30, 30],
            )
            assert C.isequal(result)


def test_assign_scalar(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            # Test block
            result_block = Matrix.from_values(
                [3, 0, 6, 0, 6, 6, 2, 4, 1, 1, 3, 5, 1, 3, 5],
                [0, 1, 2, 3, 3, 4, 5, 5, 6, 2, 2, 2, 4, 4, 4],
                [3, 2, 5, 3, 7, 3, 1, 7, 4, 0, 0, 0, 0, 0, 0],
            )
            C = A.dup()
            C[[1, 3, 5], [2, 4]] = 0
            assert C.isequal(result_block)
            C = A.dup()
            C[[1, 3, 5], [2, 4]] = Scalar.from_value(0)
            assert C.isequal(result_block)
            C = A.dup()
            C[1::2, 2:5:2] = 0
            assert C.isequal(result_block)
            C = A.dup()
            C[1::2, 2:5:2] = Scalar.from_value(0)
            assert C.isequal(result_block)
            # Test row
            result_row = Matrix.from_values(
                [3, 0, 6, 0, 6, 6, 2, 4, 1, 3, 5, 1, 1],
                [0, 1, 2, 3, 3, 4, 5, 5, 6, 2, 2, 2, 4],
                [3, 2, 5, 3, 7, 3, 1, 7, 4, 3, 1, 0, 0],
            )
            C = A.dup()
            C[1, [2, 4]] = 0
            assert C.isequal(result_row)
            C = A.dup()
            C[1, 2] = Scalar.from_value(0)
            C[1, 4] = Scalar.from_value(0)
            assert C.isequal(result_row)
            C = A.dup()
            C[1, 2:5:2] = 0
            assert C.isequal(result_row)
            # Test column
            result_column = Matrix.from_values(
                [3, 0, 6, 0, 6, 6, 2, 4, 1, 1, 1, 3, 5],
                [0, 1, 2, 3, 3, 4, 5, 5, 6, 4, 2, 2, 2],
                [3, 2, 5, 3, 7, 3, 1, 7, 4, 8, 0, 0, 0],
            )
            C = A.dup()
            C[[1, 3, 5], 2] = 0
            assert C.isequal(result_column)
            C = A.dup()
            C[1::2, 2] = 0
            assert C.isequal(result_column)


def test_assign_bad(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            with pytest.raises(TypeError, match="Bad type"):
                A[0, 0] = object()
                A.compute()
            A = A_.dup()
            with pytest.raises(TypeError, match="Bad type"):
                A[:, 0] = object()
                A.compute()
            A = A_.dup()
            with pytest.raises(TypeError, match="Bad type"):
                A[0, :] = object()
                A.compute()
            A = A_.dup()
            with pytest.raises(TypeError, match="Bad type"):
                A[:, :] = object()
                A.compute()
            A = A_.dup()
            with pytest.raises(TypeError, match="Bad type"):
                A[0, 0] = A
                A.compute()
            A = A_.dup()
            with pytest.raises(TypeError, match="Bad type"):
                A[:, 0] = A
                A.compute()
            A = A_.dup()
            with pytest.raises(TypeError, match="Bad type"):
                A[0, :] = A
                A.compute()
            A = A_.dup()
            v = A[0, :].new()
            with pytest.raises(TypeError, match="Bad type"):
                A[0, 0] = v
                A.compute()
            A = A_.dup()
            with pytest.raises(TypeError, match="Bad type"):
                A[:, :] = v
                A.compute()


def test_apply(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            result = Matrix.from_values(
                [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
                [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                [-3, -2, -3, -1, -5, -3, -7, -8, -3, -1, -7, -4],
            )
            C = A.apply(unary.ainv).new()
            assert C.isequal(result)


def test_apply_binary(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            result_right = Matrix.from_values(
                [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
                [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                dtype=bool,
            )
            w_right = A.apply(binary.gt, right=1).new()
            w_right2 = A.apply(binary.gt, right=Scalar.from_value(1)).new()
            assert w_right.isequal(result_right)
            assert w_right2.isequal(result_right)
            result_left = Matrix.from_values(
                [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
                [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                [5, 6, 5, 7, 3, 5, 1, 0, 5, 7, 1, 4],
            )
            w_left = A.apply(binary.minus, left=8).new()
            w_left2 = A.apply(binary.minus, left=Scalar.from_value(8)).new()
            assert w_left.isequal(result_left)
            assert w_left2.isequal(result_left)
            with pytest.raises(TypeError):
                A.apply(binary.plus, left=A)
            with pytest.raises(TypeError):
                A.apply(binary.plus, right=A)
            with pytest.raises(TypeError, match="Cannot provide both"):
                A.apply(binary.plus, left=1, right=1)

            # allow monoids
            w1 = A.apply(binary.plus, left=1).new()
            w2 = A.apply(monoid.plus, left=1).new()
            w3 = A.apply(monoid.plus, right=1).new()
            assert w1.isequal(w2)
            assert w1.isequal(w3)


def test_reduce_row(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            result = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [5, 12, 1, 6, 7, 1, 15])
            w = A.reduce_rowwise(monoid.plus).new()
            assert w.isequal(result)
            w2 = A.reduce_rowwise(binary.plus).new()
            assert w2.isequal(result)


@pytest.mark.slow
def test_reduce_agg(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            result = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [5, 12, 1, 6, 7, 1, 15])
            w1 = A.reduce_rowwise(agg.sum).new()
            assert w1.isequal(result)
            w2 = A.T.reduce_columnwise(agg.sum).new()
            assert w2.isequal(result)

            counts = A.dup(dtype=bool).reduce_rowwise(monoid.plus[int]).new()
            w3 = A.reduce_rowwise(agg.count).new()
            assert w3.isequal(counts)
            w4 = A.T.reduce_columnwise(agg.count).new()
            assert w4.isequal(counts)

            Asquared = monoid.times(A & A).new()
            squared = Asquared.reduce_rowwise(monoid.plus).new()
            expected = unary.sqrt[float](squared).new()
            w5 = A.reduce_rowwise(agg.hypot).new()
            assert w5.isclose(expected)
            w6 = A.reduce_rowwise(monoid.numpy.hypot[float]).new()
            assert w6.isclose(expected)
            w7 = Vector.new(w5.dtype, size=w5.size)
            w7 << A.reduce_rowwise(agg.hypot)
            assert w7.isclose(expected)

            w8 = A.reduce_rowwise(agg.logaddexp).new()
            expected = A.reduce_rowwise(monoid.numpy.logaddexp[float]).new()
            assert w8.isclose(w8)

            result = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 2, 9, 10, 11, 8, 4])
            w9 = A.reduce_columnwise(agg.sum).new()
            assert w9.isequal(result)
            w10 = A.T.reduce_rowwise(agg.sum).new()
            assert w10.isequal(result)

            counts = A.dup(dtype=bool).reduce_columnwise(monoid.plus[int]).new()
            w11 = A.reduce_columnwise(agg.count).new()
            assert w11.isequal(counts)
            w12 = A.T.reduce_rowwise(agg.count).new()
            assert w12.isequal(counts)

            w13 = A.reduce_rowwise(agg.mean).new()
            expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [2.5, 6, 1, 3, 7, 1, 5])
            assert w13.isequal(expected)
            w14 = A.reduce_columnwise(agg.mean).new()
            expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 2, 3, 5, 5.5, 4, 4])
            assert w14.isequal(expected)

            w15 = A.reduce_rowwise(agg.exists).new()
            w16 = A.reduce_columnwise(agg.exists).new()
            expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 1, 1, 1])
            assert w15.isequal(expected)
            assert w16.isequal(expected)

            assert A.reduce_scalar(agg.sum).new() == 47
            assert A.reduce_scalar(agg.prod).new() == 1270080
            assert A.reduce_scalar(agg.count).new() == 12
            assert A.reduce_scalar(agg.count_nonzero).new() == 12
            assert A.reduce_scalar(agg.count_zero).new() == 0
            assert A.reduce_scalar(agg.sum_of_squares).new() == 245
            assert A.reduce_scalar(agg.hypot).new().isclose(245 ** 0.5)
            assert A.reduce_scalar(agg.logaddexp).new().isclose(8.6071076)
            assert A.reduce_scalar(agg.logaddexp2).new().isclose(9.2288187)
            assert A.reduce_scalar(agg.mean).new().isclose(47 / 12)
            assert A.reduce_scalar(agg.exists).new() == 1

            silly = agg.Aggregator(
                "silly",
                composite=[agg.varp, agg.stdp],
                finalize=lambda x, y: binary.times(x & y),
                types=[agg.varp],
            )
            v1 = A.reduce_rowwise(agg.varp).new()
            v2 = A.reduce_rowwise(agg.stdp).new()
            assert v1.isclose(binary.times(v2 & v2).new())
            v3 = A.reduce_rowwise(silly).new()
            assert v3.isclose(binary.times(v1 & v2).new())

            s1 = A.reduce_scalar(agg.varp).new()
            s2 = A.reduce_scalar(agg.stdp).new()
            assert s1.isclose(s2.value.compute() * s2.value.compute())
            s3 = A.reduce_scalar(silly).new()
            assert s3.isclose(s1.value.compute() * s2.value.compute())


def test_reduce_agg_argminmax(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            # reduce_rowwise
            expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [1, 6, 5, 0, 5, 2, 4])
            w1b = A.reduce_rowwise(agg.argmin).new()
            assert w1b.isequal(expected)
            w1c = A.T.reduce_columnwise(agg.argmin).new()
            assert w1c.isequal(expected)
            expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 4, 5, 0, 5, 2, 3])
            w2b = A.reduce_rowwise(agg.argmax).new()
            assert w2b.isequal(expected)
            w2c = A.T.reduce_columnwise(agg.argmax).new()
            assert w2c.isequal(expected)

            # reduce_cols
            expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 0, 5, 0, 6, 2, 1])
            w7b = A.reduce_columnwise(agg.argmin).new()
            assert w7b.isequal(expected)
            w7c = A.T.reduce_rowwise(agg.argmin).new()
            assert w7c.isequal(expected)
            expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 0, 6, 6, 1, 4, 1])
            w8b = A.reduce_columnwise(agg.argmax).new()
            assert w8b.isequal(expected)
            w8c = A.T.reduce_rowwise(agg.argmax).new()
            assert w8c.isequal(expected)

            # reduce_scalar
            with pytest.raises(
                ValueError, match="Aggregator argmin may not be used with Matrix.reduce_scalar"
            ):
                A.reduce_scalar(agg.argmin)

            silly = agg.Aggregator(
                "silly",
                composite=[agg.argmin, agg.argmax],
                finalize=lambda x, y: binary.plus(x & y),
                types=[agg.argmin],
            )
            v1 = A.reduce_rowwise(agg.argmin).new()
            v2 = A.reduce_rowwise(agg.argmax).new()
            v3 = A.reduce_rowwise(silly).new()
            assert v3.isequal(binary.plus(v1 & v2).new())

            v1 = A.reduce_columnwise(agg.argmin).new()
            v2 = A.reduce_columnwise(agg.argmax).new()
            v3 = A.reduce_columnwise(silly).new()
            assert v3.isequal(binary.plus(v1 & v2).new())

            with pytest.raises(ValueError, match="Aggregator"):
                A.reduce_scalar(silly).new()


@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_reduce_agg_firstlast(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        # reduce_rowwise
        w1 = A.reduce_rowwise(agg.first).new()
        expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [2, 8, 1, 3, 7, 1, 5])
        assert w1.isequal(expected)
        w1b = A.T.reduce_columnwise(agg.first).new()
        assert w1b.isequal(expected)
        w2 = A.reduce_rowwise(agg.last).new()
        expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 4, 1, 3, 7, 1, 3])
        assert w2.isequal(expected)
        w2b = A.T.reduce_columnwise(agg.last).new()
        assert w2b.isequal(expected)

        # reduce_columnwise
        w3 = A.reduce_columnwise(agg.first).new()
        expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 2, 3, 3, 8, 1, 4])
        assert w3.isequal(expected)
        w3b = A.T.reduce_rowwise(agg.first).new()
        assert w3b.isequal(expected)
        w4 = A.reduce_columnwise(agg.last).new()
        expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 2, 5, 7, 3, 7, 4])
        assert w4.isequal(expected)
        w4b = A.T.reduce_rowwise(agg.last).new()
        assert w4b.isequal(expected)

        # reduce_scalar
        w5 = A.reduce_scalar(agg.first).new()
        assert w5 == 2
        w6 = A.reduce_scalar(agg.last).new()
        assert w6 == 3
        B = Matrix.new(float, nrows=2, ncols=3)
        assert B.reduce_scalar(agg.first).new().is_empty
        assert B.reduce_scalar(agg.last).new().is_empty
        w7 = B.reduce_rowwise(agg.first).new()
        assert w7.isequal(Vector.new(float, size=B.nrows))
        w8 = B.reduce_columnwise(agg.last).new()
        assert w8.isequal(Vector.new(float, size=B.ncols))

        silly = agg.Aggregator(
            "silly",
            composite=[agg.first, agg.last],
            finalize=lambda x, y: binary.plus(x & y),
            types=[agg.first],
        )
        v1 = A.reduce_rowwise(agg.first).new()
        v2 = A.reduce_rowwise(agg.last).new()
        v3 = A.reduce_rowwise(silly).new()
        assert v3.isequal(binary.plus(v1 & v2).new())

        s1 = A.reduce_scalar(agg.first).new()
        s2 = A.reduce_scalar(agg.last).new()
        s3 = A.reduce_scalar(silly).new()
        assert s3.isequal(s1.value.compute() + s2.value.compute())


@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_reduce_agg_firstlast_index(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        # reduce_rowwise
        w1 = A.reduce_rowwise(agg.first_index).new()
        expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [1, 4, 5, 0, 5, 2, 2])
        assert w1.isequal(expected)
        w1b = A.T.reduce_columnwise(agg.first_index).new()
        assert w1b.isequal(expected)
        w2 = A.reduce_rowwise(agg.last_index).new()
        expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 6, 5, 2, 5, 2, 4])
        assert w2.isequal(expected)
        w2b = A.T.reduce_columnwise(agg.last_index).new()
        assert w2b.isequal(expected)

        # reduce_columnwise
        w3 = A.reduce_columnwise(agg.first_index).new()
        expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 0, 3, 0, 1, 2, 1])
        assert w3.isequal(expected)
        w3b = A.T.reduce_rowwise(agg.first_index).new()
        assert w3b.isequal(expected)
        w4 = A.reduce_columnwise(agg.last_index).new()
        expected = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 0, 6, 6, 6, 4, 1])
        assert w4.isequal(expected)
        w4b = A.T.reduce_rowwise(agg.last_index).new()
        assert w4b.isequal(expected)

        # reduce_scalar
        with pytest.raises(ValueError, match="Aggregator first_index may not"):
            A.reduce_scalar(agg.first_index).new()
        with pytest.raises(ValueError, match="Aggregator last_index may not"):
            A.reduce_scalar(agg.last_index).new()

        silly = agg.Aggregator(
            "silly",
            composite=[agg.first_index, agg.last_index],
            finalize=lambda x, y: binary.plus(x & y),
            types=[agg.first_index],
        )
        v1 = A.reduce_rowwise(agg.first_index).new()
        v2 = A.reduce_rowwise(agg.last_index).new()
        v3 = A.reduce_rowwise(silly).new()
        assert v3.isequal(binary.plus(v1 & v2).new())

        with pytest.raises(ValueError, match="Aggregator"):
            A.reduce_scalar(silly).new()


@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_reduce_agg_empty(A_chunks):
    A = Matrix.new("UINT8", nrows=3, ncols=4)
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        for B in [A, A.T]:
            ve = Vector.new(bool, size=B.nrows)
            we = Vector.new(bool, size=B.ncols)
            for attr, aggr in vars(agg).items():
                if not isinstance(aggr, agg.Aggregator):
                    continue
                v = B.reduce_rowwise(aggr).new()
                assert ve.isequal(v)
                w = B.reduce_columnwise(aggr).new()
                assert we.isequal(w)
                if attr not in {"argmin", "argmax", "first_index", "last_index"}:
                    s = B.reduce_scalar(aggr).new()
                    assert compute(s.value) is None


def test_reduce_row_udf(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            result = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [5, 12, 1, 6, 7, 1, 15])
            binop = grblas.operator.BinaryOp.register_anonymous(lambda x, y: x + y)
            with pytest.raises(NotImplementedException):
                # Although allowed by the spec, SuiteSparse doesn't like user-defined binarops here
                A.reduce_rowwise(binop).new()
            # If the user creates a monoid from the binop, then we can use the monoid instead
            monoid = grblas.operator.Monoid.register_anonymous(binop, 0)
            w = A.reduce_rowwise(binop).new()
            assert w.isequal(result)
            w2 = A.reduce_rowwise(monoid).new()
            assert w2.isequal(result)


def test_reduce_column(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            result = Vector.from_values([0, 1, 2, 3, 4, 5, 6], [3, 2, 9, 10, 11, 8, 4])
            w = A.reduce_columnwise(monoid.plus).new()
            assert w.isequal(result)
            w2 = A.reduce_columnwise(binary.plus).new()
            assert w2.isequal(result)


def test_reduce_scalar(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            s = A.reduce_scalar(monoid.plus).new()
            assert s == 47
            assert A.reduce_scalar(binary.plus).new() == 47
            with pytest.raises(TypeError, match="Expected type: Monoid"):
                A.reduce_scalar(binary.minus)

            # test dtype coercion
            assert A.dtype == dtypes.INT64
            s = A.reduce_scalar().new(dtype=float)
            assert s == 47.0
            assert s.dtype == dtypes.FP64
            t = Scalar.new(float)
            t << A.reduce_scalar(monoid.plus)
            assert t == 47.0
            t = Scalar.new(float)
            t() << A.reduce_scalar(monoid.plus)
            assert t == 47.0
            t(accum=binary.times) << A.reduce_scalar(monoid.plus)
            assert t == 47 * 47
            assert A.reduce_scalar(monoid.plus[dtypes.UINT64]).new() == 47
            # Make sure we accumulate as a float, not int
            t.value = 1.23
            t(accum=binary.plus) << A.reduce_scalar()
            assert t == 48.23


def test_transpose(As, A_chunks):
    # C << A.T
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            rows, cols, vals = A.to_values()
            result = Matrix.from_values(cols, rows, vals)
            C = Matrix.new(A.dtype, A.ncols, A.nrows)
            C << A.T
            assert C.isequal(result)
            C2 = A.T.new()
            assert C2.isequal(result)
            assert A.T.T is A
            C3 = A.T.new(dtype=float)
            assert C3.isequal(result)


def test_kronecker():
    # A  0 1     B  0 1 2
    # 0 [1 -]    0 [- 2 3]
    # 1 [2 3]    1 [8 - 4]
    #
    # C  0  1  2  3  4  5
    # 0 [-  2  3  -  -  - ]
    # 1 [8  -  4  -  -  - ]
    # 2 [-  4  6  -  6  9 ]
    # 3 [16 -  8  24 -  12]
    A0 = Matrix.from_values([0, 1, 1], [0, 0, 1], [1, 2, 3])
    A1 = Matrix.from_values(
        da.from_array([0, 1, 1]),
        da.from_array([0, 0, 1]),
        da.from_array([1, 2, 3]),
    )
    As = [A0, A1]
    B0 = Matrix.from_values([0, 0, 1, 1], [1, 2, 0, 2], [2, 3, 8, 4])
    B1 = Matrix.from_values(
        da.from_array([0, 0, 1, 1]),
        da.from_array([1, 2, 0, 2]),
        da.from_array([2, 3, 8, 4]),
    )
    Bs = [B0, B1]
    result = Matrix.from_values(
        [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [1, 2, 0, 2, 1, 2, 4, 5, 0, 2, 3, 5],
        [2, 3, 8, 4, 4, 6, 6, 9, 16, 8, 24, 12],
    )
    for A in As:
        for B in Bs:
            C = A.kronecker(B, binary.times).new()
            assert C.isequal(result)


def test_simple_assignment(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            # C << A
            C = Matrix.new(A.dtype, A.nrows, A.ncols)
            C << A
            assert C.isequal(A)


def test_assign_transpose(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            C = Matrix.new(A.dtype, A.ncols, A.nrows)
            C << A.T
            assert C.isequal(A.T.new())

            with pytest.raises(TypeError):
                C.T << A
            with pytest.raises(TypeError, match="does not support item assignment"):
                C.T[:, :] << A
            with pytest.raises(TypeError, match="autocompute"):
                C[:, :].T << A

            nrows, ncols = A.nrows, A.ncols
            if A.is_dOnion:
                nrows, ncols = nrows.compute(), ncols.compute()
            C = Matrix.new(A.dtype, ncols + 1, nrows + 1)
            C[:ncols, :nrows] << A.T
            assert C[:ncols, :nrows].new().isequal(A.T.new())


def test_isequal(As, A_chunks, v):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            assert A.isequal(A)
            with pytest.raises(TypeError, match="Matrix"):
                A.isequal(v)  # equality is not type-checking
            C = Matrix.from_values([1], [1], [1])
            assert not C.isequal(A)
            D = Matrix.from_values([1], [2], [1])
            assert not C.isequal(D)
            D2 = Matrix.from_values([0], [2], [1], nrows=D.nrows, ncols=D.ncols)
            assert not D2.isequal(D)
            C2 = Matrix.from_values([1], [1], [1], nrows=7, ncols=7)
            assert not C2.isequal(A)
            C3 = Matrix.from_values(
                [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
                [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                [3.0, 2.0, 3.0, 1.0, 5.0, 3.0, 7.0, 8.0, 3.0, 1.0, 7.0, 4.0],
            )
            assert not C3.isequal(A, check_dtype=True), "different datatypes are not equal"
            C4 = Matrix.from_values(
                [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
                [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                [3.0, 2.0, 3.0, 1.0, 5.0, 3.000000000000000001, 7.0, 8.0, 3.0, 1 - 1e-11, 7.0, 4.0],
            )
            assert not C4.isequal(A)


def test_isclose(As, A_chunks, v):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            assert A.isclose(A)
            with pytest.raises(TypeError, match="Matrix"):
                A.isclose(v)  # equality is not type-checking
            C = Matrix.from_values([1], [1], [1])  # wrong size
            assert not C.isclose(A)
            D = Matrix.from_values([1], [2], [1])
            assert not C.isclose(D)
            D2 = Matrix.from_values([0], [2], [1], nrows=D.nrows, ncols=D.ncols)
            assert not D2.isclose(D)
            C2 = Matrix.from_values([1], [1], [1], nrows=7, ncols=7)  # missing values
            assert not C2.isclose(A)
            C3 = Matrix.from_values(
                [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1, 0],
                [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 2],
                [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4, 3],
            )  # extra values
            assert not C3.isclose(A)
            C4 = Matrix.from_values(
                [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
                [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                [3.0, 2.0, 3.0, 1.0, 5.0, 3.0, 7.0, 8.0, 3.0, 1.0, 7.0, 4.0],
            )
            assert not C4.isclose(A, check_dtype=True), "different datatypes are not equal"
            # fmt: off
            C5 = Matrix.from_values(
                [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
                [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                [3.0, 2.0, 3.0, 1.0, 5.0, 3.000000000000000001, 7.0, 8.0, 3.0, 1 - 1e-11, 7.0, 4.0],
            )
            # fmt: on
            assert C5.isclose(A)
            C6 = Matrix.from_values(
                [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
                [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                [3.0, 2.000001, 3.0, 1.0, 5.0, 3.0, 7.0, 7.9999999, 3.0, 1.0, 7.0, 4.0],
            )
            assert C6.isclose(A, rel_tol=1e-3)


def test_transpose_equals(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            data = [
                [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
                [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4],
            ]
            B = Matrix.from_values(*data)
            assert A.isequal(B.T)
            assert B.isequal(A.T)
            assert A.T.isequal(B)
            assert A.T.isequal(A.T)
            assert A.isclose(A)
            assert A.isclose(B.T)
            assert B.isclose(A.T)
            assert A.T.isclose(B)
            assert A.T.isclose(A.T)


def test_transpose_exceptional():
    A0 = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [True, True, False, True])
    B0 = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [1, 2, 3, 4])
    A1 = Matrix.from_values(
        da.from_array([0, 0, 1, 1]),
        da.from_array([0, 1, 0, 1]),
        da.from_array([True, True, False, True]),
    )
    B1 = Matrix.from_values(
        da.from_array([0, 0, 1, 1]),
        da.from_array([0, 1, 0, 1]),
        da.from_array([1, 2, 3, 4]),
    )
    As, Bs = [A0, A1], [B0, B1]
    for A in As:
        for B in Bs:
            with pytest.raises(TypeError, match="not callable"):
                B.T(mask=A.V) << B.ewise_mult(B, op=binary.plus)
            with pytest.raises(AttributeError):
                B(mask=A.T.V) << B.ewise_mult(B, op=binary.plus)
            with pytest.raises(AttributeError):
                B.T(mask=A.T.V) << B.ewise_mult(B, op=binary.plus)
            with pytest.raises(TypeError, match="does not support item assignment"):
                B.T[1, 0] << 10
            with pytest.raises(TypeError, match="not callable"):
                B.T[1, 0]() << 10
            with pytest.raises(TypeError, match="not callable"):
                B.T()[1, 0] << 10
            # with pytest.raises(AttributeError):
            # should use new instead--Now okay.
            assert B.T.dup().isequal(B.T.new())
            # Not exceptional, but while we're here...
            C = B.T.new(mask=A.V)
            D = B.T.new()
            D = D.dup(mask=A.V)
            assert C.isequal(D)
            assert C.isequal(Matrix.from_values([0, 0, 1], [0, 1, 1], [1, 3, 4]))


def test_nested_matrix_operations():
    """Make sure temporaries aren't garbage-collected too soon"""
    A = Matrix.new(int, 8, 8)
    A.ewise_mult(A.mxm(A.T).new()).new().reduce_scalar().new()
    A.ewise_mult(A.ewise_mult(A.ewise_mult(A.ewise_mult(A).new()).new()).new())


@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_bad_init():
    with pytest.raises(TypeError, match="CData"):
        Matrix(None, float, name="bad_matrix")


def test_equals(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            assert (A == A).new().reduce_scalar(monoid.land).new()


def test_bad_update(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            with pytest.raises(TypeError, match="Assignment value must be a valid expression"):
                A << None
                A.compute()


def test_incompatible_shapes(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            B = A[:-1, :-1].new()
            with pytest.raises(DimensionMismatch):
                A.mxm(B).new().compute()
            A = A_.dup()
            with pytest.raises(DimensionMismatch):
                A.ewise_add(B).new().compute()
            A = A_.dup()
            with pytest.raises(DimensionMismatch):
                A.ewise_mult(B).new().compute()


@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_del(capsys):
    # Exceptions in __del__ are printed to stderr
    import gc

    # shell_A does not have `gb_obj` attribute
    shell_A = Matrix.__new__(Matrix)
    del shell_A
    # A has `gb_obj` of NULL
    A = Matrix.from_values([0, 1], [0, 1], [0, 1])
    gb_obj = A.gb_obj
    A.gb_obj = grblas.ffi.NULL
    del A
    # let's clean up so we don't have a memory leak
    A2 = Matrix.__new__(Matrix)
    A2.gb_obj = gb_obj
    del A2
    gc.collect()
    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err


@pytest.mark.parametrize("do_iso", [False, True])
@pytest.mark.parametrize("methods", [("export", "import"), ("unpack", "pack")])
@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_import_export(A, A_chunks, do_iso, methods):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        if do_iso:
            A(A.S) << 1
        A1 = A.dup()
        out_method, in_method = methods
        if out_method == "export":
            d = getattr(A1.ss, out_method)("csr", give_ownership=True)
        else:
            d = getattr(A1.ss, out_method)("csr")
        assert d["is_iso"] is do_iso
        if do_iso:
            assert_array_equal(d["values"], [1])
        assert d["nrows"] == 7
        assert d["ncols"] == 7
        assert_array_equal(d["indptr"], [0, 2, 4, 5, 7, 8, 9, 12])
        assert_array_equal(d["col_indices"], [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4])
        if not do_iso:
            assert_array_equal(d["values"], [2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3])
        if in_method == "import":
            B1 = Matrix.ss.import_any(take_ownership=True, **d)
            assert B1.isequal(A)
            assert B1.ss.is_iso is do_iso
        else:
            A1.ss.pack_any(take_ownership=True, **d)
            assert A1.isequal(A)
            assert A1.ss.is_iso is do_iso

        A2 = A.dup()
        d = getattr(A2.ss, out_method)("csc")
        assert d["is_iso"] is do_iso
        if do_iso:
            assert_array_equal(d["values"], [1])
        assert d["nrows"] == 7
        assert d["ncols"] == 7
        assert_array_equal(d["indptr"], [0, 1, 2, 5, 7, 9, 11, 12])
        assert_array_equal(d["row_indices"], [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1])
        if not do_iso:
            assert_array_equal(d["values"], [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4])
        if in_method == "import":
            B2 = Matrix.ss.import_any(**d)
            assert B2.isequal(A)
            assert B2.ss.is_iso is do_iso
        else:
            A2.ss.pack_any(**d)
            assert A2.isequal(A)
            assert A2.ss.is_iso is do_iso

        A3 = A.dup()
        d = getattr(A3.ss, out_method)("hypercsr")
        assert d["is_iso"] is do_iso
        if do_iso:
            assert_array_equal(d["values"], [1])
        assert d["nrows"] == 7
        assert d["ncols"] == 7
        assert_array_equal(d["rows"], [0, 1, 2, 3, 4, 5, 6])
        assert_array_equal(d["indptr"], [0, 2, 4, 5, 7, 8, 9, 12])
        assert_array_equal(d["col_indices"], [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4])
        if not do_iso:
            assert_array_equal(d["values"], [2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3])
        if in_method == "import":
            B3 = Matrix.ss.import_any(**d)
            assert B3.isequal(A)
            assert B3.ss.is_iso is do_iso
        else:
            A3.ss.pack_any(**d)
            assert A3.isequal(A)
            assert A3.ss.is_iso is do_iso

        A4 = A.dup()
        d = getattr(A4.ss, out_method)("hypercsc")
        assert d["is_iso"] is do_iso
        if do_iso:
            assert_array_equal(d["values"], [1])
        assert d["nrows"] == 7
        assert d["ncols"] == 7
        assert_array_equal(d["cols"], [0, 1, 2, 3, 4, 5, 6])
        assert_array_equal(d["indptr"], [0, 1, 2, 5, 7, 9, 11, 12])
        assert_array_equal(d["row_indices"], [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1])
        if not do_iso:
            assert_array_equal(d["values"], [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4])
        if in_method == "import":
            B4 = Matrix.ss.import_any(**d)
            assert B4.isequal(A)
            assert B4.ss.is_iso is do_iso
        else:
            A4.ss.pack_any(**d)
            assert A4.isequal(A)
            assert A4.ss.is_iso is do_iso

        A5 = A.dup()
        d = getattr(A5.ss, out_method)("bitmapr")
        assert d["is_iso"] is do_iso
        if do_iso:
            assert_array_equal(d["values"], [1])
        assert "nrows" not in d
        assert "ncols" not in d
        if not do_iso:
            assert d["values"].shape == (7, 7)
        assert d["bitmap"].shape == (7, 7)
        assert d["nvals"] == 12
        bitmap = np.array(
            [
                [0, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 0, 1, 0],
                [1, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
            ]
        )
        assert_array_equal(d["bitmap"], bitmap)
        if not do_iso:
            assert_array_equal(
                d["values"].ravel("K")[d["bitmap"].ravel("K")], [2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3]
            )
        del d["nvals"]
        if in_method == "import":
            B5 = Matrix.ss.import_any(**d)
            assert B5.isequal(A)
            assert B5.ss.is_iso is do_iso
        else:
            A5.ss.pack_any(**d)
            assert A5.isequal(A)
            assert A5.ss.is_iso is do_iso
        d["bitmap"] = np.concatenate([d["bitmap"], d["bitmap"]], axis=0)
        B5b = Matrix.ss.import_any(**d)
        if in_method == "import":
            if not do_iso:
                assert B5b.isequal(A)
                assert B5b.ss.is_iso is do_iso
            else:
                # B5b == [A, A]
                B5b.nvals == 2 * A.nvals
                B5b.nrows == A.nrows
                B5b.ncols == 2 * A.ncols
        else:
            A5.ss.pack_any(**d)
            assert A5.isequal(A)
            assert A5.ss.is_iso is do_iso

        A6 = A.dup()
        d = getattr(A6.ss, out_method)("bitmapc")
        assert d["is_iso"] is do_iso
        if do_iso:
            assert_array_equal(d["values"], [1])
        assert_array_equal(d["bitmap"], bitmap)
        if not do_iso:
            assert_array_equal(
                d["values"].ravel("K")[d["bitmap"].ravel("K")], [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4]
            )
        del d["nvals"]
        if in_method == "import":
            B6 = Matrix.ss.import_any(nrows=7, **d)
            assert B6.isequal(A)
            assert B6.ss.is_iso is do_iso
        else:
            A6.ss.pack_any(**d)
            assert A6.isequal(A)
            assert A6.ss.is_iso is do_iso
        d["bitmap"] = np.concatenate([d["bitmap"], d["bitmap"]], axis=1)
        if in_method == "import":
            B6b = Matrix.ss.import_any(ncols=7, **d)
            assert B6b.isequal(A)
            assert B6b.ss.is_iso is do_iso
        else:
            A6.ss.pack_any(**d)
            assert A6.isequal(A)
            assert A6.ss.is_iso is do_iso

        A7 = A.dup()
        d = getattr(A7.ss, out_method)()
        assert d["is_iso"] is do_iso
        if do_iso:
            assert_array_equal(d["values"], [1])
        if in_method == "import":
            B7 = Matrix.ss.import_any(**d)
            assert B7.isequal(A)
            assert B7.ss.is_iso is do_iso
        else:
            A7.ss.pack_any(**d)
            assert A7.isequal(A)
            assert A7.ss.is_iso is do_iso

        A8 = A.dup()
        d = getattr(A8.ss, out_method)("bitmapr", raw=True)
        assert d["is_iso"] is do_iso
        if do_iso:
            assert_array_equal(d["values"], [1])
        del d["nrows"]
        del d["ncols"]
        if in_method == "import":
            with pytest.raises(ValueError, match="nrows and ncols must be provided"):
                Matrix.ss.import_any(**d)
        else:
            A8.ss.pack_any(**d)
            assert A8.isequal(A)
            assert A8.ss.is_iso is do_iso

        A9 = A.dup()
        d = getattr(A9.ss, out_method)("coo", sort=True)
        assert d["is_iso"] is do_iso
        if do_iso:
            assert_array_equal(d["values"], [1])
        assert d["nrows"] == 7
        assert d["ncols"] == 7
        assert d["rows"].shape == (12,)
        assert d["cols"].shape == (12,)
        assert d["sorted_cols"]
        assert_array_equal(d["rows"], [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6])
        assert_array_equal(d["cols"], [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4])

        if do_iso:
            assert d["values"].shape == (1,)
        else:
            assert d["values"].shape == (12,)
        if in_method == "import":
            B8 = Matrix.ss.import_any(**d)
            assert B8.isequal(A)
            assert B8.ss.is_iso is do_iso
            del d["rows"]
            del d["format"]
            with pytest.raises(ValueError, match="coo requires both"):
                Matrix.ss.import_any(**d)
        else:
            A9.ss.pack_any(**d)
            assert A9.isequal(A)
            assert A9.ss.is_iso is do_iso

        C = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [1, 2, 3, 4])
        if do_iso:
            C(C.S) << 1
        C1 = C.dup()
        d = getattr(C1.ss, out_method)("fullr")
        assert d["is_iso"] is do_iso
        if do_iso:
            assert_array_equal(d["values"], [1])
            assert "nrows" in d
            assert "ncols" in d
        else:
            assert "nrows" not in d
            assert "ncols" not in d
        assert d["values"].flags.c_contiguous
        if not do_iso:
            assert d["values"].shape == (2, 2)
            assert_array_equal(d["values"], [[1, 2], [3, 4]])
            if in_method == "import":
                D1 = Matrix.ss.import_any(ncols=2, **d)
                assert D1.isequal(C)
                assert D1.ss.is_iso is do_iso
            else:
                C1.ss.pack_any(**d)
                assert C1.isequal(C)
                assert C1.ss.is_iso is do_iso
        else:
            if in_method == "import":
                D1 = Matrix.ss.import_any(**d)
                assert D1.isequal(C)
                assert D1.ss.is_iso is do_iso
            else:
                C1.ss.pack_any(**d)
                assert C1.isequal(C)
                assert C1.ss.is_iso is do_iso

        C2 = C.dup()
        d = getattr(C2.ss, out_method)("fullc")
        assert d["is_iso"] is do_iso
        if do_iso:
            assert_array_equal(d["values"], [1])
        if not do_iso:
            assert_array_equal(d["values"], [[1, 2], [3, 4]])
        assert d["values"].flags.f_contiguous
        if in_method == "import":
            D2 = Matrix.ss.import_any(**d)
            assert D2.isequal(C)
            assert D2.ss.is_iso is do_iso
        else:
            C2.ss.pack_any(**d)
            assert C2.isequal(C)
            assert C2.ss.is_iso is do_iso

        # all elements must have values
        with pytest.raises(InvalidValue):
            getattr(A.dup().ss, out_method)("fullr")
        with pytest.raises(InvalidValue):
            getattr(A.dup().ss, out_method)("fullc")

        a = np.array([0, 1, 2])
        for bad_combos in [
            ["indptr", "bitmap"],
            ["indptr"],
            ["indptr", "row_indices", "col_indices"],
            ["indptr", "rows", "cols"],
            ["indptr", "col_indices", "rows", "cols"],
            ["indptr", "rows"],
            ["indptr", "cols"],
            ["indptr", "row_indices", "rows"],
            ["indptr", "col_indices", "cols"],
            ["bitmap", "col_indices"],
            ["bitmap", "row_indices"],
            ["bitmap", "rows"],
            ["bitmap", "cols"],
        ]:
            with pytest.raises(TypeError):
                Matrix.ss.import_any(nrows=3, ncols=3, values=a, **dict.fromkeys(bad_combos, a))
        with pytest.raises(ValueError, match="Invalid format"):
            A.ss.export("coobad")
        D = Matrix.ss.import_csc(**A.ss.export("csc"))
        info = D.ss.export("coo", sort=True)
        assert info["sorted_rows"]
        E = Matrix.ss.import_any(**info)
        assert E.isequal(A)

        info = D.ss.export("coor")
        info["sorted_rows"] = False
        with pytest.raises(ValueError, match="sorted_rows must be True"):
            Matrix.ss.import_coor(**info)
        info["sorted_rows"] = True
        info["sorted_cols"] = False
        del info["format"]
        E = Matrix.ss.import_any(**info)
        assert E.isequal(D)

        info = D.ss.export("cooc")
        info["sorted_cols"] = False
        with pytest.raises(ValueError, match="sorted_cols must be True"):
            Matrix.ss.import_cooc(**info)
        info["sorted_cols"] = True
        info["sorted_rows"] = False
        del info["format"]
        E = Matrix.ss.import_any(**info)
        assert E.isequal(D)


@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_import_on_view():
    A = Matrix.from_values([0, 0, 1, 1], [0, 1, 0, 1], [1, 2, 3, 4])
    B = Matrix.ss.import_any(nrows=2, ncols=2, values=np.array([1, 2, 3, 4, 99, 99, 99])[:4])
    assert A.isequal(B)


@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_import_export_empty():
    A = Matrix.new(int, 2, 3)
    A1 = A.dup()
    d = A1.ss.export("csr")
    assert d["nrows"] == 2
    assert d["ncols"] == 3
    assert_array_equal(d["indptr"], [0, 0, 0])
    assert len(d["col_indices"]) == 0
    assert len(d["values"]) == 0
    B1 = Matrix.ss.import_any(**d)
    assert B1.isequal(A)

    A2 = A.dup()
    d = A2.ss.export("csc")
    assert d["nrows"] == 2
    assert d["ncols"] == 3
    assert_array_equal(d["indptr"], [0, 0, 0, 0])
    assert len(d["row_indices"]) == 0
    assert len(d["values"]) == 0
    B2 = Matrix.ss.import_any(**d)
    assert B2.isequal(A)

    A3 = A.dup()
    d = A3.ss.export("hypercsr")
    assert d["nrows"] == 2
    assert d["ncols"] == 3
    assert len(d["indptr"]) == 1
    assert d["indptr"][0] == 0
    assert len(d["col_indices"]) == 0
    assert len(d["values"]) == 0
    assert len(d["rows"]) == 0
    B3 = Matrix.ss.import_any(**d)
    assert B3.isequal(A)

    A4 = A.dup()
    d = A4.ss.export("hypercsc")
    assert d["nrows"] == 2
    assert d["ncols"] == 3
    assert len(d["indptr"]) == 1
    assert d["indptr"][0] == 0
    assert len(d["row_indices"]) == 0
    assert len(d["values"]) == 0
    assert len(d["cols"]) == 0
    B4 = Matrix.ss.import_any(**d)
    assert B4.isequal(A)

    A5 = A.dup()
    d = A5.ss.export("bitmapr")
    assert d["bitmap"].shape == (2, 3)
    assert d["bitmap"].flags.c_contiguous
    assert d["nvals"] == 0
    assert_array_equal(d["bitmap"].ravel(), 6 * [0])
    B5 = Matrix.ss.import_any(**d)
    assert B5.isequal(A)

    A6 = A.dup()
    d = A6.ss.export("bitmapc")
    assert d["bitmap"].shape == (2, 3)
    assert d["bitmap"].flags.f_contiguous
    assert d["nvals"] == 0
    assert_array_equal(d["bitmap"].ravel(), 6 * [0])
    B6 = Matrix.ss.import_any(**d)
    assert B6.isequal(A)

    # all elements must have values
    with pytest.raises(InvalidValue):
        A.dup().ss.export("fullr")
    with pytest.raises(InvalidValue):
        A.dup().ss.export("fullc")

    A7 = A.dup()
    d = A7.ss.export("coo")
    assert d["nrows"] == 2
    assert d["ncols"] == 3
    assert len(d["rows"]) == 0
    assert len(d["cols"]) == 0
    assert len(d["values"]) == 0

    # if we give the same value, make sure it's copied
    for format, key1, key2 in [
        ("csr", "values", "col_indices"),
        ("hypercsr", "values", "col_indices"),
        ("csc", "values", "row_indices"),
        ("hypercsc", "values", "row_indices"),
        ("bitmapr", "values", "bitmap"),
        ("bitmapc", "values", "bitmap"),
        ("coo", "values", "rows"),
    ]:
        # No assertions here, but code coverage should be "good enough"
        d = A.ss.export(format, raw=True)
        d[key1] = d[key2]
        Matrix.ss.import_any(take_ownership=True, **d)

    with pytest.raises(ValueError, match="Invalid format"):
        A.ss.export(format="bad_format")


@pytest.mark.parametrize("do_iso", [False, True])
@pytest.mark.parametrize("methods", [("export", "import"), ("unpack", "pack")])
@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_import_export_auto(A, A_chunks, do_iso, methods):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        if do_iso:
            A(A.S) << 1
        A_orig = A.dup()
        out_method, in_method = methods
        for format in [
            "csr",
            "csc",
            "hypercsr",
            "hypercsc",
            "bitmapr",
            "bitmapc",
            "coo",
            "coor",
            "cooc",
        ]:
            for (
                sort,
                raw,
                import_format,
                give_ownership,
                take_ownership,
                import_name,
            ) in itertools.product(
                [False, True],
                [False, True],
                [format, None],
                [False, True],
                [False, True],
                ["any", format],
            ):
                A2 = A.dup() if give_ownership or out_method == "unpack" else A
                if out_method == "export":
                    d = A2.ss.export(format, sort=sort, raw=raw, give_ownership=give_ownership)
                else:
                    d = A2.ss.unpack(format, sort=sort, raw=raw)
                if in_method == "import":
                    import_func = getattr(Matrix.ss, f"import_{import_name}")
                else:

                    def import_func(**kwargs):
                        getattr(A2.ss, f"pack_{import_name}")(**kwargs)
                        return A2

                d["format"] = import_format
                other = import_func(take_ownership=take_ownership, **d)
                if format == "bitmapc" and raw and import_format is None and import_name == "any":
                    # It's 1d, so we can't tell we're column-oriented w/o format keyword
                    assert other.isequal(A_orig.T)
                else:
                    assert other.isequal(A_orig)
                assert other.ss.is_iso is do_iso
                d["format"] = "bad_format"
                with pytest.raises(ValueError, match="Invalid format"):
                    import_func(**d)
        assert A.isequal(A_orig)
        assert A.ss.is_iso is do_iso
        assert A_orig.ss.is_iso is do_iso

        C = Matrix.from_values([0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1], [1, 2, 3, 4, 5, 6])
        if do_iso:
            C(C.S) << 1
        C_orig = C.dup()
        for format in ["fullr", "fullc"]:
            for (
                raw,
                import_format,
                give_ownership,
                take_ownership,
                import_name,
            ) in itertools.product(
                [False, True],
                [format, None],
                [False, True],
                [False, True],
                ["any", format],
            ):
                assert C.shape == (C.nrows, C.ncols)
                C2 = C.dup() if give_ownership or out_method == "unpack" else C
                if out_method == "export":
                    d = C2.ss.export(format, raw=raw, give_ownership=give_ownership)
                else:
                    d = C2.ss.unpack(format, raw=raw)
                if in_method == "import":
                    import_func = getattr(Matrix.ss, f"import_{import_name}")
                else:

                    def import_func(**kwargs):
                        getattr(C2.ss, f"pack_{import_name}")(**kwargs)
                        return C2

                d["format"] = import_format
                other = import_func(take_ownership=take_ownership, **d)
                if format == "fullc" and raw and import_format is None and import_name == "any":
                    # It's 1d, so we can't tell we're column-oriented w/o format keyword
                    if do_iso:
                        values = [1, 1, 1, 1, 1, 1]
                    else:
                        values = [1, 3, 5, 2, 4, 6]
                    assert other.isequal(
                        Matrix.from_values([0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1], values)
                    )
                else:
                    assert other.isequal(C_orig)
                assert other.ss.is_iso is do_iso
                d["format"] = "bad_format"
                with pytest.raises(ValueError, match="Invalid format"):
                    import_func(**d)
        assert C.isequal(C_orig)
        assert C.ss.is_iso is do_iso
        assert C_orig.ss.is_iso is do_iso


def test_no_bool_or_eq(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            with pytest.raises(TypeError, match="not defined"):
                bool(A)
            # with pytest.raises(TypeError, match="not defined"):
            A == A
            with pytest.raises(TypeError, match="not defined"):
                bool(A.S)
            with pytest.raises(TypeError, match="not defined"):
                A.S == A.S
            expr = A.ewise_mult(A)
            with pytest.raises(TypeError, match="not defined"):
                bool(expr)
            with pytest.raises(TypeError, match="not enabled"):
                expr == expr
            assigner = A[1, 2]()
            with pytest.raises(TypeError, match="not defined"):
                bool(assigner)
            with pytest.raises(TypeError, match="not defined"):
                assigner == assigner
            updater = A()
            with pytest.raises(TypeError, match="not defined"):
                bool(updater)
            with pytest.raises(TypeError, match="not defined"):
                updater == updater


@autocompute
def test_bool_eq_on_scalar_expressions(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            expr = A.reduce_scalar()
            assert expr == 47
            assert bool(expr)
            assert int(expr) == 47
            assert float(expr) == 47.0
            assert range(expr) == range(47)

            expr = A[0, 1]
            assert expr == 2
            assert bool(expr)
            assert int(expr) == 2
            assert float(expr) == 2.0
            assert range(expr) == range(2)

            expr = A[0, [1, 1]]
            # with pytest.raises(TypeError, match="not defined"):
            expr == expr  # Now okay
            with pytest.raises(TypeError, match="not defined"):
                bool(expr)
            with pytest.raises(TypeError):
                int(expr)
            with pytest.raises(TypeError):
                float(expr)
            with pytest.raises(TypeError):
                range(expr)


def test_bool_eq_on_scalar_expressions_no_auto(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            expr = A.reduce_scalar()
            with pytest.raises(TypeError, match="autocompute"):
                expr == 47
            with pytest.raises(TypeError, match="autocompute"):
                bool(expr)
            with pytest.raises(TypeError, match="autocompute"):
                int(expr)


def test_contains(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            assert (0, 1) in A
            assert (1, 0) in A.T

            assert (0, 1) not in A.T
            assert (1, 0) not in A

            with pytest.raises(TypeError):
                1 in A
            with pytest.raises(TypeError):
                (1,) in A.T
            with pytest.raises(TypeError, match="Invalid index"):
                (1, [1, 2]) in A


@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_iter(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        assert set(A) == set(
            zip(
                [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
                [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
            )
        )
        assert set(A.T) == set(
            zip(
                [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
                [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
            )
        )


def test_wait(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        A2 = A.dup()
        A2.wait()
        assert A2.isequal(A)


def test_pickle(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        s = pickle.dumps(A)
        A2 = pickle.loads(s)
        assert A.isequal(A2, check_dtype=True)
        assert A.name == A2.name


def test_weakref(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        d = weakref.WeakValueDictionary()
        d["A"] = A
        assert d["A"] is A
        AT = A.T
        d["A.T"] = AT
        assert d["A.T"] is AT
        expr = A.mxm(A)
        d["expr"] = expr
        assert d["expr"] is expr


def test_not_to_array(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        with pytest.raises(TypeError, match="Matrix can't be directly converted to a numpy array"):
            np.array(A)
        with pytest.raises(
            TypeError, match="TransposedMatrix can't be directly converted to a numpy array"
        ):
            np.array(A.T)


@pytest.mark.parametrize(
    "params",
    [
        (0, [], []),
        (1, [0, 4], [2, 7]),
        (3, [0, 1, 2], [3, 8, 1]),
        (10, [], []),
        (-1, [2], [3]),
        (-3, [0, 2, 3], [3, 1, 7]),
        (-10, [], []),
    ],
)
def test_diag(A, A_chunks, params):
    A_ = A
    for out_chunks in A_chunks:
        for in_chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=in_chunks, inplace=True)
            k, indices, values = params
            expected = Vector.from_values(
                indices, values, dtype=A.dtype, size=max(0, A.nrows - abs(k))
            )
            v = dask_grblas.ss.diag(A, k=k, chunks=out_chunks)
            assert expected.isequal(v)
            v[:] = 0
            v.ss.diag(A, k=k)
            assert expected.isequal(v)
            v = dask_grblas.ss.diag(A.T, k=-k, chunks=out_chunks)
            assert expected.isequal(v)
            v[:] = 0
            v.ss.diag(A.T, -k, chunks=out_chunks)
            assert expected.isequal(v)


def test_normalize_chunks():
    from grblas._ss.matrix import normalize_chunks

    shape = (20, 20)
    assert normalize_chunks(10, shape) == [[10, 10], [10, 10]]
    assert normalize_chunks(15.0, shape) == [[15, 5], [15, 5]]
    assert normalize_chunks(((10, 10), [10, 10]), shape) == [[10, 10], [10, 10]]
    assert normalize_chunks((15, [10, 10.0]), shape) == [[15, 5], [10, 10]]
    assert normalize_chunks((None, np.array([10, 10])), shape) == [[20], [10, 10]]
    assert normalize_chunks([[5, None], (None, 6)], shape) == [[5, 15], [14, 6]]
    assert normalize_chunks(np.array([10, 10]), shape) == [[10, 10], [10, 10]]

    with pytest.raises(TypeError, match="chunks argument must be a list"):
        normalize_chunks(None, shape)
    with pytest.raises(TypeError, match="None value in chunks"):
        normalize_chunks([[5, 5, None, None], 10], shape)
    with pytest.raises(TypeError, match="expected int or None, but got"):
        normalize_chunks([[15.5, 4.5], 10], shape)
    with pytest.raises(TypeError, match="Chunks for a dimension must be"):
        normalize_chunks([10, 10.5], shape)
    with pytest.raises(TypeError, match="must be integer dtype; got float64"):
        normalize_chunks([10, np.array([1.5, 2.5])], shape)
    with pytest.raises(TypeError, match="numpy array for chunks must be 1-dimension"):
        normalize_chunks([10, np.array([[1, 2], [3, 4]])], shape)

    with pytest.raises(ValueError, match="hunks argument must be of length 2"):
        normalize_chunks([10], shape)
    with pytest.raises(ValueError, match="Chunksize must be greater than 0"):
        normalize_chunks(-10, shape)
    with pytest.raises(ValueError, match="Chunksize must be greater than 0"):
        normalize_chunks([-10, -10], shape)
    with pytest.raises(ValueError, match="Chunksize must be greater than 0"):
        normalize_chunks([[-10, 30], [-10, 30]], shape)
    with pytest.raises(ValueError, match="chunks argument must be of length 2"):
        normalize_chunks([5, 5, 5], shape)
    with pytest.raises(ValueError, match="Chunks are too large"):
        normalize_chunks([[30, None], 10], shape)
    with pytest.raises(ValueError, match="Chunksize must be greater than 0"):
        normalize_chunks([10, np.array([-1, 2])], shape)


@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_split(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        results = A.ss.split([4, 3])
        for results in [A.ss.split([4, 3]), A.ss.split([[4, None], 3], name="split")]:
            row_boundaries = [0, 4, 7]
            col_boundaries = [0, 3, 6, 7]
            for i, (i1, i2) in enumerate(zip(row_boundaries[:-1], row_boundaries[1:])):
                for j, (j1, j2) in enumerate(zip(col_boundaries[:-1], col_boundaries[1:])):
                    expected = A[i1:i2, j1:j2].new()
                    assert expected.isequal(results[i][j])
        with pytest.raises(DimensionMismatch):
            A.ss.split([[5, 5], 3])


@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_concat(A, A_chunks, v):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        B1 = grblas.ss.concat([[A, A]], dtype=float)
        assert B1.dtype == "FP64"
        expected = Matrix.new(A.dtype, nrows=A.nrows, ncols=2 * A.ncols)
        expected[:, : A.ncols] = A
        expected[:, A.ncols :] = A
        assert B1.isequal(expected)

        B2 = Matrix.new(A.dtype, nrows=2 * A.nrows, ncols=A.ncols)
        B2.ss.concat([[A], [A]])
        expected = Matrix.new(A.dtype, nrows=2 * A.nrows, ncols=A.ncols)
        expected[: A.nrows, :] = A
        expected[A.nrows :, :] = A
        assert B2.isequal(expected)

        tiles = A.ss.split([4, 3])
        A2 = grblas.ss.concat(tiles)
        assert A2.isequal(A)

        with pytest.raises(TypeError, match="tiles argument must be list or tuple"):
            grblas.ss.concat(1)
        # with pytest.raises(TypeError, match="Each tile must be a Matrix"):
        assert grblas.ss.concat([[A.T]]).isequal(A.T)
        with pytest.raises(TypeError, match="tiles must be lists or tuples"):
            grblas.ss.concat([A])

        with pytest.raises(ValueError, match="tiles argument must not be empty"):
            grblas.ss.concat([])
        with pytest.raises(ValueError, match="tiles must not be empty"):
            grblas.ss.concat([[]])
        with pytest.raises(ValueError, match="tiles must all be the same length"):
            grblas.ss.concat([[A], [A, A]])

        # Treat vectors like Nx1 matrices
        B3 = grblas.ss.concat([[v, v]])
        expected = Matrix.new(v.dtype, nrows=v.size, ncols=2)
        expected[:, 0] = v
        expected[:, 1] = v
        assert B3.isequal(expected)

        B4 = grblas.ss.concat([[v], [v]])
        expected = Matrix.new(v.dtype, nrows=2 * v.size, ncols=1)
        expected[: v.size, 0] = v
        expected[v.size :, 0] = v
        assert B4.isequal(expected)

        B5 = grblas.ss.concat([[A, v]])
        expected = Matrix.new(v.dtype, nrows=v.size, ncols=A.ncols + 1)
        expected[:, : A.ncols] = A
        expected[:, A.ncols] = v
        assert B5.isequal(expected)

        with pytest.raises(TypeError, match=""):
            grblas.ss.concat([v, [v]])
        with pytest.raises(TypeError):
            grblas.ss.concat([[v], v])


@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_nbytes(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        assert A.ss.nbytes > 0


@autocompute
def test_auto(As, A_chunks, v):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            expected = binary.land[bool](A & A).new()
            B = A.dup(dtype=bool)
            for expr in [(B & B), binary.land[bool](A & A)]:
                assert expr.dtype == expected.dtype
                assert expr.nrows == expected.nrows
                assert expr.ncols == expected.ncols
                assert expr.shape == expected.shape
                assert expr.nvals == expected.nvals
                assert expr.isclose(expected)
                assert expected.isclose(expr)
                assert expr.isequal(expected)
                assert expected.isequal(expr)
                assert expr.mxv(v).isequal(expected.mxv(v))
                assert expected.T.mxv(v).isequal(expr.T.mxv(v))
                for method in [
                    # "ewise_add",
                    # "ewise_mult",
                    # "mxm",
                    # "__matmul__",
                    "__and__",
                    "__or__",
                    "kronecker",
                ]:
                    val1 = getattr(expected, method)(expected).new()
                    val2 = getattr(expected, method)(expr)
                    val3 = getattr(expr, method)(expected)
                    val4 = getattr(expr, method)(expr)
                    assert val1.isequal(val2)
                    assert val1.isequal(val3)
                    assert val1.isequal(val4)
                for method in ["reduce_rowwise", "reduce_columnwise", "reduce_scalar"]:
                    s1 = getattr(expected, method)(monoid.lor).new()
                    s2 = getattr(expr, method)(monoid.lor)
                    assert s1.isequal(s2.new())
                    assert s1.isequal(s2)

            expected = binary.times(A & A).new()
            for expr in [binary.times(A & A)]:
                assert expr.dtype == expected.dtype
                assert expr.nrows == expected.nrows
                assert expr.ncols == expected.ncols
                assert expr.shape == expected.shape
                assert expr.nvals == expected.nvals
                assert expr.isclose(expected)
                assert expected.isclose(expr)
                assert expr.isequal(expected)
                assert expected.isequal(expr)
                assert expr.mxv(v).isequal(expected.mxv(v))
                assert expected.T.mxv(v).isequal(expr.T.mxv(v))
                for method in [
                    "ewise_add",
                    "ewise_mult",
                    "mxm",
                    # "__matmul__",
                    # "__and__",
                    # "__or__",
                    # "kronecker",
                ]:
                    val1 = getattr(expected, method)(expected).new()
                    val2 = getattr(expected, method)(expr)
                    val3 = getattr(expr, method)(expected)
                    val4 = getattr(expr, method)(expr)
                    assert val1.isequal(val2)
                    assert val1.isequal(val3)
                    assert val1.isequal(val4)
                for method in ["reduce_rowwise", "reduce_columnwise", "reduce_scalar"]:
                    s1 = getattr(expected, method)().new()
                    s2 = getattr(expr, method)()
                    assert s1.isequal(s2.new())
                    assert s1.isequal(s2)

            expected = semiring.plus_times(A @ v).new()
            for expr in [(A @ v), (v @ A.T), semiring.plus_times(A @ v)]:
                assert expr.vxm(A).isequal(expected.vxm(A))
                assert expr.vxm(A).new(mask=expr.S).isequal(expected.vxm(A).new(mask=expected.S))
                assert expr.vxm(A).new(mask=expr.V).isequal(expected.vxm(A).new(mask=expected.V))


@autocompute
def test_auto_assign(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            expected = A.dup()
            B = A[1:4, 1:4].new(dtype=bool)
            expr = B & B
            expected[:3, :3] = expr.new()
            A[:3, :3] = expr
            assert expected.isequal(A)
            v = A[2:5, 5].new(dtype=bool)
            expr = v & v
            A[:3, 4] << expr
            expected[:3, 4] << expr.new()
            assert expected.isequal(A)
            C = A[1:4, 1:4].new()
            A[:3, :3] = A[1:4, 1:4]
            assert A[:3, :3].isequal(C)


@autocompute
@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_expr_is_like_matrix(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        B = A.dup(dtype=bool)
        attrs = {attr for attr, val in inspect.getmembers(B)}
        expr_attrs = {attr for attr, val in inspect.getmembers(binary.times(B & B))}
        infix_attrs = {attr for attr, val in inspect.getmembers(B & B)}
        transposed_attrs = {attr for attr, val in inspect.getmembers(B.T)}
        # Should we make any of these raise informative errors?
        expected = {
            "__call__",
            "__del__",
            "__delitem__",
            "__lshift__",
            "__setitem__",
            "_assign_element",
            "_delete_element",
            "_deserialize",
            "_extract_element",
            "_name_counter",
            "_parent",
            "_prep_for_assign",
            "_prep_for_extract",
            "_update",
            "build",
            "clear",
            "from_pygraphblas",
            "from_values",
            "resize",
            "update",
        }
        assert attrs - expr_attrs == expected
        assert attrs - infix_attrs == expected | {
            "_expect_op",
            "_expect_type",
        }
        # TransposedMatrix is used differently than other expressions,
        # so maybe it shouldn't support everything.
        assert attrs - transposed_attrs == (expected | {"S", "V", "ss", "to_pygraphblas"}) - {
            "_prep_for_extract",
            "_extract_element",
        }


@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_flatten(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        data = [
            [3, 0, 3, 5, 6, 0, 6, 1, 6, 2, 4, 1],
            [0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6],
            [3, 2, 3, 1, 5, 3, 7, 8, 3, 1, 7, 4],
        ]
        # row-wise
        indices = [row * A.ncols + col for row, col in zip(data[0], data[1])]
        expected = Vector.from_values(indices, data[2], size=A.nrows * A.ncols)
        for fmt in ["csr", "hypercsr", "bitmapr"]:
            B = Matrix.ss.import_any(**A.ss.export(format=fmt))
            v = B.ss.flatten()
            assert v.isequal(expected)
            C = v.ss.reshape(*B.shape)
            assert C.isequal(B)
        B(mask=~B.S)[:, :] = 10
        expected(mask=~expected.S)[:] = 10
        B = Matrix.ss.import_fullr(**B.ss.export(format="fullr"))
        v = B.ss.flatten()
        assert v.isequal(expected)
        C = v.ss.reshape(*B.shape)
        assert C.isequal(B)
        C = v.ss.reshape(B.shape)
        assert C.isequal(B)

        # column-wise
        indices = [col * A.nrows + row for row, col in zip(data[0], data[1])]
        expected = Vector.from_values(indices, data[2], size=A.nrows * A.ncols)
        for fmt in ["csc", "hypercsc", "bitmapc"]:
            B = Matrix.ss.import_any(**A.ss.export(format=fmt))
            v = B.ss.flatten(order="col")
            assert v.isequal(expected)
            C = v.ss.reshape(*B.shape, order="col")
            assert C.isequal(B)
        B(mask=~B.S)[:, :] = 10
        expected(mask=~expected.S)[:] = 10
        B = Matrix.ss.import_fullc(**B.ss.export(format="fullc"))
        v = B.ss.flatten(order="F")
        assert v.isequal(expected)
        C = v.ss.reshape(*B.shape, order="F")
        assert C.isequal(B)
        C = v.ss.reshape(B.shape, order="F")
        assert C.isequal(B)
        with pytest.raises(ValueError, match="Bad value for order"):
            A.ss.flatten(order="bad")
        with pytest.raises(ValueError, match="cannot reshape"):
            v.ss.reshape(100, 100)
        with pytest.raises(ValueError):
            v.ss.reshape(A.shape + (1,))


def test_autocompute_argument_messages(As, A_chunks, v):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            with pytest.raises(TypeError, match="autocompute"):
                A.ewise_mult(A & A)
            with pytest.raises(TypeError, match="autocompute"):
                A.mxv(A @ v)


@autocompute
def test_infix_sugar(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            assert type(A + 1) is not Matrix
            assert binary.plus(A, 1).isequal(A + 1)
            assert binary.plus(A.T, 1).isequal(A.T + 1)
            assert binary.plus(1, A).isequal(1 + A)
            assert binary.minus(A, 1).isequal(A - 1)
            assert binary.minus(1, A).isequal(1 - A)
            assert binary.times(A, 2).isequal(A * 2)
            assert binary.times(2, A).isequal(2 * A)
            assert binary.truediv(A, 2).isequal(A / 2)
            assert binary.truediv(5, A).isequal(5 / A)
            assert binary.floordiv(A, 2).isequal(A // 2)
            assert binary.floordiv(5, A).isequal(5 // A)
            assert binary.numpy.mod(A, 2).isequal(A % 2)
            assert binary.numpy.mod(5, A).isequal(5 % A)
            assert binary.pow(A, 2).isequal(A ** 2)
            assert binary.pow(2, A).isequal(2 ** A)
            assert binary.pow(A, 2).isequal(pow(A, 2))
            assert unary.ainv(A).isequal(-A)
            assert unary.ainv(A.T).isequal(-A.T)
            B = A.dup(dtype=bool)
            assert unary.lnot(B).isequal(~B)
            assert unary.lnot(B.T).isequal(~B.T)
            with pytest.raises(TypeError):
                assert unary.lnot(A).isequal(~A)
            with pytest.raises(TypeError):
                assert unary.lnot(A.T).isequal(~A.T)
            assert binary.lxor(True, B).isequal(True ^ B)
            assert binary.lxor(B, True).isequal(B ^ True)
            with pytest.raises(TypeError):
                A ^ True
            with pytest.raises(TypeError):
                A ^ B
            with pytest.raises(TypeError):
                6 ^ B
            assert binary.lt(A, 4).isequal(A < 4)
            assert binary.le(A, 4).isequal(A <= 4)
            assert binary.gt(A, 4).isequal(A > 4)
            assert binary.ge(A, 4).isequal(A >= 4)
            assert binary.eq(A, 4).isequal(A == 4)
            assert binary.ne(A, 4).isequal(A != 4)
            x, y = divmod(A, 3)
            assert binary.floordiv(A, 3).isequal(x)
            assert binary.numpy.mod(A, 3).isequal(y)
            assert binary.fmod(A, 3).isequal(y)
            assert A.isequal(binary.plus((3 * x) & y))
            x, y = divmod(-A, 3)
            assert binary.floordiv(-A, 3).isequal(x)
            assert binary.numpy.mod(-A, 3).isequal(y)
            # assert binary.fmod(-A, 3).isequal(y)  # The reason we use numpy.mod
            assert (-A).isequal(binary.plus((3 * x) & y))
            x, y = divmod(3, A)
            assert binary.floordiv(3, A).isequal(x)
            assert binary.numpy.mod(3, A).isequal(y)
            assert binary.fmod(3, A).isequal(y)
            assert binary.plus(binary.times(A & x) & y).isequal(3 * unary.one(A))
            x, y = divmod(-3, A)
            assert binary.floordiv(-3, A).isequal(x)
            assert binary.numpy.mod(-3, A).isequal(y)
            # assert binary.fmod(-3, A).isequal(y)  # The reason we use numpy.mod
            assert binary.plus(binary.times(A & x) & y).isequal(-3 * unary.one(A))

            assert binary.eq(A & A).isequal(A == A)
            assert binary.ne(A.T & A.T).isequal(A.T != A.T)
            assert binary.lt(A & A.T).isequal(A < A.T)
            assert binary.ge(A.T & A).isequal(A.T >= A)

            B = A.dup()
            B += 1
            assert type(B) is Matrix
            assert binary.plus(A, 1).isequal(B)
            B = A.dup()
            B -= 1
            assert type(B) is Matrix
            assert binary.minus(A, 1).isequal(B)
            B = A.dup()
            B *= 2
            assert type(B) is Matrix
            assert binary.times(A, 2).isequal(B)
            B = A.dup(dtype=float)
            B /= 2
            assert type(B) is Matrix
            assert binary.truediv(A, 2).isequal(B)
            B = A.dup()
            B //= 2
            assert type(B) is Matrix
            assert binary.floordiv(A, 2).isequal(B)
            B = A.dup()
            B %= 2
            assert type(B) is Matrix
            assert binary.numpy.mod(A, 2).isequal(B)
            B = A.dup()
            B **= 2
            assert type(B) is Matrix
            assert binary.pow(A, 2).isequal(B)
            B = A.dup(dtype=bool)
            B ^= True
            assert type(B) is Matrix
            assert B.isequal(~A.dup(dtype=bool))
            B = A.dup(dtype=bool)
            B ^= B
            assert type(B) is Matrix
            assert not B.reduce_scalar(agg.any).new()

            expr = binary.plus(A & A)
            assert unary.abs(expr).isequal(abs(expr))
            assert unary.ainv(expr).isequal(-expr)
            with pytest.raises(TypeError):
                assert unary.lnot(expr).isequal(~expr)
            with pytest.raises(TypeError):
                expr += 1
            with pytest.raises(TypeError):
                expr -= 1
            with pytest.raises(TypeError):
                expr *= 1
            with pytest.raises(TypeError):
                expr /= 1
            with pytest.raises(TypeError):
                expr //= 1
            with pytest.raises(TypeError):
                expr %= 1
            with pytest.raises(TypeError):
                expr **= 1
            with pytest.raises(TypeError):
                expr ^= 1


@pytest.mark.slow
@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_random(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        R = A.ss.selectk_rowwise("random", 1)
        counts = R.reduce_rowwise(agg.count).new()
        expected = Vector.from_values(range(A.ncols), 1)
        assert counts.isequal(expected)

        R = A.ss.selectk_columnwise("random", 1)
        counts = R.reduce_columnwise(agg.count).new()
        expected = Vector.from_values(range(A.nrows), 1)
        assert counts.isequal(expected)

        R = A.ss.selectk_rowwise("random", 2)
        counts = R.reduce_rowwise(agg.count).new()
        assert counts.reduce(monoid.min).new() == 1
        assert counts.reduce(monoid.max).new() == 2

        # test iso
        A(A.S) << 1
        R = A.ss.selectk_rowwise("random", 1)
        counts = R.reduce_rowwise(agg.count).new()
        expected = Vector.from_values(range(A.ncols), 1)
        assert counts.isequal(expected)

        with pytest.raises(ValueError):
            A.ss.selectk_rowwise("bad", 1)
        with pytest.raises(ValueError):
            A.ss.selectk_columnwise("bad", 1)
        with pytest.raises(ValueError):
            A.ss.selectk_columnwise("random", -1)


@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_firstk(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        B = A.ss.selectk_rowwise("first", 1)
        expected = Matrix.from_values(
            [0, 1, 2, 3, 4, 5, 6],
            [1, 4, 5, 0, 5, 2, 2],
            [2, 8, 1, 3, 7, 1, 5],
            nrows=A.nrows,
            ncols=A.ncols,
        )
        assert B.isequal(expected)

        B = A.ss.selectk_rowwise("first", 2)
        expected = Matrix.from_values(
            [3, 0, 3, 5, 6, 0, 6, 1, 2, 4, 1],
            [0, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6],
            [3, 2, 3, 1, 5, 3, 7, 8, 1, 7, 4],
            nrows=A.nrows,
            ncols=A.ncols,
        )
        assert B.isequal(expected)

        B = A.ss.selectk_rowwise("first", 3)
        assert B.isequal(A)

        B = A.ss.selectk_columnwise("first", 1)
        expected = Matrix.from_values(
            [3, 0, 3, 0, 1, 2, 1],
            [0, 1, 2, 3, 4, 5, 6],
            [3, 2, 3, 3, 8, 1, 4],
            nrows=A.nrows,
            ncols=A.ncols,
        )
        assert B.isequal(expected)

        B = A.ss.selectk_columnwise("first", 2)
        expected = Matrix.from_values(
            [3, 0, 3, 5, 0, 6, 1, 6, 2, 4, 1],
            [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
            [3, 2, 3, 1, 3, 7, 8, 3, 1, 7, 4],
            nrows=A.nrows,
            ncols=A.ncols,
        )
        assert B.isequal(expected)

        B = A.ss.selectk_columnwise("first", 3)
        assert B.isequal(A)


@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_lastk(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        B = A.ss.selectk_rowwise("last", 1)
        expected = Matrix.from_values(
            [0, 3, 5, 6, 2, 4, 1],
            [3, 2, 2, 4, 5, 5, 6],
            [3, 3, 1, 3, 1, 7, 4],
            nrows=A.nrows,
            ncols=A.ncols,
        )
        assert B.isequal(expected)

        B = A.ss.selectk_rowwise("last", 2)
        expected = Matrix.from_values(
            [3, 0, 3, 5, 0, 6, 1, 6, 2, 4, 1],
            [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
            [3, 2, 3, 1, 3, 7, 8, 3, 1, 7, 4],
            nrows=A.nrows,
            ncols=A.ncols,
        )
        assert B.isequal(expected)

        B = A.ss.selectk_rowwise("last", 3)
        assert B.isequal(A)

        B = A.ss.selectk_columnwise("last", 1)
        expected = Matrix.from_values(
            [3, 0, 6, 6, 6, 4, 1],
            [0, 1, 2, 3, 4, 5, 6],
            [3, 2, 5, 7, 3, 7, 4],
            nrows=A.nrows,
            ncols=A.ncols,
        )
        assert B.isequal(expected)

        B = A.ss.selectk_columnwise("last", 2)
        expected = Matrix.from_values(
            [3, 0, 5, 6, 0, 6, 1, 6, 2, 4, 1],
            [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
            [3, 2, 1, 5, 3, 7, 8, 3, 1, 7, 4],
            nrows=A.nrows,
            ncols=A.ncols,
        )
        assert B.isequal(expected)

        B = A.ss.selectk_columnwise("last", 3)
        assert B.isequal(A)


@pytest.mark.parametrize("do_iso", [False, True])
@pytest.mark.slow
@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_compactify(A, A_chunks, do_iso):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        if do_iso:
            r, c, v = A.to_values()
            A = Matrix.from_values(r, c, 1)
        rows = [0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6]
        new_cols = [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 2]
        orig_cols = [1, 3, 4, 6, 5, 0, 2, 5, 2, 2, 3, 4]

        def check(A, expected, *args, stop=0, **kwargs):
            B = A.ss.compactify_rowwise(*args, **kwargs)
            assert B.isequal(expected)
            for n in reversed(range(stop, 4)):
                expected = expected[:, :n].new()
                B = A.ss.compactify_rowwise(*args, ncols=n, **kwargs)
                assert B.isequal(expected)

        def reverse(A, A_chunks):
            return A[:, ::-1].new().ss.compactify_rowwise("first", A.ncols)

        def check_reverse(A, expected, *args, stop=0, **kwargs):
            B = A.ss.compactify_rowwise(*args, reverse=True, **kwargs)
            C = reverse(expected)
            assert B.isequal(C)
            for n in reversed(range(stop, 4)):
                C = reverse(expected[:, :n].new())
                B = A.ss.compactify_rowwise(*args, ncols=n, reverse=True, **kwargs)
                assert B.isequal(C)

        expected = Matrix.from_values(
            rows,
            new_cols,
            1 if do_iso else [2, 3, 8, 4, 1, 3, 3, 7, 1, 5, 7, 3],
            nrows=A.nrows,
            ncols=3,
        )
        check(A, expected, "first")
        check_reverse(A, expected, "first")
        check(A, reverse(expected), "last")
        check_reverse(A, reverse(expected), "last")

        expected = Matrix.from_values(
            rows,
            new_cols,
            orig_cols,
            nrows=A.nrows,
            ncols=3,
        )
        check(A, expected, "first", asindex=True)
        check_reverse(A, expected, "first", asindex=True)
        check(A, reverse(expected), "last", asindex=True)
        check_reverse(A, reverse(expected), "last", asindex=True)

        expected = Matrix.from_values(
            rows,
            new_cols,
            1 if do_iso else [2, 3, 4, 8, 1, 3, 3, 7, 1, 3, 5, 7],
            nrows=A.nrows,
            ncols=3,
        )
        check(A, expected, "smallest")
        check_reverse(A, expected, "smallest")
        check(A, reverse(expected), "largest")
        check_reverse(A, reverse(expected), "largest")

        if not do_iso:
            expected = Matrix.from_values(
                rows,
                new_cols,
                [1, 3, 6, 4, 5, 0, 2, 5, 2, 4, 2, 3],
                nrows=A.nrows,
                ncols=3,
            )
            check(A, expected, "smallest", asindex=True, stop=2)
            check_reverse(A, expected, "smallest", asindex=True, stop=2)
            check(A, reverse(expected), "largest", asindex=True, stop=2)
            check_reverse(A, reverse(expected), "largest", asindex=True, stop=2)

        def compare(A, expected, isequal=True, **kwargs):
            for _ in range(1000):
                B = A.ss.compactify_rowwise("random", **kwargs)
                if B.isequal(expected) == isequal:
                    break
            else:
                raise AssertionError("random failed")

        with pytest.raises(AssertionError):
            compare(A, A[:, ::-1].new())

        for asindex in [False, True]:
            compare(A, A.ss.compactify_rowwise("first", asindex=asindex), asindex=asindex)
            compare(
                A, A.ss.compactify_rowwise("first", 3, asindex=asindex), ncols=3, asindex=asindex
            )
            compare(
                A, A.ss.compactify_rowwise("first", 2, asindex=asindex), ncols=2, asindex=asindex
            )
            compare(
                A,
                A.ss.compactify_rowwise("first", 2, asindex=asindex),
                ncols=2,
                asindex=asindex,
                isequal=do_iso,
            )
            compare(
                A, A.ss.compactify_rowwise("first", 1, asindex=asindex), ncols=1, asindex=asindex
            )
            compare(
                A,
                A.ss.compactify_rowwise("first", 1, asindex=asindex),
                ncols=1,
                asindex=asindex,
                isequal=do_iso,
            )
            compare(
                A, A.ss.compactify_rowwise("last", 1, asindex=asindex), ncols=1, asindex=asindex
            )
            compare(
                A, A.ss.compactify_rowwise("smallest", 1, asindex=asindex), ncols=1, asindex=asindex
            )
            compare(
                A, A.ss.compactify_rowwise("largest", 1, asindex=asindex), ncols=1, asindex=asindex
            )
            compare(
                A, A.ss.compactify_rowwise("first", 0, asindex=asindex), ncols=0, asindex=asindex
            )

        B = A.ss.compactify_columnwise("first", nrows=1)
        expected = Matrix.from_values(
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 3, 4, 5, 6],
            1 if do_iso else [3, 2, 3, 3, 8, 1, 4],
        )
        assert B.isequal(expected)
        B = A.ss.compactify_columnwise("last", nrows=1, asindex=True)
        expected = Matrix.from_values(
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 3, 4, 5, 6],
            [3, 0, 6, 6, 6, 4, 1],
        )
        assert B.isequal(expected)
        with pytest.raises(ValueError):
            A.ss.compactify_rowwise("bad_how")


@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_deprecated(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        with pytest.warns(DeprecationWarning):
            A.reduce_rows()
        with pytest.warns(DeprecationWarning):
            A.reduce_columns()
        with pytest.warns(DeprecationWarning):
            A.ss.scan_rows()
        with pytest.warns(DeprecationWarning):
            A.ss.scan_columns()


def test_ndim(As, A_chunks):
    for A_ in As:
        for chunks in A_chunks:
            A = A_.dup()
            A.rechunk(chunks=chunks, inplace=True)
            assert A.ndim == 2
            assert A.ewise_mult(A).ndim == 2
            assert (A & A).ndim == 2
            assert (A @ A).ndim == 2


@pytest.mark.xfail("'Needs investigation'", strict=True)
def test_sizeof(A, A_chunks):
    A_ = A
    for chunks in A_chunks:
        A = A_.dup()
        A.rechunk(chunks=chunks, inplace=True)
        assert sys.getsizeof(A) > A.nvals * 16
