from graphblas.core.base import _expect_type
from ..matrix import Matrix, TransposedMatrix
from ..scalar import Scalar
from ..vector import Vector


class _grblas_ss:
    """Used in `_expect_type`"""


_grblas_ss.__name__ = "grblas.ss"
_grblas_ss = _grblas_ss()


def diag(x, k=0, dtype=None, chunks="auto", *, name=None):
    """
    GxB_Matrix_diag, GxB_Vector_diag

    Extract a diagonal Vector from a Matrix, or construct a diagonal Matrix
    from a Vector.  Unlike ``Matrix.diag`` and ``Vector.diag``, this function
    returns a new object.

    Parameters
    ----------
    x : Vector or Matrix
        The Vector to assign to the diagonal, or the Matrix from which to
        extract the diagonal.
    k : int, default 0
        Diagonal in question.  Use `k>0` for diagonals above the main diagonal,
        and `k<0` for diagonals below the main diagonal.

    See Also
    --------
    Vector.ss.diag
    Matrix.ss.diag

    """
    x = _expect_type(_grblas_ss, x, (Matrix, TransposedMatrix, Vector), within="diag", argname="x")
    if type(k) is Scalar:
        k = k.value.compute()
    if dtype is None:
        dtype = x.dtype
    typ = type(x)
    if typ is Vector:
        size = x._size + abs(k)
        rv = Matrix.new(dtype, nrows=size, ncols=size, name=name)
        rv.ss.diag(x, k, dtype=dtype, chunks=chunks)
    else:
        if k < 0:
            size = min(x._nrows + k, x._ncols)
        else:
            size = min(x._ncols - k, x._nrows)
        if size < 0:
            size = 0
        rv = Vector.new(dtype, size=size, name=name)
        rv.ss.diag(x, k, dtype=dtype, chunks=chunks)
    return rv
