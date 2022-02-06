import dask_grblas as dgb


class ss:
    __slots__ = "_parent"

    def __init__(self, parent):
        self._parent = parent

    def diag(self, matrix, k=0, chunks="auto", dtype=None):
        """
        GxB_Vector_diag

        Extract a diagonal from a Matrix or TransposedMatrix into a Vector.
        Existing entries in the Vector are discarded.

        Parameters
        ----------
        matrix : Matrix or TransposedMatrix
            Extract a diagonal from this matrix.
        k : int, default 0
            Diagonal in question.  Use `k>0` for diagonals above the main diagonal,
            and `k<0` for diagonals below the main diagonal.

        See Also
        --------
        grblas.ss.diag
        Matrix.ss.diag

        """
        matrix = self._parent._expect_type(
            matrix, (dgb.Matrix, dgb.matrix.TransposedMatrix), within="ss.diag", argname="matrix"
        )
        if type(matrix) is dgb.matrix.TransposedMatrix:
            # Transpose descriptor doesn't do anything, so use the parent
            k = -k
            matrix = matrix._matrix
        rv = matrix._diag(k, chunks=chunks, dtype=dtype)
        self._parent.__init__(rv._delayed, nvals=rv._nvals)

