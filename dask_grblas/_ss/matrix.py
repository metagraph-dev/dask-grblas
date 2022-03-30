import dask_grblas as dgb


class ss:
    __slots__ = "_parent"

    def __init__(self, parent):
        self._parent = parent

    def diag(self, vector, k=0, chunks="auto", dtype=None):
        """
        GxB_Matrix_diag

        Construct a diagonal Matrix from the given vector.
        Existing entries in the Matrix are discarded.

        Parameters
        ----------
        vector : Vector
            Create a diagonal from this Vector.
        k : int, default 0
            Diagonal in question.  Use `k>0` for diagonals above the main diagonal,
            and `k<0` for diagonals below the main diagonal.

        See Also
        --------
        grblas.ss.diag
        Vector.ss.diag
        """
        vector = self._parent._expect_type(vector, dgb.Vector, within="ss.diag", argname="vector")
        rv = vector._diag(k, chunks=chunks, dtype=dtype)
        self._parent.__init__(rv._delayed, nvals=rv._nvals)

    def build_scalar(
        self,
        rows,
        columns,
        values,
        *,
        dup_op=None,
        clear=False,
        nrows=None,
        ncols=None,
        chunks=None,
        in_dOnion=False,  # not part of the API
    ):
        self._parent.build(
            rows,
            columns,
            values,
            dup_op=dup_op,
            clear=clear,
            nrows=nrows,
            ncols=ncols,
            chunks=chunks,
        )
