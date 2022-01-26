import dask.array as da
import numpy as np
import grblas as gb
from dask.base import tokenize
from dask.delayed import Delayed, delayed
from grblas import binary, monoid, semiring

from .base import BaseType, InnerBaseType
from .expr import AmbiguousAssignOrExtract, GbDelayed, Updater
from .mask import StructuralMask, ValueMask
from .utils import (
    np_dtype,
    wrap_inner,
    build_ranges_dask_array_from_chunks,
    build_chunk_ranges_dask_array,
    wrap_dataframe,
)


class InnerMatrix(InnerBaseType):
    ndim = 2

    def __init__(self, grblas_matrix):
        assert type(grblas_matrix) is gb.Matrix
        self.value = grblas_matrix
        self.shape = grblas_matrix.shape
        self.dtype = np_dtype(grblas_matrix.dtype)

    def __getitem__(self, index):
        # This always copies!
        assert type(index) is tuple and len(index) == 2
        value = self.value[index].new()
        return wrap_inner(value)


class Matrix(BaseType):
    ndim = 2
    _is_transposed = False

    @classmethod
    def from_delayed(cls, matrix, dtype, nrows, ncols, *, name=None):
        if not isinstance(matrix, Delayed):
            raise TypeError(
                "Value is not a dask delayed object.  "
                "Please use dask.delayed to create a grblas.Matrix"
            )
        inner = delayed(InnerMatrix)(matrix)
        value = da.from_delayed(inner, (nrows, ncols), dtype=np_dtype(dtype), name=name)
        return cls(value)

    @classmethod
    def from_matrix(cls, matrix, chunks=None, *, name=None):
        if not isinstance(matrix, gb.Matrix):
            raise TypeError("Value is not a grblas.Matrix")
        if chunks is not None:
            raise NotImplementedError()
        return cls.from_delayed(delayed(matrix), matrix.dtype, *matrix.shape, name=name)

    @classmethod
    def from_MMfile(cls, filename, chunks="auto", nreaders=3):
        df, nrows, ncols = wrap_dataframe(filename, nreaders)
        rows = df["r"].to_dask_array()
        cols = df["c"].to_dask_array()
        vals = df["d"].to_dask_array()
        return cls.from_values(rows, cols, vals, chunks=chunks, nrows=nrows, ncols=ncols)
        # chunks = da.core.normalize_chunks(chunks, (nrows, ncols), dtype=np.int64)
        # row_ranges = build_ranges_dask_array_from_chunks(chunks[0], 'chunks-' + tokenize(chunks[0]))
        # col_ranges = build_ranges_dask_array_from_chunks(chunks[1], 'chunks-' + tokenize(chunks[1]))
        # dtype = df['d'].dtype
        # meta = InnerMatrix(gb.Matrix.new(dtype))
        # delayed = da.core.blockwise(_df_to_chunk, 'ij', row_ranges, 'i', col_ranges, 'j', rcd_df=df, dtype=dtype, meta=meta)
        # return Matrix(delayed)

    def to_MMfile(self, target):
        import os

        delayed = self._delayed
        row_ranges = build_chunk_ranges_dask_array(delayed, 0, "row-ranges-" + tokenize(delayed, 0))
        col_ranges = build_chunk_ranges_dask_array(delayed, 1, "col-ranges-" + tokenize(delayed, 1))
        saved = da.core.blockwise(
            *(_mmwrite_chunk, "ij"),
            *(delayed, "ij"),
            *(row_ranges, "i"),
            *(col_ranges, "j"),
            nrows=self.nrows,
            ncols=self.ncols,
            final_target=target,
            adjust_chunks={"i": 4, "j": 0},
            dtype=object,
            meta=np.array([], dtype=object),
        )
        saved = da.reduction(
            saved,
            _identity,
            _concatenate_files,
            axis=(1,),
            concatenate=False,
            dtype=object,
            meta=np.array([], dtype=object),
        )
        saved = da.reduction(
            saved,
            _identity,
            _concatenate_files,
            axis=(0,),
            concatenate=False,
            dtype=object,
            meta=np.array([], dtype=object),
        ).compute()

        os.rename(saved[0], target)

    @classmethod
    def from_values(
        cls,
        rows,
        columns,
        values,
        *,
        nrows=None,
        ncols=None,
        dup_op=None,
        dtype=None,
        chunks="auto",
        name=None,
    ):
        if (
            dup_op is None
            and type(rows) is da.Array
            and type(columns) is da.Array
            and type(values) is da.Array
        ):
            implied_nrows = 1 + da.max(rows).compute()
            implied_ncols = 1 + da.max(columns).compute()
            if nrows is not None and implied_nrows > nrows:
                raise Exception()
            if ncols is not None and implied_ncols > ncols:
                raise Exception()
            nrows = implied_nrows if nrows is None else nrows
            ncols = implied_ncols if ncols is None else ncols

            idtype = gb.Matrix.new(rows.dtype).dtype
            np_idtype_ = np_dtype(idtype)
            vdtype = gb.Matrix.new(values.dtype).dtype
            np_vdtype_ = np_dtype(vdtype)

            chunks = da.core.normalize_chunks(chunks, (nrows, ncols), dtype=np_idtype_)

            name_ = name
            name = str(name) if name else ""
            rname = name + "-row-ranges" + tokenize(cls, chunks[0])
            cname = name + "-col-ranges" + tokenize(cls, chunks[1])
            row_ranges = build_ranges_dask_array_from_chunks(chunks[0], rname)
            col_ranges = build_ranges_dask_array_from_chunks(chunks[1], cname)
            fragments = da.core.blockwise(
                *(_pick2D, "ijk"),
                *(rows, "k"),
                *(columns, "k"),
                *(values, "k"),
                *(row_ranges, "i"),
                *(col_ranges, "j"),
                dtype=np_idtype_,
                meta=np.array([]),
            )
            meta = InnerMatrix(gb.Matrix.new(vdtype))
            delayed = da.core.blockwise(
                *(_from_values2D, "ij"),
                *(fragments, "ijk"),
                *(row_ranges, "i"),
                *(col_ranges, "j"),
                concatenate=False,
                gb_dtype=vdtype,
                dtype=np_vdtype_,
                meta=meta,
                name=name_,
            )
            return Matrix(delayed)

        chunks = None
        matrix = gb.Matrix.from_values(
            rows, columns, values, nrows=nrows, ncols=ncols, dup_op=dup_op, dtype=dtype
        )
        return cls.from_matrix(matrix, chunks=chunks, name=name)

    @classmethod
    def new(cls, dtype, nrows=0, ncols=0, *, name=None):
        matrix = gb.Matrix.new(dtype, nrows, ncols)
        return cls.from_delayed(
            delayed(matrix), matrix.dtype, matrix.nrows, matrix.ncols, name=name
        )

    def __init__(self, delayed, meta=None):
        assert type(delayed) is da.Array
        assert delayed.ndim == 2
        self._delayed = delayed
        if meta is None:
            meta = gb.Matrix.new(delayed.dtype, *delayed.shape)
        self._meta = meta
        self.dtype = meta.dtype

    @property
    def S(self):
        return StructuralMask(self)

    @property
    def V(self):
        return ValueMask(self)

    @property
    def T(self):
        return TransposedMatrix(self)

    @property
    def nrows(self):
        return self._meta.nrows

    @property
    def ncols(self):
        return self._meta.ncols

    @property
    def shape(self):
        return (self._meta.nrows, self._meta.ncols)

    def resize(self, nrows, ncols):
        self._meta.resize(nrows, ncols)
        raise NotImplementedError()

    def __getitem__(self, index):
        return AmbiguousAssignOrExtract(self, index)

    def __delitem__(self, index):
        raise NotImplementedError()

    def __setitem__(self, index, delayed):
        Updater(self)[index] = delayed

    def __contains__(self, index):
        extractor = self[index]
        if not extractor.resolved_indexes.is_single_element:
            raise TypeError(
                f"Invalid index to Matrix contains: {index!r}.  A 2-tuple of ints is expected.  "
                "Doing `(i, j) in my_matrix` checks whether a value is present at that index."
            )
        scalar = extractor.new(name="s_contains")
        return not scalar.is_empty

    def __iter__(self):
        rows, columns, _ = self.to_values()
        return zip(rows.flat, columns.flat)

    def ewise_add(self, other, op=monoid.plus, *, require_monoid=True):
        assert type(other) is Matrix  # TODO: or TransposedMatrix
        meta = self._meta.ewise_add(other._meta, op=op, require_monoid=require_monoid)
        return GbDelayed(self, "ewise_add", other, op, require_monoid=require_monoid, meta=meta)

    def ewise_mult(self, other, op=binary.times):
        assert type(other) is Matrix  # TODO: or TransposedMatrix
        meta = self._meta.ewise_mult(other._meta, op=op)
        return GbDelayed(self, "ewise_mult", other, op, meta=meta)

    def mxv(self, other, op=semiring.plus_times):
        from .vector import Vector

        assert type(other) is Vector
        meta = self._meta.mxv(other._meta, op=op)
        return GbDelayed(self, "mxv", other, op, meta=meta)

    def mxm(self, other, op=semiring.plus_times):
        assert type(other) in (Matrix, TransposedMatrix)
        meta = self._meta.mxm(other._meta, op=op)
        return GbDelayed(self, "mxm", other, op, meta=meta)

    def kronecker(self, other, op=binary.times):
        assert type(other) is Matrix  # TODO: or TransposedMatrix
        meta = self._meta.kronecker(other._meta, op=op)
        return GbDelayed(self, "kronecker", other, op, meta=meta)

    def apply(self, op, right=None, *, left=None):
        from .scalar import Scalar

        left_meta = left
        right_meta = right

        if type(left) is Scalar:
            left_meta = left.dtype.np_type(0)
        if type(right) is Scalar:
            right_meta = right.dtype.np_type(0)

        meta = self._meta.apply(op=op, left=left_meta, right=right_meta)
        return GbDelayed(self, "apply", op, right, meta=meta, left=left)

    def reduce_rowwise(self, op=monoid.plus):
        meta = self._meta.reduce_rowwise(op)
        return GbDelayed(self, "reduce_rowwise", op, meta=meta)

    def reduce_columnwise(self, op=monoid.plus):
        meta = self._meta.reduce_columnwise(op)
        return GbDelayed(self, "reduce_columnwise", op, meta=meta)

    def reduce_scalar(self, op=monoid.plus):
        meta = self._meta.reduce_scalar(op)
        return GbDelayed(self, "reduce_scalar", op, meta=meta)

    def build(self, rows, columns, values, *, dup_op=None, clear=False, nrows=None, ncols=None):
        # This doesn't do anything special yet.  Should we have name= and chunks= keywords?
        # TODO: raise if output is not empty
        # This operation could, perhaps, partition rows, columns, and values if there are chunks
        if nrows is None:
            nrows = self.nrows
        if ncols is None:
            ncols = self.ncols
        matrix = gb.Matrix.new(self.dtype, nrows, ncols)
        matrix.build(rows, columns, values, dup_op=dup_op)
        m = Matrix.from_matrix(matrix)
        self._delayed = m._delayed
        self._meta = m._meta

    def to_values(self, dtype=None):
        # TODO: make this lazy; can we do something smart with this?
        return self.compute().to_values(dtype=dtype)

    def isequal(self, other, *, check_dtype=False):
        other = self._expect_type(
            other, (Matrix, TransposedMatrix), within="isequal", argname="other"
        )
        return super().isequal(other, check_dtype=check_dtype)

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        other = self._expect_type(
            other, (Matrix, TransposedMatrix), within="isclose", argname="other"
        )
        return super().isclose(other, rel_tol=rel_tol, abs_tol=abs_tol, check_dtype=check_dtype)

    def _delete_element(self, resolved_indexes):
        row = resolved_indexes.indices[0]
        col = resolved_indexes.indices[1]
        delayed = self._optional_dup()
        row_ranges = build_chunk_ranges_dask_array(delayed, 0, "index-ranges-" + tokenize(delayed, 0))
        col_ranges = build_chunk_ranges_dask_array(delayed, 1, "index-ranges-" + tokenize(delayed, 1))
        deleted = da.core.blockwise(
            *(_delitem_chunk, "ij"),
            *(delayed, "ij"),
            *(row_ranges, "i"),
            *(col_ranges, "j"),
            *(row.index, None),
            *(col.index, None),
            dtype=delayed.dtype,
            meta=delayed._meta,
        )
        self._delayed = deleted


class TransposedMatrix:
    ndim = 2
    _is_transposed = True

    def __init__(self, matrix):
        assert type(matrix) is Matrix
        self._matrix = matrix
        self._meta = matrix._meta.T

    def new(self, *, dtype=None, mask=None):
        raise NotImplementedError("A.T.new()")

    @property
    def T(self):
        return self._matrix

    @property
    def dtype(self):
        return self._meta.dtype

    def to_values(self):
        # TODO: make this lazy; can we do something smart with this?
        rows, cols, vals = self._matrix.to_values()
        return cols, rows, vals

    # Properties
    nrows = Matrix.ncols
    ncols = Matrix.nrows
    shape = Matrix.shape
    nvals = Matrix.nvals

    # Delayed methods
    ewise_add = Matrix.ewise_add
    ewise_mult = Matrix.ewise_mult
    mxv = Matrix.mxv
    mxm = Matrix.mxm
    kronecker = Matrix.kronecker
    apply = Matrix.apply
    reduce_rowwise = Matrix.reduce_rowwise
    reduce_columnwise = Matrix.reduce_columnwise
    reduce_scalar = Matrix.reduce_scalar

    # Misc.
    isequal = Matrix.isequal
    isclose = Matrix.isclose
    __getitem__ = Matrix.__getitem__
    __array__ = Matrix.__array__
    name = Matrix.name


def _delitem_chunk(inner_mat, row_range, col_range, row, col):
    if isinstance(row, gb.Scalar):
        row = row.value
    if isinstance(col, gb.Scalar):
        col = col.value
    if row_range[0].start <= row and row < row_range[0].stop:
        if col_range[0].start <= col and col < col_range[0].stop:
            del inner_mat.value[row - row_range[0].start, col - col_range[0].start]
    return InnerMatrix(inner_mat.value)


def _from_values2D(fragments, row_range, col_range, gb_dtype=None):
    rows = np.concatenate([rows for (rows, _, _) in fragments])
    cols = np.concatenate([cols for (_, cols, _) in fragments])
    vals = np.concatenate([vals for (_, _, vals) in fragments])
    nrows = row_range[0].stop - row_range[0].start
    ncols = col_range[0].stop - col_range[0].start
    return InnerMatrix(
        gb.Matrix.from_values(rows, cols, vals, nrows=nrows, ncols=ncols, dtype=gb_dtype)
    )


def _pick2D(rows, cols, values, row_range, col_range):
    row_range, col_range = row_range[0], col_range[0]
    rows_in = (row_range.start <= rows) & (rows < row_range.stop)
    cols_in = (col_range.start <= cols) & (cols < col_range.stop)
    rows = rows[rows_in & cols_in] - row_range.start
    cols = cols[rows_in & cols_in] - col_range.start
    values = values[rows_in & cols_in]
    return (rows, cols, values)


def _df_to_chunk(row_range, col_range, rcd_df):
    row_range, col_range = row_range[0], col_range[0]
    df = rcd_df[
        (row_range.start <= rcd_df["r"])
        & (rcd_df["r"] < row_range.stop)
        & (col_range.start <= rcd_df["c"])
        & (rcd_df["c"] < col_range.stop)
    ]
    return InnerMatrix(
        gb.Matrix.from_values(df["r"].to_numpy(), df["c"].to_numpy(), df["d"].to_numpy())
    )


def _mmwrite_chunk(chunk, row_range, col_range, nrows, ncols, final_target):
    import os
    from scipy.sparse import coo_matrix
    from scipy.io import mmwrite

    row_range, col_range = row_range[0], col_range[0]

    r, c, d = chunk.value.to_values()
    coo = coo_matrix((d, (row_range.start + r, col_range.start + c)), shape=(nrows, ncols))

    path, basename = os.path.split(final_target)
    basename = (
        f"i{row_range.start}-{row_range.stop}"
        f"j{col_range.start}-{col_range.stop}"
        f"_{basename}.{tokenize(chunk)}.mtx"
    )
    chunk_target = os.path.join(path, basename)

    mmwrite(chunk_target, coo, symmetry="general")
    return np.array([chunk_target, final_target, row_range, col_range])


def _identity(chunk, keepdims=None, axis=None):
    return chunk


def _concatenate_files(chunk_files, keepdims=None, axis=None):
    import os
    import shutil
    from scipy.io.mmio import MMFile, mminfo

    chunk_files = chunk_files if type(chunk_files) is list else [chunk_files]
    first_chunk_file, _, row_range_first, col_range_first = chunk_files[0]
    _, final_target, row_range_last, col_range_last = chunk_files[-1]

    if axis == (0,):
        row_range = slice(row_range_first.start, row_range_last.stop)
        col_range = col_range_first
    else:
        row_range = row_range_first
        col_range = slice(col_range_first.start, col_range_last.stop)

    path, basename = os.path.split(final_target)
    basename = (
        f"i{row_range.start}-{row_range.stop}"
        f"j{col_range.start}-{col_range.stop}"
        f"_{basename}.{tokenize(chunk_files)}.mtx"
    )
    temp_target = os.path.join(path, basename)

    # compute shape spec
    nnz = 0
    for source, _, _, _ in chunk_files:
        nrows, ncols, entries, _, _, _ = mminfo(source)
        nnz += entries

    with open(temp_target, "wb") as outstream:
        # copy header lines from the first chunk file:
        stream, close_it = MMFile()._open(first_chunk_file)

        while stream.read(1) == b"%":
            stream.seek(-1, os.SEEK_CUR)
            data = stream.readline()
            outstream.write(data)

        # write shape specs:
        data = "%i %i %i\n" % (nrows, ncols, nnz)
        outstream.write(data.encode("latin1"))

        if close_it:
            stream.close()

        for source, _, _, _ in chunk_files:
            MMf = MMFile()
            instream, close_it = MMf._open(source)

            try:
                MMf._parse_header(instream)
                shutil.copyfileobj(instream, outstream)

            finally:
                if close_it:
                    instream.close()
                os.remove(source)

    return np.array([temp_target, final_target, row_range, col_range])


@da.core.concatenate_lookup.register(InnerMatrix)
def _concat_matrix(seq, axis=0):
    if axis not in {0, 1}:
        raise ValueError(f"Can only concatenate for axis 0 or 1.  Got {axis}")
    if axis == 0:
        ncols = set(item.value.ncols for item in seq if item.value.ncols > 0)
        if len(ncols) > 1:
            raise Exception("Mismatching number of columns while stacking along axis 0")
        if len(ncols) == 1:
            (ncols,) = ncols
            seq = [
                InnerMatrix(
                    gb.Matrix.new(dtype=item.value.dtype, nrows=item.value.nrows, ncols=ncols)
                )
                if item.value.ncols == 0
                else item
                for item in seq
            ]
        value = gb.ss.concat([[item.value] for item in seq])
    else:
        nrows = set(item.value.nrows for item in seq if item.value.nrows > 0)
        if len(nrows) > 1:
            raise Exception("Mismatching number of rows while stacking along axis 1")
        if len(nrows) == 1:
            (nrows,) = nrows
            seq = [
                InnerMatrix(
                    gb.Matrix.new(dtype=item.value.dtype, nrows=nrows, ncols=item.value.ncols)
                )
                if item.value.nrows == 0
                else item
                for item in seq
            ]
        value = gb.ss.concat([[item.value for item in seq]])
    return InnerMatrix(value)


gb.utils._output_types[Matrix] = gb.Matrix
gb.utils._output_types[TransposedMatrix] = gb.matrix.TransposedMatrix
