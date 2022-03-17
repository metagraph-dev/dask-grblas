import dask.array as da
import numpy as np
import grblas as gb

from numbers import Integral, Number
from dask.base import tokenize, is_dask_collection
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph
from grblas import binary, monoid, semiring
from grblas.dtypes import lookup_dtype
from grblas.exceptions import IndexOutOfBound, EmptyObject, DimensionMismatch

from . import _automethods
from .base import BaseType, InnerBaseType, DOnion, is_DOnion, any_dOnions, Box, skip
from .base import _nvals as _nvals_in_chunk
from .expr import AmbiguousAssignOrExtract, GbDelayed, Updater
from .mask import StructuralMask, ValueMask
from ._ss.matrix import ss
from .utils import (
    pack_args,
    pack_kwargs,
    np_dtype,
    get_return_type,
    get_grblas_type,
    wrap_inner,
    build_chunk_offsets_dask_array,
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
    __slots__ = ("ss",)
    ndim = 2
    _is_transposed = False

    __abs__ = gb.Matrix.__abs__
    __add__ = gb.Matrix.__add__
    __divmod__ = gb.Matrix.__divmod__
    __eq__ = gb.Matrix.__eq__
    __floordiv__ = gb.Matrix.__floordiv__
    __ge__ = gb.Matrix.__ge__
    __gt__ = gb.Matrix.__gt__
    __iadd__ = gb.Matrix.__iadd__
    __iand__ = gb.Matrix.__iand__
    __ifloordiv__ = gb.Matrix.__ifloordiv__
    __imod__ = gb.Matrix.__imod__
    __imul__ = gb.Matrix.__imul__
    __invert__ = gb.Matrix.__invert__
    __ior__ = gb.Matrix.__ior__
    __ipow__ = gb.Matrix.__ipow__
    __isub__ = gb.Matrix.__isub__
    __itruediv__ = gb.Matrix.__itruediv__
    __ixor__ = gb.Matrix.__ixor__
    __le__ = gb.Matrix.__le__
    __lt__ = gb.Matrix.__lt__
    __mod__ = gb.Matrix.__mod__
    __mul__ = gb.Matrix.__mul__
    __ne__ = gb.Matrix.__ne__
    __neg__ = gb.Matrix.__neg__
    __pow__ = gb.Matrix.__pow__
    __radd__ = gb.Matrix.__radd__
    __rdivmod__ = gb.Matrix.__rdivmod__
    __rfloordiv__ = gb.Matrix.__rfloordiv__
    __rmod__ = gb.Matrix.__rmod__
    __rmul__ = gb.Matrix.__rmul__
    __rpow__ = gb.Matrix.__rpow__
    __rsub__ = gb.Matrix.__rsub__
    __rtruediv__ = gb.Matrix.__rtruediv__
    __rxor__ = gb.Matrix.__rxor__
    __sub__ = gb.Matrix.__sub__
    __truediv__ = gb.Matrix.__truediv__
    __xor__ = gb.Matrix.__xor__

    @classmethod
    def from_delayed(cls, matrix, dtype, nrows, ncols, *, nvals=None, name=None):
        if not isinstance(matrix, Delayed):
            raise TypeError(
                "Value is not a dask delayed object.  "
                "Please use dask.delayed to create a grblas.Matrix"
            )
        inner = delayed(InnerMatrix)(matrix)
        value = da.from_delayed(inner, (nrows, ncols), dtype=np_dtype(dtype), name=name)
        return cls(value, nvals=nvals)

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
        /,
        nrows=None,
        ncols=None,
        *,
        dup_op=None,
        dtype=None,
        chunks="auto",
        name=None,
    ):
        if isinstance(values, Number):
            dtype = lookup_dtype(type(values) if dtype is None else dtype)
        elif hasattr(values, "dtype"):
            dtype = lookup_dtype(values.dtype if dtype is None else dtype)

        meta = gb.Matrix.new(
            dtype,
            nrows=nrows if isinstance(nrows, Number) else 0,
            ncols=ncols if isinstance(ncols, Number) else 0,
        )

        # check for any dOnions:
        args = pack_args(rows, columns, values, nrows, ncols)
        kwargs = pack_kwargs(dup_op=dup_op, dtype=dtype, chunks=chunks, name=name)
        if any_dOnions(*args, **kwargs):
            # dive into dOnion(s):
            out_donion = DOnion.multi_access(meta, Matrix.from_values, *args, **kwargs)
            return Matrix(out_donion, meta=meta)

        # no dOnions
        if type(rows) is da.Array or type(columns) is da.Array or type(values) is da.Array:
            nrows_, ncols_ = nrows, ncols
            if type(rows) in {tuple, list, np.ndarray}:
                nrows_ = nrows or (np.max(rows) + 1)
                rows = da.asarray(rows)
            if type(columns) in {tuple, list, np.ndarray}:
                ncols_ = ncols or (np.max(columns) + 1)
                columns = da.asarray(columns)
            if type(values) in {tuple, list, np.ndarray}:
                values = da.asarray(values)

            np_idtype_ = np_dtype(lookup_dtype(rows.dtype))
            if isinstance(nrows_, Integral) and isinstance(ncols_, Integral):
                nrows, ncols = nrows_, ncols_
                chunks = da.core.normalize_chunks(chunks, (nrows, ncols), dtype=np_idtype_)
            else:
                if nrows is None and rows.size == 0:
                    raise ValueError("No row indices provided. Unable to infer nrows.")

                if ncols is None and columns.size == 0:
                    raise ValueError("No column indices provided. Unable to infer ncols.")

                if type(values) is da.Array and (
                    rows.size != columns.size or columns.size != values.size
                ):
                    raise ValueError(
                        "`rows` and `columns` and `values` lengths must match: "
                        f"{rows.size}, {columns.size}, {values.size}"
                    )
                elif rows.size != columns.size:
                    raise ValueError(
                        f"`rows` and `columns` lengths must match: {rows.size}, {columns.size}"
                    )

                if rows.dtype.kind not in "ui":
                    raise ValueError(f"rows must be integers, not {rows.dtype}")

                if columns.dtype.kind not in "ui":
                    raise ValueError(f"columns must be integers, not {columns.dtype}")

                nrows = nrows_
                if nrows is None:
                    nrows = da.max(rows) + np.asarray(1, dtype=rows.dtype)

                ncols = ncols_
                if ncols is None:
                    ncols = da.max(columns) + np.asarray(1, dtype=columns.dtype)

                # Create dOnion from `nrows` and/or `ncols`, that is,
                # use the inner value of `nrows` and/or `ncols` to create the new Matrix:
                shape = (nrows, ncols)
                _shape = [skip if is_dask_collection(x) else x for x in shape]
                dasks = [x for x in shape if is_dask_collection(x)]
                args = pack_args(rows, columns, values, *_shape)
                kwargs = pack_kwargs(dup_op=dup_op, dtype=dtype, chunks=chunks, name=name)
                donion = DOnion.sprout(dasks, meta, Matrix.from_values, *args, **kwargs)
                return Matrix(donion, meta=meta)

            # output shape `(nrows, ncols)` is completely determined
            vdtype = dtype
            np_vdtype_ = np_dtype(vdtype)

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
                *(values, "k" if type(values) is da.Array else None),
                *(row_ranges, "i"),
                *(col_ranges, "j"),
                shape=(nrows, ncols),
                dtype=np_idtype_,
                meta=np.array([]),
            )
            meta = InnerMatrix(gb.Matrix.new(vdtype, nrows=nrows, ncols=ncols))
            delayed = da.core.blockwise(
                *(_from_values2D, "ij"),
                *(values if isinstance(values, Number) else None, None),
                *(fragments, "ijk"),
                *(row_ranges, "i"),
                *(col_ranges, "j"),
                concatenate=False,
                dup_op=dup_op,
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

    def build(
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
        if not clear and self._nvals != 0:
            raise gb.exceptions.OutputNotEmpty()

        # TODO: delayed nrows/ncols
        nrows = nrows or self._nrows
        ncols = ncols or self._ncols
        meta = self._meta

        # check for any DOnions:
        args = pack_args(self, rows, columns, values)
        kwargs = pack_kwargs(
            dup_op=dup_op, clear=clear, nrows=nrows, ncols=ncols, chunks=chunks, in_dOnion=True
        )
        if any_dOnions(*args, **kwargs):
            # dive into DOnion(s):
            out_donion = DOnion.multi_access(meta, Matrix.build, *args, **kwargs)
            self.__init__(out_donion, meta=meta)
            return

        # no DOnions
        if clear:
            self.clear()

        self.resize(nrows, ncols)

        if chunks is not None:
            self.rechunk(inplace=True, chunks=chunks)

        x = self._optional_dup()
        if type(rows) in {tuple, list, np.ndarray}:
            if np.max(rows) >= self._nrows:
                raise gb.exceptions.IndexOutOfBound
            rows = da.core.from_array(np.array(rows), name="rows-" + tokenize(rows))

        if type(columns) in {tuple, list, np.ndarray}:
            if np.max(columns) >= self._ncols:
                raise gb.exceptions.IndexOutOfBound
            columns = da.core.from_array(np.array(columns), name="columns-" + tokenize(columns))

        if type(values) in {tuple, list, np.ndarray}:
            values = da.core.from_array(np.array(values), name="values-" + tokenize(values))

        if type(values) is da.Array and (rows.size != columns.size or columns.size != values.size):
            raise ValueError(
                "`rows` and `columns` and `values` lengths must match: "
                f"{rows.size}, {columns.size}, {values.size}"
            )
        elif rows.size != columns.size:
            raise ValueError(
                f"`rows` and `columns` lengths must match: {rows.size}, {columns.size}"
            )
        elif values is None:
            raise EmptyObject()

        idtype = gb.Matrix.new(rows.dtype).dtype
        np_idtype_ = np_dtype(idtype)
        vdtype = (
            lookup_dtype(type(values))
            if isinstance(values, Number)
            else gb.Matrix.new(values.dtype).dtype
        )
        np_vdtype_ = np_dtype(vdtype)

        rname = "-row-ranges" + tokenize(x, x.chunks[0])
        cname = "-col-ranges" + tokenize(x, x.chunks[1])
        row_ranges = build_chunk_ranges_dask_array(x, 0, rname)
        col_ranges = build_chunk_ranges_dask_array(x, 1, cname)
        fragments = da.core.blockwise(
            *(_pick2D, "ijk"),
            *(rows, "k"),
            *(columns, "k"),
            *(values, None if isinstance(values, Number) else "k"),
            *(row_ranges, "i"),
            *(col_ranges, "j"),
            shape=(nrows, ncols),
            dtype=np_idtype_,
            meta=np.array([]),
        )
        meta = InnerMatrix(gb.Matrix.new(vdtype))
        delayed = da.core.blockwise(
            *(_build_2D_chunk, "ij"),
            *(x, "ij"),
            *(row_ranges, "i"),
            *(col_ranges, "j"),
            *(fragments, "ijk"),
            values=values if isinstance(values, Number) else None,
            dup_op=dup_op,
            clear=False,
            concatenate=False,
            dtype=np_vdtype_,
            meta=meta,
        )
        if in_dOnion:
            return Matrix(delayed)
        self.__init__(delayed)

    @classmethod
    def new(cls, dtype, nrows=0, ncols=0, *, chunks="auto", name=None):
        if any_dOnions(nrows, ncols):
            meta = gb.Matrix.new(dtype)
            donion = DOnion.multi_access(
                meta, cls.new, dtype, nrows=nrows, ncols=ncols, chunks=chunks, name=name
            )
            return Matrix(donion, meta=meta)

        if type(nrows) is Box:
            nrows = nrows.content

        if type(ncols) is Box:
            ncols = ncols.content

        dtype = dtype.lower() if isinstance(dtype, str) else dtype
        if nrows == 0 and ncols == 0:
            matrix = gb.Matrix.new(dtype, nrows, ncols)
            return cls.from_delayed(
                delayed(matrix), matrix.dtype, matrix.nrows, matrix.ncols, nvals=0, name=name
            )
        else:
            chunks = da.core.normalize_chunks(chunks, (nrows, ncols), dtype=int)
            name_ = name
            name = str(name) if name else ""
            rname = name + "-row-ranges" + tokenize(cls, chunks[0])
            cname = name + "-col-ranges" + tokenize(cls, chunks[1])
            row_ranges = build_ranges_dask_array_from_chunks(chunks[0], rname)
            col_ranges = build_ranges_dask_array_from_chunks(chunks[1], cname)

            meta = InnerMatrix(gb.Matrix.new(dtype))
            try:
                np_dtype_ = np_dtype(dtype)
            except AttributeError:
                np_dtype_ = np.dtype(dtype)

            _delayed = da.core.blockwise(
                *(_new_Matrix_chunk, "ij"),
                *(row_ranges, "i"),
                *(col_ranges, "j"),
                gb_dtype=meta.dtype,
                dtype=np_dtype_,
                meta=meta,
                name=name_,
            )
            return Matrix(_delayed, nvals=0)

    def __init__(self, delayed, meta=None, nvals=None):
        # We recommend always using __init__() to set the attribute
        # ._delayed indirectly rather than directly.
        # Note: `nvals` is provided here as a parameter mainly for
        # optimization purposes.  A value for `nvals` may be given
        # if it is already known  at the time of initialization of
        # this Matrix,  otherwise its value should be left as None
        # (the default)
        assert type(delayed) in {da.Array, DOnion}
        self._delayed = delayed
        if type(delayed) is da.Array:
            assert delayed.ndim == 2
            if meta is None:
                meta = gb.Matrix.new(delayed.dtype, *delayed.shape)
        else:
            if meta is None:
                meta = gb.Matrix.new(delayed.dtype)
        self._meta = meta
        self.dtype = meta.dtype
        self._nrows = self.nrows
        self._ncols = self.ncols
        self._nvals = nvals
        # Add ss extension methods
        self.ss = ss(self)

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
        if self.is_dOnion:
            return DOnion.multi_access(self._meta.nrows, getattr, self, "nrows")
        return self._meta.nrows

    @property
    def ncols(self):
        if self.is_dOnion:
            return DOnion.multi_access(self._meta.ncols, getattr, self, "ncols")
        return self._meta.ncols

    @property
    def shape(self):
        if self.is_dOnion:
            return (self.nrows, self.ncols)
            # return DOnion.multi_access(self._meta.shape, getattr, self, "shape")
        return self._meta.shape

    def resize(self, nrows, ncols, inplace=True, chunks="auto"):
        if self.is_dOnion:
            donion = self._delayed.getattr(
                self._meta, "resize", nrows, ncols, inplace=False, chunks=chunks
            )
            if inplace:
                self.__init__(donion, meta=self._meta)
                return
            else:
                return Matrix(donion, meta=self._meta)

        chunks = da.core.normalize_chunks(chunks, (nrows, ncols), dtype=np.int64)
        output_row_ranges = build_ranges_dask_array_from_chunks(chunks[0], "output_row_ranges-")
        output_col_ranges = build_ranges_dask_array_from_chunks(chunks[1], "output_col_ranges-")

        x = self._optional_dup()
        _meta = x._meta
        dtype_ = np_dtype(self.dtype)
        row_ranges = build_chunk_ranges_dask_array(x, 0, "row_ranges-")
        col_ranges = build_chunk_ranges_dask_array(x, 1, "col_ranges-")
        x = da.core.blockwise(
            *(_resize, "ijkl"),
            *(output_row_ranges, "k"),
            *(output_col_ranges, "l"),
            *(x, "ij"),
            *(row_ranges, "i"),
            *(col_ranges, "j"),
            old_shape=self.shape,
            new_shape=(nrows, ncols),
            dtype=dtype_,
            meta=np.array([[[[]]]]),
        )
        x = da.core.blockwise(
            *(_identity, "kl"),
            *(x, "ijkl"),
            concatenate=True,
            dtype=dtype_,
            meta=_meta,
        )

        if nrows >= self.nrows and ncols >= self.ncols:
            nvals = self.nvals
        else:
            nvals = None

        if inplace:
            self.__init__(x, nvals=nvals)
        else:
            return Matrix(x, nvals=nvals)

    def rechunk(self, inplace=False, chunks="auto"):
        if self.is_dOnion:
            meta = self._meta
            donion = self._delayed.getattr(meta, "rechunk", inplace=False, chunks=chunks)
            if inplace:
                self.__init__(donion, meta=meta)
                return
            else:
                return Matrix(donion, meta=meta)

        delayed = self._delayed.rechunk(chunks=chunks)
        if inplace:
            self._delayed = delayed
            return
        else:
            return Matrix(delayed, meta=self._meta, nvals=self._nvals)
        # chunks = da.core.normalize_chunks(chunks, self.shape, dtype=np.int64)
        # if inplace:
        #     self.resize(*self.shape, chunks=chunks)
        #     return
        # else:
        #     return self.resize(*self.shape, chunks=chunks, inplace=False)

    def _diag(self, k=0, dtype=None, chunks="auto"):
        kdiag_row_start = max(0, -k)
        kdiag_col_start = max(0, k)
        kdiag_row_stop = min(self.nrows, self.ncols - k)
        len_kdiag = kdiag_row_stop - kdiag_row_start

        gb_dtype = self.dtype if dtype is None else lookup_dtype(dtype)
        meta = wrap_inner(gb.Vector.new(gb_dtype))
        if len_kdiag <= 0:
            return get_return_type(meta).new(gb_dtype)

        A = self._delayed
        name = "diag-" + tokenize(A)

        row_stops_ = np.cumsum(A.chunks[0])
        row_starts = np.roll(row_stops_, 1)
        row_starts[0] = 0

        col_stops_ = np.cumsum(A.chunks[1])
        col_starts = np.roll(col_stops_, 1)
        col_starts[0] = 0

        row_blockid = np.arange(A.numblocks[0])
        col_blockid = np.arange(A.numblocks[1])

        # locate first chunk containing diagonal:
        row_filter = (row_starts <= kdiag_row_start) & (kdiag_row_start < row_stops_)
        col_filter = (col_starts <= kdiag_col_start) & (kdiag_col_start < col_stops_)
        (R,) = row_blockid[row_filter]
        (C,) = col_blockid[col_filter]

        # follow k-diagonal through chunks while constructing dask graph:
        # equation of diagonal: i = j - k
        dsk = dict()
        i = 0
        out_chunks = ()
        while kdiag_row_start < A.shape[0] and kdiag_col_start < A.shape[1]:
            # localize block info:
            nrows, ncols = A.chunks[0][R], A.chunks[1][C]
            kdiag_row_start -= row_starts[R]
            kdiag_col_start -= col_starts[C]
            k = -kdiag_row_start if kdiag_row_start > 0 else kdiag_col_start
            kdiag_row_end = min(nrows, ncols - k)
            kdiag_len = kdiag_row_end - kdiag_row_start
            # increment dask graph:
            dsk[(name, i)] = (_chunk_diag_v2, (A.name, R, C), k)
            out_chunks += (kdiag_len,)
            # prepare for next iteration:
            i += 1
            kdiag_row_start = kdiag_row_end + row_starts[R]
            kdiag_col_start = min(ncols, nrows + k) + col_starts[C]
            R = R + 1 if kdiag_row_start == row_stops_[R] else R
            C = C + 1 if kdiag_col_start == col_stops_[C] else C

        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[A])
        out = da.core.Array(graph, name, (out_chunks,), meta=meta)

        nvals = 0 if self._nvals == 0 else None
        out = get_return_type(meta.value)(out, nvals=nvals)
        return out.rechunk(chunks=chunks)

    def _diag_old(self, k=0, dtype=None, chunks="auto"):
        nrows, ncols = self.nrows, self.ncols
        # equation of diagonal: i = j - k
        kdiag_row_start = max(0, -k)
        kdiag_row_stop = min(nrows, ncols - k)
        len_kdiag = kdiag_row_stop - kdiag_row_start

        gb_dtype = self.dtype if dtype is None else lookup_dtype(dtype)
        meta = gb.Vector.new(gb_dtype)
        if len_kdiag <= 0:
            return get_return_type(meta).new(gb_dtype)

        chunks = da.core.normalize_chunks(chunks, (len_kdiag,), dtype=np.int64)
        output_indx_ranges = build_ranges_dask_array_from_chunks(chunks[0], "output_indx_ranges-")

        x = self._delayed
        row_ranges = build_chunk_ranges_dask_array(x, 0, "row_ranges-")
        col_ranges = build_chunk_ranges_dask_array(x, 1, "col_ranges-")

        dtype_ = np_dtype(gb_dtype)
        fragments = da.core.blockwise(
            *(_chunk_diag, "ijk"),
            *(output_indx_ranges, "k"),
            *(x, "ij"),
            *(row_ranges, "i"),
            *(col_ranges, "j"),
            k=k,
            kdiag_row_start=kdiag_row_start,
            gb_dtype=gb_dtype,
            dtype=dtype_,
            meta=np.array([[[]]]),
        )
        fragments = da.reduction(
            fragments, _identity, _identity, axis=0, dtype=dtype_, meta=np.array([[]])
        )
        delayed = da.reduction(
            fragments, _identity, _identity, axis=0, dtype=dtype_, meta=wrap_inner(meta)
        )
        nvals = 0 if self._nvals == 0 else None
        return get_return_type(meta)(delayed, nvals=nvals)

    def __getitem__(self, index):
        return AmbiguousAssignOrExtract(self, index)

    def __delitem__(self, keys, in_dOnion=False):
        if is_DOnion(self._delayed):
            good_keys = [x for x in keys if isinstance(x, Integral)]
            if len(good_keys) != 2:
                raise TypeError("Remove Element only supports scalars.")

            donion = self._delayed.getattr(self._meta, "__delitem__", keys, in_dOnion=True)
            self.__init__(donion, meta=self._meta)
            return

        del Updater(self)[keys]
        if in_dOnion:
            return self

    def __setitem__(self, index, delayed, in_dOnion=False):
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
        gb_types = (gb.Matrix, gb.matrix.TransposedMatrix)
        other = self._expect_type(
            other, (Matrix, TransposedMatrix) + gb_types, within="ewise_add", argname="other"
        )

        try:
            meta = self._meta.ewise_add(other._meta, op=op, require_monoid=require_monoid)
        except DimensionMismatch:
            if any_dOnions(self, other):
                meta = self._meta.ewise_add(self._meta, op=op, require_monoid=require_monoid)
            else:
                raise

        return MatrixExpression(self, "ewise_add", other, op, require_monoid=require_monoid, meta=meta)

    def ewise_mult(self, other, op=binary.times):
        gb_types = (gb.Matrix, gb.matrix.TransposedMatrix)
        other = self._expect_type(
            other, (Matrix, TransposedMatrix) + gb_types, within="ewise_mult", argname="other"
        )

        try:
            meta = self._meta.ewise_mult(other._meta, op=op)
        except DimensionMismatch:
            if any_dOnions(self, other):
                meta = self._meta.ewise_mult(self._meta, op=op)
            else:
                raise

        return MatrixExpression(self, "ewise_mult", other, op, meta=meta)

    def mxv(self, other, op=semiring.plus_times):
        from .vector import Vector, VectorExpression

        other = self._expect_type(
            other, (Vector, gb.Vector), within="mxv", argname="other"
        )

        try:
            meta = self._meta.mxv(other._meta, op=op)
        except DimensionMismatch:
            if any_dOnions(self, other):
                other_meta = gb.Vector.new(dtype=other._meta.dtype, size=self._meta.ncols)
                meta = self._meta.mxv(other_meta, op=op)
            else:
                raise

        return VectorExpression(self, "mxv", other, op, meta=meta, size=self.nrows)

    def mxm(self, other, op=semiring.plus_times):
        gb_types = (gb.Matrix, gb.matrix.TransposedMatrix)
        other = self._expect_type(
            other, (Matrix, TransposedMatrix) + gb_types, within="mxm", argname="other"
        )

        try:
            meta = self._meta.mxm(other._meta, op=op)
        except DimensionMismatch:
            if any_dOnions(self, other):
                other_meta = gb.Matrix.new(
                    dtype=other._meta.dtype, nrows=self._meta.ncols, ncols=other._meta.ncols
                )
                meta = self._meta.mxm(other_meta, op=op)
            else:
                raise

        return MatrixExpression(self, "mxm", other, op, meta=meta, nrows=self.nrows, ncols=other.ncols)

    def kronecker(self, other, op=binary.times):
        gb_types = (gb.Matrix, gb.matrix.TransposedMatrix)
        other = self._expect_type(
            other, (Matrix, TransposedMatrix) + gb_types, within="kronecker", argname="other"
        )
        meta = self._meta.kronecker(other._meta, op=op)
        return MatrixExpression(self, "kronecker", other, op, meta=meta)

    def apply(self, op, right=None, *, left=None):
        from .scalar import Scalar

        left_meta = left
        right_meta = right

        if type(left) is Scalar:
            left_meta = left.dtype.np_type(0)
        if type(right) is Scalar:
            right_meta = right.dtype.np_type(0)

        if self._meta.shape == (0,) * self.ndim:
            self._meta.resize(*((1,) * self.ndim))
        meta = self._meta.apply(op=op, left=left_meta, right=right_meta)
        return MatrixExpression(self, "apply", op, right, meta=meta, left=left)

    def reduce_rowwise(self, op=monoid.plus):
        from .vector import VectorExpression

        meta = self._meta.reduce_rowwise(op)
        return VectorExpression(self, "reduce_rowwise", op, meta=meta, size=self.nrows)

    def reduce_columnwise(self, op=monoid.plus):
        from .vector import VectorExpression

        meta = self._meta.reduce_columnwise(op)
        return VectorExpression(self, "reduce_columnwise", op, meta=meta, size=self.ncols)

    def reduce_scalar(self, op=monoid.plus):
        from .scalar import ScalarExpression

        meta = self._meta.reduce_scalar(op)
        return ScalarExpression(self, "reduce_scalar", op, meta=meta)

    def to_values(self, dtype=None, chunks="auto"):
        dtype = lookup_dtype(self.dtype if dtype is None else dtype)
        meta_i, _, meta_v = self._meta.to_values(dtype)

        if self.is_dOnion:
            meta = np.array([])
            result = DOnion.multi_access(
                meta, self.__class__.to_values, self, dtype=dtype, chunks=chunks
            )
            rows = DOnion.multi_access(meta_i, tuple.__getitem__, result, 0)
            columns = DOnion.multi_access(meta_i, tuple.__getitem__, result, 1)
            values = DOnion.multi_access(meta_v, tuple.__getitem__, result, 2)
            return rows, columns, values

        # first find the number of values in each chunk and return
        # them as a 2D numpy array whose shape is equal to x.numblocks
        x = self._delayed
        nvals_2D = da.core.blockwise(
            *(_nvals_in_chunk, "ij"),
            *(x, "ij"),
            adjust_chunks={"i": 1, "j": 1},
            dtype=np.int64,
            meta=np.array([[]]),
        )

        # use the above array to determine the output tuples' array
        # bounds (`starts` and `stops_`) for each chunk of this
        # Matrix (self)
        stops_ = da.cumsum(nvals_2D)  # BEWARE: this function rechunks!
        starts = da.roll(stops_, 1)
        starts = starts.copy() if starts.size == 1 else starts  # bug!!
        starts[0] = 0
        nnz = stops_[-1]
        starts = starts.reshape(nvals_2D.shape).rechunk(1)
        stops_ = stops_.reshape(nvals_2D.shape).rechunk(1)

        def _to_values(x, starts, stops_, dtype, chunks, nnz):
            # the following changes the `.chunks` attribute of `starts` and `stops_` so that
            # `blockwise()` can align them with `x`
            starts = da.core.Array(starts.dask, starts.name, x.chunks, starts.dtype, meta=x._meta)
            stops_ = da.core.Array(stops_.dask, stops_.name, x.chunks, stops_.dtype, meta=x._meta)

            chunks = da.core.normalize_chunks(chunks, (nnz,), dtype=np.int64)
            output_ranges = build_ranges_dask_array_from_chunks(chunks[0], "output_ranges-")

            gb_dtype = lookup_dtype(dtype)
            dtype_ = np_dtype(gb_dtype)
            # Compute row/col offsets as dask arrays that can align with this
            # Matrix's (self's) chunks to convert chunk row/col indices to
            # full dask-grblas Matrix indices.
            row_offsets = build_chunk_offsets_dask_array(x, 0, "row_offset-")
            col_offsets = build_chunk_offsets_dask_array(x, 1, "col_offset-")
            x = da.core.blockwise(
                *(MatrixTupleExtractor, "ijk"),
                *(output_ranges, "k"),
                *(x, "ij"),
                *(row_offsets, "i"),
                *(col_offsets, "j"),
                *(starts, "ij"),
                *(stops_, "ij"),
                gb_dtype=dtype,
                dtype=dtype_,
                meta=np.array([[[]]]),
            )
            x = da.reduction(
                x, _identity, _flatten, axis=1, concatenate=False, dtype=dtype_, meta=np.array([[]])
            )
            return da.reduction(
                x, _identity, _flatten, axis=0, concatenate=False, dtype=dtype_, meta=np.array([])
            )

        # since the size of the output (rows, columns, values) depends on nnz, a delayed quantity,
        # we need to return the output as DOnions (twice-delayed dask-arrays)
        meta = np.array([])
        rcv_donion = DOnion.sprout(nnz, meta, _to_values, x, starts, stops_, dtype, chunks)

        dtype_i = np_dtype(lookup_dtype(meta_i.dtype))
        rows = rcv_donion.deep_extract(meta_i, da.map_blocks, _get_rows, dtype=dtype_i, meta=meta_i)
        cols = rcv_donion.deep_extract(meta_i, da.map_blocks, _get_cols, dtype=dtype_i, meta=meta_i)
        dtype_v = np_dtype(lookup_dtype(meta_v.dtype))
        vals = rcv_donion.deep_extract(meta_v, da.map_blocks, _get_vals, dtype=dtype_v, meta=meta_v)
        return rows, cols, vals

    def isequal(self, other, *, check_dtype=False):
        gb_types = (gb.Matrix, gb.matrix.TransposedMatrix)
        other = self._expect_type(
            other, (Matrix, TransposedMatrix) + gb_types, within="isequal", argname="other"
        )
        return super().isequal(other, check_dtype=check_dtype)

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        gb_types = (gb.Matrix, gb.matrix.TransposedMatrix)
        other = self._expect_type(
            other, (Matrix, TransposedMatrix) + gb_types, within="isclose", argname="other"
        )
        return super().isclose(other, rel_tol=rel_tol, abs_tol=abs_tol, check_dtype=check_dtype)

    def _delete_element(self, resolved_indexes):
        row = resolved_indexes.indices[0]
        col = resolved_indexes.indices[1]
        delayed = self._optional_dup()
        row_ranges = build_chunk_ranges_dask_array(
            delayed, 0, "index-ranges-" + tokenize(delayed, 0)
        )
        col_ranges = build_chunk_ranges_dask_array(
            delayed, 1, "index-ranges-" + tokenize(delayed, 1)
        )
        deleted = da.core.blockwise(
            *(_delitem_in_chunk, "ij"),
            *(delayed, "ij"),
            *(row_ranges, "i"),
            *(col_ranges, "j"),
            *(row.index, None),
            *(col.index, None),
            dtype=delayed.dtype,
            meta=delayed._meta,
        )
        self.__init__(deleted)


Matrix.ss = gb.utils.class_property(Matrix.ss, ss)


class TransposedMatrix:
    ndim = 2
    _is_scalar = False
    _is_transposed = True

    __and__ = gb.matrix.TransposedMatrix.__and__
    __bool__ = gb.matrix.TransposedMatrix.__bool__
    __or__ = gb.matrix.TransposedMatrix.__or__

    __abs__ = gb.matrix.TransposedMatrix.__abs__
    __add__ = gb.matrix.TransposedMatrix.__add__
    __divmod__ = gb.matrix.TransposedMatrix.__divmod__
    __eq__ = gb.matrix.TransposedMatrix.__eq__
    __floordiv__ = gb.matrix.TransposedMatrix.__floordiv__
    __ge__ = gb.matrix.TransposedMatrix.__ge__
    __gt__ = gb.matrix.TransposedMatrix.__gt__
    __invert__ = gb.matrix.TransposedMatrix.__invert__
    __le__ = gb.matrix.TransposedMatrix.__le__
    __lt__ = gb.matrix.TransposedMatrix.__lt__
    __mod__ = gb.matrix.TransposedMatrix.__mod__
    __mul__ = gb.matrix.TransposedMatrix.__mul__
    __ne__ = gb.matrix.TransposedMatrix.__ne__
    __neg__ = gb.matrix.TransposedMatrix.__neg__
    __pow__ = gb.matrix.TransposedMatrix.__pow__
    __radd__ = gb.matrix.TransposedMatrix.__radd__
    __rdivmod__ = gb.matrix.TransposedMatrix.__rdivmod__
    __rfloordiv__ = gb.matrix.TransposedMatrix.__rfloordiv__
    __rmod__ = gb.matrix.TransposedMatrix.__rmod__
    __rmul__ = gb.matrix.TransposedMatrix.__rmul__
    __rpow__ = gb.matrix.TransposedMatrix.__rpow__
    __rsub__ = gb.matrix.TransposedMatrix.__rsub__
    __rtruediv__ = gb.matrix.TransposedMatrix.__rtruediv__
    __rxor__ = gb.matrix.TransposedMatrix.__rxor__
    __sub__ = gb.matrix.TransposedMatrix.__sub__
    __truediv__ = gb.matrix.TransposedMatrix.__truediv__
    __xor__ = gb.matrix.TransposedMatrix.__xor__

    def __init__(self, matrix, meta=None):
        assert type(matrix) is Matrix
        self._matrix = matrix
        self._meta = matrix._meta.T if meta is None else meta

        # Aggregator-specific requirements:
        self._nrows = self._meta.nrows
        self._ncols = self._meta.ncols

    @property
    def is_dOnion(self):
        return is_DOnion(self._matrix._delayed)

    @property
    def dOnion_if(self):
        return self._matrix._delayed if self.is_dOnion else self

    def dup(self, dtype=None, *, mask=None, name=None):
        return self.new(dtype=dtype, mask=mask)

    def new(self, *, dtype=None, mask=None):
        if any_dOnions(self, mask):
            donion = DOnion.multi_access(
                self._meta.new(dtype), self.__class__.new, self, dtype=dtype, mask=mask
            )
            return Matrix(donion)

        gb_dtype = self._matrix.dtype if dtype is None else lookup_dtype(dtype)
        dtype = np_dtype(gb_dtype)

        delayed = self._matrix._delayed
        if mask is None:
            mask_type = None
            mask_ind = None
        else:
            mask_type = get_grblas_type(mask)
            mask = mask.mask._delayed
            mask_ind = "ji"
        delayed = da.core.blockwise(
            *(_transpose, "ji"),
            *(delayed, "ij"),
            *(mask, mask_ind),
            mask_type=mask_type,
            gb_dtype=gb_dtype,
            dtype=dtype,
            meta=delayed._meta,
        )
        return Matrix(delayed)

    @property
    def T(self):
        return self._matrix

    @property
    def dtype(self):
        return self._meta.dtype

    def to_values(self, dtype=None, chunks="auto"):
        rows, cols, vals = self._matrix.to_values(dtype=dtype, chunks=chunks)
        return cols, rows, vals

    # Properties
    def __getitem__(self, index):
        return AmbiguousAssignOrExtract(self, index)

    def isequal(self, other, *, check_dtype=False):
        other = self._expect_type(
            other, (Matrix, TransposedMatrix), within="isequal", argname="other"
        )
        return BaseType.isequal(self, other, check_dtype=check_dtype)

    def isclose(self, other, *, rel_tol=1e-7, abs_tol=0.0, check_dtype=False):
        other = self._expect_type(
            other, (Matrix, TransposedMatrix), within="isclose", argname="other"
        )
        return BaseType.isclose(
            self, other, rel_tol=rel_tol, abs_tol=abs_tol, check_dtype=check_dtype
        )

    # Delayed methods
    __contains__ = Matrix.__contains__
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
    nrows = Matrix.nrows
    ncols = Matrix.ncols
    shape = Matrix.shape
    nvals = Matrix.nvals
    _expect_type = Matrix._expect_type
    __array__ = Matrix.__array__
    name = Matrix.name


class MatrixExpression(GbDelayed):
    __slots__ = ()
    output_type = gb.Matrix
    ndim = 2
    _is_scalar = False

    # automethods:
    __and__ = gb.matrix.MatrixExpression.__and__
    __bool__ = gb.matrix.MatrixExpression.__bool__
    __or__ = gb.matrix.MatrixExpression.__or__
    _get_value = _automethods._get_value
    S = gb.matrix.MatrixExpression.S
    T = gb.matrix.MatrixExpression.T
    V = gb.matrix.MatrixExpression.V
    apply = gb.matrix.MatrixExpression.apply
    ewise_add = gb.matrix.MatrixExpression.ewise_add
    ewise_mult = gb.matrix.MatrixExpression.ewise_mult
    isclose = gb.matrix.MatrixExpression.isclose
    isequal = gb.matrix.MatrixExpression.isequal
    kronecker = gb.matrix.MatrixExpression.kronecker
    mxm = gb.matrix.MatrixExpression.mxm
    mxv = gb.matrix.MatrixExpression.mxv
    ncols = gb.matrix.MatrixExpression.ncols
    nrows = gb.matrix.MatrixExpression.nrows
    nvals = gb.matrix.MatrixExpression.nvals
    reduce_rowwise = gb.matrix.MatrixExpression.reduce_rowwise
    reduce_columnwise = gb.matrix.MatrixExpression.reduce_columnwise
    reduce_scalar = gb.matrix.MatrixExpression.reduce_scalar
    shape = gb.matrix.MatrixExpression.shape
    nvals = gb.matrix.MatrixExpression.nvals

    # infix sugar:
    __abs__ = gb.matrix.MatrixExpression.__abs__
    __add__ = gb.matrix.MatrixExpression.__add__
    __divmod__ = gb.matrix.MatrixExpression.__divmod__
    __eq__ = gb.matrix.MatrixExpression.__eq__
    __floordiv__ = gb.matrix.MatrixExpression.__floordiv__
    __ge__ = gb.matrix.MatrixExpression.__ge__
    __gt__ = gb.matrix.MatrixExpression.__gt__
    __invert__ = gb.matrix.MatrixExpression.__invert__
    __le__ = gb.matrix.MatrixExpression.__le__
    __lt__ = gb.matrix.MatrixExpression.__lt__
    __mod__ = gb.matrix.MatrixExpression.__mod__
    __mul__ = gb.matrix.MatrixExpression.__mul__
    __ne__ = gb.matrix.MatrixExpression.__ne__
    __neg__ = gb.matrix.MatrixExpression.__neg__
    __pow__ = gb.matrix.MatrixExpression.__pow__
    __radd__ = gb.matrix.MatrixExpression.__radd__
    __rdivmod__ = gb.matrix.MatrixExpression.__rdivmod__
    __rfloordiv__ = gb.matrix.MatrixExpression.__rfloordiv__
    __rmod__ = gb.matrix.MatrixExpression.__rmod__
    __rmul__ = gb.matrix.MatrixExpression.__rmul__
    __rpow__ = gb.matrix.MatrixExpression.__rpow__
    __rsub__ = gb.matrix.MatrixExpression.__rsub__
    __rtruediv__ = gb.matrix.MatrixExpression.__rtruediv__
    __rxor__ = gb.matrix.MatrixExpression.__rxor__
    __sub__ = gb.matrix.MatrixExpression.__sub__
    __truediv__ = gb.matrix.MatrixExpression.__truediv__
    __xor__ = gb.matrix.MatrixExpression.__xor__

    # bad sugar:
    __itruediv__ = gb.matrix.MatrixExpression.__itruediv__
    __imul__ = gb.matrix.MatrixExpression.__imul__
    __imatmul__ = gb.matrix.MatrixExpression.__imatmul__
    __iadd__ = gb.matrix.MatrixExpression.__iadd__
    __iand__ = gb.matrix.MatrixExpression.__iand__
    __ipow__ = gb.matrix.MatrixExpression.__ipow__
    __imod__ = gb.matrix.MatrixExpression.__imod__
    __isub__ = gb.matrix.MatrixExpression.__isub__
    __ixor__ = gb.matrix.MatrixExpression.__ixor__
    __ifloordiv__ = gb.matrix.MatrixExpression.__ifloordiv__
    __ior__ = gb.matrix.MatrixExpression.__ior__

    def __init__(
        self,
        parent,
        method_name,
        *args,
        meta=None,
        ncols=None,
        nrows=None,
        **kwargs,
    ):
        super().__init__(
            parent,
            method_name,
            *args,
            meta=meta,
            **kwargs,
        )
        if ncols is None:
            ncols = self.parent._ncols
        if nrows is None:
            nrows = self.parent._nrows
        self._ncols = ncols
        self._nrows = nrows

    # def __getattr__(self, item):
    #     return getattr(gb.matrix.MatrixExpression, item)

    # def construct_output(self, dtype=None, *, name=None):
    #     if dtype is None:
    #         dtype = self.dtype
    #     nrows = 0 if self._nrows.is_dOnion else self._nrows
    #     ncols = 0 if self._ncols.is_dOnion else self._ncols
    #     return Matrix.new(dtype, nrows, ncols, name=name)


def _chunk_diag_v2(inner_matrix, k):
    return wrap_inner(gb.ss.diag(inner_matrix.value, k))


def _chunk_diag(
    output_range,
    inner_matrix,
    row_range,
    col_range,
    k,
    kdiag_row_start,
    gb_dtype,
):
    """
    Return a new vector chunk with size determined by various conditions.

    The returned vector is either of zero-length or it is a piece of the
    k-diagonal in inner_matrix
    """
    # There's perhaps a one-line formula summarizing this entire function
    output_range = output_range[0]
    rows = row_range[0]
    cols = col_range[0]
    matrix = inner_matrix.value
    # REFERENCE POINT: global matrix row 0 col 0
    # Find intersection of k-diagonal with chunk column bounds:
    # use diagonal equation: i = j - k  (i: row index; j: column index)
    j = cols.start
    chunk_kdiag_row_start = j - k
    j = cols.stop
    chunk_kdiag_row_stop = j - k

    # intersect chunk row range with k-diagonal within chunk column bounds
    if rows.start < chunk_kdiag_row_stop and chunk_kdiag_row_start < rows.stop:
        chunk_kdiag_row_start = max(chunk_kdiag_row_start, rows.start)
        chunk_kdiag_row_stop = min(chunk_kdiag_row_stop, rows.stop)

        # CHANGE REFERENCE POINT: to global k-diagonal start
        # NOTE: here we choose to project the k-diagonal onto axis 0 in
        #       order to output it as a vector
        vector_kdiag_start = chunk_kdiag_row_start - kdiag_row_start
        vector_kdiag_stop = chunk_kdiag_row_stop - kdiag_row_start

        # intersect output-range with row-range of k-diagonal within chunk
        if output_range.start < vector_kdiag_stop and vector_kdiag_start < output_range.stop:
            vector_kdiag_start = max(output_range.start, vector_kdiag_start)
            vector_kdiag_stop = min(output_range.stop, vector_kdiag_stop)
            # CHANGE REFERENCE POINT: to global matrix row 0 col 0
            chunk_kdiag_row_start = vector_kdiag_start + kdiag_row_start
            chunk_kdiag_row_stop = vector_kdiag_stop + kdiag_row_start
            # compute column-range of chunk k-diagonal
            # use diagonal equation:  j = i + k
            chunk_kdiag_col_start = chunk_kdiag_row_start + k
            chunk_kdiag_col_stop = chunk_kdiag_row_stop + k
            # CHANGE REFERENCE POINT: to chunk matrix row 0 col 0
            chunk_kdiag_row_start -= rows.start
            chunk_kdiag_row_stop -= rows.start
            chunk_kdiag_col_start -= cols.start
            chunk_kdiag_col_stop -= cols.start
            # extract square sub-matrix containing k-diagonal
            matrix = matrix[
                chunk_kdiag_row_start:chunk_kdiag_row_stop,
                chunk_kdiag_col_start:chunk_kdiag_col_stop,
            ]
            # extract its diagonal
            vector = gb.ss.diag(matrix.new(), k=0, dtype=gb_dtype)
            return wrap_inner(vector)
    return wrap_inner(gb.Vector.new(gb_dtype))


def _resize(
    output_row_range, output_col_range, inner_matrix, row_range, col_range, old_shape, new_shape
):
    """
    Returns, in general, an extracted part of chunk `inner_matrix` that
    lies entirely within the bounds specified by `output_row_range` and
    `output_col_range`.

    If the chunk does not lie within those bounds but lies BEYOND them
    then then a Matrix with one or both of its dimensions equal to 0 is
    returned depending on which dimension intersected with the given bounds.

    If the chunk INTERSECTS with the bounds and is the LAST in its
    block-row (-column) then its extract is expanded so that its last row
    (col) aligns with the end of `output_row_range` (`output_col_range`).

    If the chunk is the LAST in its block-row (-column) AND does not lie
    within those bounds but lies BEFORE them, then then an empty Matrix
    with one or both of its dimensions equal to the lengths of
    `output_row_range` or `output_col_range` is returned.
    """
    output_range = (output_row_range[0], output_col_range[0])
    index_range = (row_range[0], col_range[0])
    old_meta = ()
    new_meta = ()
    for axis in (0, 1):
        if (
            output_range[axis].start < index_range[axis].stop
            and index_range[axis].start < output_range[axis].stop
        ):
            start = max(output_range[axis].start, index_range[axis].start)
            stop = min(output_range[axis].stop, index_range[axis].stop)
            start = start - index_range[axis].start
            stop = stop - index_range[axis].start
            if (
                index_range[axis].stop == old_shape[axis]
                and new_shape[axis] > old_shape[axis]
                and stop < output_range[axis].stop - index_range[axis].start
            ):
                old_meta += (slice(start, stop),)
                new_meta += (output_range[axis].stop - index_range[axis].start - start,)
            else:
                old_meta += (slice(start, stop),)
                new_meta += (stop - start,)
        elif (
            index_range[axis].stop == old_shape[axis]
            and old_shape[axis] <= output_range[axis].start
        ):
            old_meta += (slice(0, 0),)
            new_meta += (output_range[axis].stop - output_range[axis].start,)
        else:
            old_meta += (slice(0, 0),)
            new_meta += (0,)
    # -------------------end of for loop -----------------

    new_mat = inner_matrix.value[old_meta].new()
    new_mat.resize(*new_meta)
    return InnerMatrix(new_mat)


def _transpose(chunk, mask, mask_type, gb_dtype):
    """
    Transposes the chunk
    """
    mask = None if mask is None else mask_type(mask.value)
    return InnerMatrix(chunk.value.T.new(mask=mask, dtype=gb_dtype))


def _delitem_in_chunk(inner_mat, row_range, col_range, row, col):
    """
    Returns the chunk with the specified element removed if it is located
    within the chunk otherwise the chunk is returned unchanged.
    """
    if isinstance(row, gb.Scalar):
        row = row.value
    if isinstance(col, gb.Scalar):
        col = col.value
    if row_range[0].start <= row and row < row_range[0].stop:
        if col_range[0].start <= col and col < col_range[0].stop:
            del inner_mat.value[row - row_range[0].start, col - col_range[0].start]
    return InnerMatrix(inner_mat.value)


def _build_2D_chunk(
    inner_matrix,
    out_row_range,
    out_col_range,
    fragments,
    values,
    dup_op=None,
    clear=False,
):
    """
    Reassembles filtered tuples (row, col, val) in the list `fragments`
    obtained from _pick2D() for the chunk within the given row and column
    ranges (`out_row_range` and `out_col_range`) and returns chunk
    `inner_matrix` built using these tuples.
    """
    rows = np.concatenate([rows for (rows, _, _) in fragments])
    cols = np.concatenate([cols for (_, cols, _) in fragments])
    nrows = out_row_range[0].stop - out_row_range[0].start
    ncols = out_col_range[0].stop - out_col_range[0].start
    if not clear and inner_matrix.value.nvals > 0:
        raise gb.exceptions.OutputNotEmpty()

    if values is None:
        vals = np.concatenate([vals for (_, _, vals) in fragments])
        inner_matrix.value.build(
            rows,
            cols,
            vals,
            nrows=nrows,
            ncols=ncols,
            dup_op=dup_op,
            clear=clear,
        )
    else:
        vals = values
        inner_matrix.value.ss.build_scalar(rows, cols, vals)
    return InnerMatrix(inner_matrix.value)


def _new_Matrix_chunk(out_row_range, out_col_range, gb_dtype=None):
    """
    Return a new chunk with dimensions given by `out_row_range` and `out_col_range`
    """
    nrows = out_row_range[0].stop - out_row_range[0].start
    ncols = out_col_range[0].stop - out_col_range[0].start
    return InnerMatrix(gb.Matrix.new(gb_dtype, nrows=nrows, ncols=ncols))


def _from_values2D(values, fragments, out_row_range, out_col_range, dup_op=None, gb_dtype=None):
    """
    Reassembles filtered tuples (row, col, val) in the list `fragments`
    obtained from _pick2D() for the chunk within the given row and column
    ranges (`out_row_range` and `out_col_range`) and returns a Matrix
    chunk containing these tuples.
    """
    rows = np.concatenate([rows for (rows, _, _) in fragments])
    cols = np.concatenate([cols for (_, cols, _) in fragments])
    if values is None:
        vals = np.concatenate([vals for (_, _, vals) in fragments])
    else:
        vals = values
    nrows = out_row_range[0].stop - out_row_range[0].start
    ncols = out_col_range[0].stop - out_col_range[0].start
    if rows.size == 0 or cols.size == 0:
        return InnerMatrix(gb.Matrix.new(gb_dtype, nrows=nrows, ncols=ncols))
    return InnerMatrix(
        gb.Matrix.from_values(
            rows, cols, vals, nrows=nrows, ncols=ncols, dup_op=dup_op, dtype=gb_dtype
        )
    )


def _pick2D(rows, cols, values, row_range, col_range, shape):
    """
    Filters out only those tuples (row, col, val) that lie within
    the given row and column ranges.  Indices are also offset
    appropriately.
    """
    # validate indices:
    rows = np.where(rows < 0, rows + shape[0], rows)
    bad_indices = (rows < 0) | (shape[0] <= rows)
    if np.any(bad_indices):
        raise IndexOutOfBound

    cols = np.where(cols < 0, cols + shape[1], cols)
    bad_indices = (cols < 0) | (shape[1] <= cols)
    if np.any(bad_indices):
        raise IndexOutOfBound

    # filter into chunk:
    row_range, col_range = row_range[0], col_range[0]
    rows_in = (row_range.start <= rows) & (rows < row_range.stop)
    cols_in = (col_range.start <= cols) & (cols < col_range.stop)
    rows = rows[rows_in & cols_in] - row_range.start
    cols = cols[rows_in & cols_in] - col_range.start
    if isinstance(values, np.ndarray):
        values = values[rows_in & cols_in]
    return (rows, cols, values)


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
    from .io import MMFile
    from scipy.io import mminfo

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


def _get_rows(tuple_extractor):
    return tuple_extractor.rows


def _get_cols(tuple_extractor):
    return tuple_extractor.cols


def _get_vals(tuple_extractor):
    return tuple_extractor.vals


def _flatten(x, axis=None, keepdims=None):
    if type(x) is list:
        x[0].rows = np.concatenate([y.rows for y in x])
        x[0].cols = np.concatenate([y.cols for y in x])
        x[0].vals = np.concatenate([y.vals for y in x])
        return x[0]
    else:
        return x


class MatrixTupleExtractor:
    def __init__(
        self,
        output_range,
        inner_matrix,
        row_offset,
        col_offset,
        nval_start,
        nval_stop,
        gb_dtype=None,
    ):
        """
        Extracts the tuples from a chunk `inner_matrix` but returns only those
        portions of the tuple-arrays whose positions lie within the
        intersection of ranges:
            `output_range` and `[nval_start, nval_stop]`
        Tuple row/col indices are offset appropriately according to the offsets:
            `row_offset` and `col_offset`
        of the chunk.
        """
        self.rows, self.cols, self.vals = inner_matrix.value.to_values(gb_dtype)
        if output_range[0].start < nval_stop[0, 0] and nval_start[0, 0] < output_range[0].stop:
            start = max(output_range[0].start, nval_start[0, 0])
            stop = min(output_range[0].stop, nval_stop[0, 0])
            self.rows += row_offset[0]
            self.cols += col_offset[0]
            start = start - nval_start[0, 0]
            stop = stop - nval_start[0, 0]
            self.rows = self.rows[start:stop]
            self.cols = self.cols[start:stop]
            self.vals = self.vals[start:stop]
        else:
            self.rows = np.array([], dtype=self.rows.dtype)
            self.cols = np.array([], dtype=self.cols.dtype)
            self.vals = np.array([], dtype=self.vals.dtype)


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
gb.utils._output_types[MatrixExpression] = gb.Matrix
