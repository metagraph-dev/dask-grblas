import os

from math import floor, sqrt
from numpy import asarray, conj, zeros, concatenate, ones, empty
from scipy.io import mmio  # noqa


def symm_I_J(pos, n):
    """
    returns row and column indices of matrix element at position `pos`
    of flattened symmetric matrix of number of rows `n`
    """
    col = symm_col(pos, n)
    row = pos - symm_area(col, n) + col
    # deal with any possible roundoff errors from symm_col():
    while row >= n:
        col += 1
        row = pos - symm_area(col, n) + col
    while row < col:
        col -= 1
        row = pos - symm_area(col, n) + col
    return row, col


def symm_col(area, n):
    """
    returns largest integer x such that area >= n + (n - 1) + ... + (n - x + 1)
    """
    return floor((2 * n + 1 - sqrt((2 * n + 1) ** 2 - 8 * area)) / 2)


def symm_area(col, n):
    """
    returns n + (n - 1) + ... + (n - col + 1)
    i.e., the number of matrix elements below and including the diagonal and
    from column 0 to column `col`
    """
    return col * (2 * n - col + 1) // 2


def skew_I_J(pos, n):
    """
    returns row and column indices of matrix element at position `pos`
    of flattened skew matrix of number of rows `n`
    """
    col = skew_col(pos, n)
    row = pos - skew_area(col, n) + col + 1
    # deal with any possible roundoff errors from skew_col():
    while row >= n:
        col += 1
        row = pos - symm_area(col, n) + col + 1
    while row <= col:
        col -= 1
        row = pos - symm_area(col, n) + col + 1
    return row, col


def skew_col(area, n):
    """
    returns largest integer x such that area >= (n - 1) + (n - 2) + ... + (n - x)
    """
    return floor((2 * n - 1 - sqrt((2 * n - 1) ** 2 - 8 * area)) / 2)


def skew_area(col, n):
    """
    returns (n - 1) + (n - 2) + ... + (n - col)
    i.e., the number of matrix elements below the diagonal and
    from column 0 to column `col`
    """
    return col * (2 * n - col - 1) // 2


# -----------------------------------------------------------------------------


def home(stream, search_window_size=8):
    """
    moves cursor to the beginning of the current line
    """
    if stream.tell() == 0:
        return
    else:
        stream.seek(-1, os.SEEK_CUR)
        if stream.read(1) == b"\n":
            return

    step = min(search_window_size, stream.tell())
    stream.seek(-step, os.SEEK_CUR)
    lines = stream.readlines(step - 1) if step > 1 else [stream.readline()]
    while len(lines) == 1 and search_window_size == step:
        search_window_size *= 2
        step = min(search_window_size, stream.tell())
        stream.seek(-step, os.SEEK_CUR)
        lines = stream.readlines(step - 1) if step > 1 else [stream.readline()]
    stream.seek(-len(lines[-1]), os.SEEK_CUR)


# -----------------------------------------------------------------------------


def mmread(source, *, dup_op=None, name=None, row_begin=0, row_end=None, col_begin=0, col_end=None):
    """
    Read the contents of a Matrix Market filename or file into a new Matrix.

    This uses `scipy.io.mmread`:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.mmread.html

    For more information on the Matrix Market format, see:
    https://math.nist.gov/MatrixMarket/formats.html
    """
    from . import Matrix

    try:
        from scipy.sparse import coo_matrix  # noqa
    except ImportError:  # pragma: no cover
        raise ImportError("scipy is required to read Matrix Market files")
    array = MMFile().read(
        source, row_begin=row_begin, row_end=row_end, col_begin=col_begin, col_end=col_end
    )
    if isinstance(array, coo_matrix):
        nrows, ncols = array.shape
        return Matrix.from_values(
            array.row, array.col, array.data, nrows=nrows, ncols=ncols, dup_op=dup_op, name=name
        )
    # SS, SuiteSparse-specific: import_full
    return Matrix.ss.import_fullr(values=array, take_ownership=True, name=name)


# -----------------------------------------------------------------------------


class MMFile(mmio.MMFile):
    def get_data_begin(self, source):
        """
        Reads the contents of a Matrix Market file-like 'source' into a matrix.

        Parameters
        ----------
        source : str or file-like
            Matrix Market filename (extensions .mtx, .mtz.gz)
            or open file object.

        Returns
        -------
        a : ndarray or coo_matrix
            Dense or sparse matrix depending on the matrix format in the
            Matrix Market file.
        """
        stream, close_it = self._open(source)

        try:
            return self._get_data_begin(stream)

        finally:
            if close_it:
                stream.close()

    # -------------------------------------------------------------------------
    def _get_data_begin(self, stream):
        _ = self.__class__.info(stream)
        return stream.tell()

    # -----------------------------------------------------------------------------
    def read_part(self, source, line_start=None, line_stop=None, read_begin=None, read_end=None):
        """
        Reads the contents of a Matrix Market file-like 'source' into a matrix.

        Parameters
        ----------
        source : str or file-like
            Matrix Market filename (extensions .mtx, .mtz.gz)
            or open file object.

        Returns
        -------
        a : ndarray or coo_matrix
            Dense or sparse matrix depending on the matrix format in the
            Matrix Market file.
        """
        stream, close_it = self._open(source)

        try:
            self._parse_header(stream)
            return self._parse_body_part(
                stream,
                line_start=line_start,
                line_stop=line_stop,
                read_begin=read_begin,
                read_end=read_end,
            )

        finally:
            if close_it:
                stream.close()

    # -----------------------------------------------------------------------------
    def read_chunk(self, source, row_begin=0, row_end=None, col_begin=0, col_end=None):
        """
        Reads the contents of a Matrix Market file-like 'source' into a matrix.

        Parameters
        ----------
        source : str or file-like
            Matrix Market filename (extensions .mtx, .mtz.gz)
            or open file object.

        Returns
        -------
        a : ndarray or coo_matrix
            Dense or sparse matrix depending on the matrix format in the
            Matrix Market file.
        """
        stream, close_it = self._open(source)

        try:
            self._parse_header(stream)
            return self._parse_body_chunk(
                stream,
                row_begin=row_begin,
                row_end=row_end,
                col_begin=col_begin,
                col_end=col_end,
            )

        finally:
            if close_it:
                stream.close()

    # -----------------------------------------------------------------------------

    def _parse_body_chunk(self, stream, row_begin=0, row_end=None, col_begin=0, col_end=None):
        rows, cols, entries, format, field, symm = (
            self.rows,
            self.cols,
            self.entries,
            self.format,
            self.field,
            self.symmetry,
        )

        if row_end:
            chunk_rows = row_end - row_begin
        else:
            chunk_rows = rows
        if col_end:
            chunk_cols = col_end - col_begin
        else:
            chunk_cols = cols
        try:
            from scipy.sparse import coo_matrix
        except ImportError:
            coo_matrix = None

        dtype = self.DTYPES_BY_FIELD.get(field, None)

        has_symmetry = self.has_symmetry
        is_integer = field == self.FIELD_INTEGER
        is_unsigned_integer = field == self.FIELD_UNSIGNED
        is_complex = field == self.FIELD_COMPLEX
        is_skew = symm == self.SYMMETRY_SKEW_SYMMETRIC
        is_herm = symm == self.SYMMETRY_HERMITIAN
        is_pattern = field == self.FIELD_PATTERN

        if format == self.FORMAT_ARRAY:
            a = zeros((chunk_rows, chunk_cols), dtype=dtype)
            line = 1
            i, j = 0, 0
            row_is_hit = row_begin <= i and i < row_end
            col_is_hit = col_begin <= j and j < col_end
            if is_skew:
                if row_is_hit and col_is_hit:
                    a[i, j] = 0
                if i < rows - 1:
                    i += 1
            while line:
                line = stream.readline()
                # line.startswith('%')
                if not line or line[0] in ["%", 37] or not line.strip():
                    continue
                if is_integer:
                    aij = int(line)
                elif is_unsigned_integer:
                    aij = int(line)
                elif is_complex:
                    aij = complex(*map(float, line.split()))
                else:
                    aij = float(line)

                row_is_hit = row_begin <= i and i < row_end
                col_is_hit = col_begin <= j and j < col_end

                if row_is_hit and col_is_hit:
                    a[i - row_begin, j - col_begin] = aij
                if has_symmetry and i != j:
                    row_is_hit = row_begin < j and j <= row_end
                    col_is_hit = col_begin < i and i <= col_end
                    if row_is_hit and col_is_hit:
                        if is_skew:
                            a[j - row_begin, i - col_begin] = -aij
                        elif is_herm:
                            a[j - row_begin, i - col_begin] = conj(aij)
                        else:
                            a[j - row_begin, i - col_begin] = aij
                if i < rows - 1:
                    i = i + 1
                else:
                    j = j + 1
                    if not has_symmetry:
                        i = 0
                    else:
                        i = j
                        if is_skew:
                            row_is_hit = row_begin <= i and i < row_end
                            col_is_hit = col_begin <= j and j < col_end
                            if row_is_hit and col_is_hit:
                                a[i - row_begin, j - col_begin] = 0
                            if i < rows - 1:
                                i += 1

            if is_skew:
                if not (i in [0, j] and j == cols - 1):
                    raise ValueError("Parse error, did not read all lines.")
            else:
                if not (i in [0, j] and j == cols):
                    raise ValueError("Parse error, did not read all lines.")

        elif format == self.FORMAT_COORDINATE and coo_matrix is None:
            # Read sparse matrix to dense when coo_matrix is not available.
            a = zeros((chunk_rows, chunk_cols), dtype=dtype)
            line = 1
            k = 0
            while line:
                line = stream.readline()
                # line.startswith('%')
                if not line or line[0] in ["%", 37] or not line.strip():
                    continue
                word = line.split()
                i, j = map(int, word[:2])
                i, j = i - 1, j - 1
                if is_integer:
                    aij = int(word[2])
                elif is_unsigned_integer:
                    aij = int(word[2])
                elif is_complex:
                    aij = complex(*map(float, word[2:]))
                else:
                    aij = float(word[2])
                row_is_hit = row_begin <= i and i < row_end
                col_is_hit = col_begin <= j and j < col_end
                if row_is_hit and col_is_hit:
                    a[i - row_begin, j - col_begin] = aij
                if has_symmetry and i != j:
                    row_is_hit = row_begin < j and j <= row_end
                    col_is_hit = col_begin < i and i <= col_end
                    if row_is_hit and col_is_hit:
                        if is_skew:
                            a[j - row_begin, i - col_begin] = -aij
                        elif is_herm:
                            a[j - row_begin, i - col_begin] = conj(aij)
                        else:
                            a[j - row_begin, i - col_begin] = aij
                k = k + 1
            if not k == entries:
                ValueError("Did not read all entries")

        elif format == self.FORMAT_COORDINATE:
            # Read sparse COOrdinate format

            if entries == 0:
                # empty matrix
                return coo_matrix((chunk_rows, chunk_cols), dtype=dtype)

            I1 = zeros(0, dtype="intc")
            I2 = zeros(0, dtype="intc")
            J = zeros(0, dtype="intc")
            J2 = zeros(0, dtype="intc")
            if is_pattern:
                V = ones(0, dtype="int8")
                V2 = ones(0, dtype="int8")
            elif is_integer:
                V = zeros(0, dtype="intp")
                V2 = zeros(0, dtype="intp")
            elif is_unsigned_integer:
                V = zeros(0, dtype="uint64")
                V2 = zeros(0, dtype="uint64")
            elif is_complex:
                V = zeros(0, dtype="complex")
                V2 = zeros(0, dtype="complex")
            else:
                V = zeros(0, dtype="float")
                V2 = zeros(0, dtype="float")

            entry_number = 0
            for line in stream:
                # line.startswith('%')
                if not line or line[0] in ["%", 37] or not line.strip():
                    continue

                if entry_number + 1 > entries:
                    raise ValueError("'entries' in header is smaller than " "number of entries")
                word = line.split()
                i, j = map(int, word[:2])

                row_is_hit = row_begin < i and i <= row_end
                col_is_hit = col_begin < j and j <= col_end

                if not row_is_hit or not col_is_hit:
                    # check transpose
                    row_is_hit = row_begin < j and j <= row_end
                    col_is_hit = col_begin < i and i <= col_end
                    if row_is_hit and col_is_hit:
                        I2.resize(I2.size + 1)
                        J2.resize(J2.size + 1)
                        V2.resize(V2.size + 1)
                        I2[-1] = i
                        J2[-1] = j
                        if not is_pattern:
                            if is_integer:
                                V2[-1] = int(word[2])
                            elif is_unsigned_integer:
                                V2[-1] = int(word[2])
                            elif is_complex:
                                V2[-1] = complex(*map(float, word[2:]))
                            else:
                                V2[-1] = float(word[2])
                    continue

                I1.resize(entry_number + 1)
                J.resize(entry_number + 1)
                V.resize(entry_number + 1)

                I1[entry_number] = i
                J[entry_number] = j
                if not is_pattern:
                    if is_integer:
                        V[entry_number] = int(word[2])
                    elif is_unsigned_integer:
                        V[entry_number] = int(word[2])
                    elif is_complex:
                        V[entry_number] = complex(*map(float, word[2:]))
                    else:
                        V[entry_number] = float(word[2])
                entry_number += 1

            I1 -= 1  # adjust indices (base 1 -> base 0)
            J -= 1

            if has_symmetry:
                mask = I1 != J  # off diagonal mask
                od_I = I1[mask]
                od_J = J[mask]
                od_V = V[mask]

                row_is_hit = (row_begin <= od_J) & (od_J < row_end)
                col_is_hit = (col_begin <= od_I) & (od_I < col_end)
                in_chunk = row_is_hit & col_is_hit
                od_I = I1[in_chunk]
                od_J = J[in_chunk]
                od_V = V[in_chunk]

                I1 = concatenate((I1, od_J, J2))
                J = concatenate((J, od_I, I2))

                if is_skew:
                    od_V *= -1
                    V2 *= -1
                elif is_herm:
                    od_V = od_V.conjugate()
                    V2 = V2.conjugate()

                V = concatenate((V, od_V, V2))

                I1 -= row_begin
                J -= col_begin

            a = coo_matrix((V, (I1, J)), shape=(chunk_rows, chunk_cols), dtype=dtype)
        else:
            raise NotImplementedError(format)

        return a

    # -----------------------------------------------------------------------------

    def _parse_body_part(
        self, stream, line_start=None, line_stop=None, read_begin=None, read_end=None
    ):
        rows, entries, format, field, symm = (
            self.rows,
            self.entries,
            self.format,
            self.field,
            self.symmetry,
        )

        has_symmetry = self.has_symmetry
        is_integer = field == self.FIELD_INTEGER
        is_unsigned_integer = field == self.FIELD_UNSIGNED
        is_complex = field == self.FIELD_COMPLEX
        is_skew = symm == self.SYMMETRY_SKEW_SYMMETRIC
        is_herm = symm == self.SYMMETRY_HERMITIAN
        is_pattern = field == self.FIELD_PATTERN

        I1 = empty(0, dtype="intc")
        J = empty(0, dtype="intc")
        if is_pattern:
            V = empty(0, dtype="int8")
        elif is_integer:
            V = empty(0, dtype="intp")
        elif is_unsigned_integer:
            V = empty(0, dtype="uint64")
        elif is_complex:
            V = empty(0, dtype="complex")
        else:
            V = empty(0, dtype="float")

        if format == self.FORMAT_ARRAY:
            if line_start is None:
                line_start = 0
                if line_stop is None:
                    if not (read_begin is None and read_end is None):
                        raise ValueError(
                            "Keyword arguments `read_begin` and `read_end` are not applicable "
                            "for this format.  Use `line_start` and `line_stop` instead.\n"
                        )

            if has_symmetry:
                if is_skew:
                    i, j = skew_I_J(line_start, rows)
                else:
                    i, j = symm_I_J(line_start, rows)
            else:
                i, j = line_start % rows, line_start // rows

            matrix_line_no = -1
            line = 1
            I_, J_, V_ = [], [], []
            while line:
                line = stream.readline()
                # line.startswith('%')
                if not line or line[0] in ["%", 37] or not line.strip():
                    continue

                matrix_line_no += 1
                if matrix_line_no < line_start:
                    continue
                if line_stop is not None and matrix_line_no >= line_stop:
                    break

                if is_integer:
                    aij = int(line)
                elif is_unsigned_integer:
                    aij = int(line)
                elif is_complex:
                    aij = complex(*map(float, line.split()))
                else:
                    aij = float(line)
                if aij != 0:
                    # a[i, j] = aij
                    I_ += [i]
                    J_ += [j]
                    V_ += [aij]
                    if has_symmetry and i != j:
                        I_ += [j]
                        J_ += [i]
                        if is_skew:
                            # a[j, i] = -aij
                            V_ += [-aij]
                        elif is_herm:
                            # a[j, i] = conj(aij)
                            V_ += [conj(aij)]
                        else:
                            # a[j, i] = aij
                            V_ += [aij]
                if i < rows - 1:
                    i = i + 1
                else:
                    j = j + 1
                    if not has_symmetry:
                        i = 0
                    else:
                        i = j
                        if is_skew:
                            if i < rows - 1:
                                i += 1

            I1 = asarray(I_, dtype=I1.dtype)
            J = asarray(J_, dtype=J.dtype)
            V = asarray(V_, dtype=V.dtype)

        elif format == self.FORMAT_COORDINATE:
            # Read sparse COOrdinate format
            if read_begin is None:
                read_begin = 0
                if read_end is None:
                    if not (line_start is None and line_stop is None):
                        raise ValueError(
                            "Keyword arguments `line_start` and `line_stop` are not applicable "
                            "for this format.  Use `read_begin` and `read_end` instead.\n"
                        )

            if entries == 0:
                # empty matrix
                return I1, J, V

            current_pos = stream.tell()
            stream.seek(0, os.SEEK_END)
            eof = stream.tell()
            # print(f'INIT: {read_begin=}; {read_end=}; {current_pos=}; {eof=}')
            if read_end < eof:
                stream.seek(read_end, os.SEEK_SET)
                home(stream)
                read_end = stream.tell()
                if current_pos >= read_end:
                    # empty matrix
                    return I1, J, V

            stream.seek(read_begin, os.SEEK_SET)
            home(stream)
            read_begin = max(stream.tell(), current_pos)
            stream.seek(read_begin, os.SEEK_SET)

            # print(f'BEGIN: {read_begin=}; {read_end=}; {current_pos=}; {eof=}')
            I_, J_, V_ = [], [], []
            for line in stream:
                if read_end is not None and stream.tell() > read_end:
                    # print(f'STOP: {read_begin=}; {read_end=}; {stream.tell()=}')
                    break

                if not line or line[0] in ["%", 37] or not line.strip():
                    continue

                # print(f'CONT: {read_begin=}; {read_end=}; {stream.tell()=}')
                word = line.split()

                i, j = map(int, word[:2])
                I_ += [i]
                J_ += [j]

                if not is_pattern:
                    if is_integer:
                        V_ += [int(word[2])]
                    elif is_unsigned_integer:
                        V_ += [int(word[2])]
                    elif is_complex:
                        V_ += [complex(*map(float, word[2:]))]
                    else:
                        V_ += [float(word[2])]
                else:
                    V_ += [1]

            I1 = asarray(I_, dtype=I1.dtype)
            J = asarray(J_, dtype=J.dtype)
            V = asarray(V_, dtype=V.dtype)
            if I1.size == 0:
                # print(f'STOP: {read_begin=}; {read_end=}; {I1=}; {J=}; {V=}')
                return I1, J, V

            I1 -= 1  # adjust indices (base 1 -> base 0)
            J -= 1

            if has_symmetry:
                mask = I1 != J  # off diagonal mask
                od_I = I1[mask]
                od_J = J[mask]
                od_V = V[mask]

                I1 = concatenate((I1, od_J))
                J = concatenate((J, od_I))

                if is_skew:
                    od_V *= -1
                elif is_herm:
                    od_V = od_V.conjugate()

                V = concatenate((V, od_V))

        else:
            raise NotImplementedError(format)

        # print(f'STOP: {read_begin=}; {read_end=}; {I1=}; {J=}; {V=}')
        return I1, J, V

    #  ------------------------------------------------------------------------
