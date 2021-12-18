import dask.array as da

from .matrix import Matrix
from .vector import Vector


def row_stack(seq):
    for i, x in enumerate(seq):
        if not isinstance(x, (Vector, Matrix)):
            raise TypeError(
                f"Bad type of value in position {i}.  Got {type(x)}, expected Vector or Matrix"
            )
    delayed = da.vstack([x._delayed for x in seq])
    return Matrix(delayed)


def column_stack(seq):
    for i, x in enumerate(seq):
        if not isinstance(x, (Vector, Matrix)):
            raise TypeError(
                f"Bad type of value in position {i}.  Got {type(x)}, expected Vector or Matrix"
            )
    delayed = da.hstack([x._delayed[:, None] if x._delayed.ndim == 1 else x._delayed for x in seq])
    return Matrix(delayed)


def concat_vectors(seq):
    # TODO: we could include scalars here as well
    for i, x in enumerate(seq):
        if not isinstance(x, Vector):
            raise TypeError(f"Bad type of value in position {i}.  Got {type(x)}, expected Vector")
    delayed = da.concatenate([x._delayed for x in seq])
    return Vector(delayed)
