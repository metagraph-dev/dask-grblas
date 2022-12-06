import graphblas.core.mask
from graphblas import replace  # noqa

from . import _version, mask, scalar, utils, vector, ss  # noqa
from .construction import column_stack, concat_vectors, row_stack  # noqa
from .matrix import Matrix
from .scalar import Scalar
from .vector import Vector

from . import matrix  # isort:skip (here to avoid cyclic imports)

for dgb_type, inner_type, gb_type in [
    (mask.StructuralMask, None, graphblas.core.mask.StructuralMask),
    (mask.ValueMask, None, graphblas.core.mask.ValueMask),
    (mask.ComplementedStructuralMask, None, graphblas.core.mask.ComplementedStructuralMask),
    (mask.ComplementedValueMask, None, graphblas.core.mask.ComplementedValueMask),
    (Scalar, scalar.InnerScalar, graphblas.Scalar),
    (Vector, vector.InnerVector, graphblas.Vector),
    (Matrix, matrix.InnerMatrix, graphblas.Matrix),
]:
    utils._grblas_types[dgb_type] = gb_type
    utils._grblas_types[gb_type] = gb_type
    if inner_type is not None:
        utils._grblas_types[inner_type] = gb_type
        utils._inner_types[dgb_type] = inner_type
        utils._inner_types[inner_type] = inner_type
        utils._inner_types[gb_type] = inner_type
        utils._return_types[inner_type] = dgb_type
    utils._return_types[dgb_type] = dgb_type
    utils._return_types[gb_type] = dgb_type

del dgb_type, inner_type, gb_type

__version__ = _version.get_versions()["version"]
