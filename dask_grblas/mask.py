from .utils import get_grblas_type


class Mask:
    def __init__(self, mask):
        from . import matrix, vector
        assert type(mask) in {vector.Vector, matrix.Matrix}
        self.mask = mask
        self._meta = get_grblas_type(self)(mask._meta)


class StructuralMask(Mask):
    def __invert__(self):
        return ComplementedStructuralMask(self.mask)


class ValueMask(Mask):
    def __invert__(self):
        return ComplementedValueMask(self.mask)


class ComplementedStructuralMask(Mask):
    def __invert__(self):
        return StructuralMask(self.mask)


class ComplementedValueMask(Mask):
    def __invert__(self):
        return ValueMask(self.mask)
