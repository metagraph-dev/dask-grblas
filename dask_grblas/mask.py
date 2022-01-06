from .utils import get_grblas_type


class Mask:
    complement = False
    structure = False
    value = False

    def __init__(self, mask):
        from . import matrix, vector

        assert type(mask) in {vector.Vector, matrix.Matrix}
        self.mask = mask
        self._meta = get_grblas_type(self)(mask._meta)


class StructuralMask(Mask):
    complement = False
    structure = True
    value = False

    def __invert__(self):
        return ComplementedStructuralMask(self.mask)


class ValueMask(Mask):
    complement = False
    structure = False
    value = True

    def __invert__(self):
        return ComplementedValueMask(self.mask)


class ComplementedStructuralMask(Mask):
    complement = True
    structure = True
    value = False

    def __invert__(self):
        return StructuralMask(self.mask)


class ComplementedValueMask(Mask):
    complement = True
    structure = False
    value = True

    def __invert__(self):
        return ValueMask(self.mask)
