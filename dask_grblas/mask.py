from grblas.mask import Mask as gb_Mask
from .utils import get_grblas_type


class Mask:
    complement = False
    structure = False
    value = False

    __bool__ = gb_Mask.__bool__
    __eq__ = gb_Mask.__eq__

    def __init__(self, mask):
        from . import matrix, vector

        assert type(mask) in {vector.Vector, matrix.Matrix}
        self.mask = mask
        self._meta = get_grblas_type(self)(mask._meta)

    @property
    def is_dOnion(self):
        return getattr(self.mask, "is_dOnion", False)

    @property
    def dOnion_if(self):
        return self.mask._delayed if self.is_dOnion else self


class StructuralMask(Mask):
    complement = False
    structure = True
    value = False

    def __invert__(self):
        return ComplementedStructuralMask(self.mask)

    @property
    def name(self):
        return f"{self.mask.name}.S"


class ValueMask(Mask):
    complement = False
    structure = False
    value = True

    def __invert__(self):
        return ComplementedValueMask(self.mask)

    @property
    def name(self):
        return f"{self.mask.name}.V"


class ComplementedStructuralMask(Mask):
    complement = True
    structure = True
    value = False

    def __invert__(self):
        return StructuralMask(self.mask)

    @property
    def name(self):
        return f"~{self.mask.name}.S"


class ComplementedValueMask(Mask):
    complement = True
    structure = False
    value = True

    def __invert__(self):
        return ValueMask(self.mask)

    @property
    def name(self):
        return f"~{self.mask.name}.V"
