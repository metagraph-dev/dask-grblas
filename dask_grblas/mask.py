from .utils import get_grblas_type


class Mask:
    complement = False
    structure = False
    value = False

    def __init__(self, mask):
        from . import matrix, vector, base

        assert type(mask) in {vector.Vector, matrix.Matrix}
        self.mask = mask
        self._meta = get_grblas_type(self)(mask._meta)
        if base.is_DOnion(mask._delayed):
            self.mask = mask._delayed.deep_extract(self._meta, self.__class__)

    @property
    def is_dOnion(self):
        from .base import is_DOnion

        return is_DOnion(self.mask)

    @property
    def dOnion_if(self):
        return self.mask if self.is_dOnion else self


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
