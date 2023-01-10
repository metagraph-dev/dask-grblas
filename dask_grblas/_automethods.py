from grblas import config


def _get_value(self, attr=None, default=None):
    if config.get("autocompute"):
        if self._value is None:
            self._value = self.new()
            if getattr(self, "is_dOnion", False):
                self._value = self._value.strip()
        if attr is None:
            return self._value
        else:
            return getattr(self._value, attr)
    if default is not None:
        return default.__get__(self)
    raise TypeError(
        f"{attr} not enabled for objects of type {type(self)}.  "
        f"Use `.new()` to create a new {self.output_type.__name__}.\n\n"
        "Hint: use `grblas.config.set(autocompute=True)` to enable "
        "automatic computation of expressions."
    )
