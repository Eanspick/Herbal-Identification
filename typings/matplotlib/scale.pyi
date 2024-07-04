from _typeshed import Incomplete
from matplotlib.ticker import AsinhLocator as AsinhLocator, AutoLocator as AutoLocator, AutoMinorLocator as AutoMinorLocator, LogFormatterSciNotation as LogFormatterSciNotation, LogLocator as LogLocator, LogitFormatter as LogitFormatter, LogitLocator as LogitLocator, NullFormatter as NullFormatter, NullLocator as NullLocator, ScalarFormatter as ScalarFormatter, SymmetricalLogLocator as SymmetricalLogLocator
from matplotlib.transforms import IdentityTransform as IdentityTransform, Transform as Transform

class ScaleBase:
    def __init__(self, axis) -> None: ...
    def get_transform(self) -> None: ...
    def set_default_locators_and_formatters(self, axis) -> None: ...
    def limit_range_for_scale(self, vmin, vmax, minpos): ...

class LinearScale(ScaleBase):
    name: str
    def __init__(self, axis) -> None: ...
    def set_default_locators_and_formatters(self, axis) -> None: ...
    def get_transform(self): ...

class FuncTransform(Transform):
    input_dims: int
    output_dims: int
    def __init__(self, forward, inverse) -> None: ...
    def transform_non_affine(self, values): ...
    def inverted(self): ...

class FuncScale(ScaleBase):
    name: str
    def __init__(self, axis, functions) -> None: ...
    def get_transform(self): ...
    def set_default_locators_and_formatters(self, axis) -> None: ...

class LogTransform(Transform):
    input_dims: int
    output_dims: int
    base: Incomplete
    def __init__(self, base, nonpositive: str = 'clip') -> None: ...
    def transform_non_affine(self, values): ...
    def inverted(self): ...

class InvertedLogTransform(Transform):
    input_dims: int
    output_dims: int
    base: Incomplete
    def __init__(self, base) -> None: ...
    def transform_non_affine(self, values): ...
    def inverted(self): ...

class LogScale(ScaleBase):
    name: str
    subs: Incomplete
    def __init__(self, axis, *, base: int = 10, subs: Incomplete | None = None, nonpositive: str = 'clip') -> None: ...
    base: Incomplete
    def set_default_locators_and_formatters(self, axis) -> None: ...
    def get_transform(self): ...
    def limit_range_for_scale(self, vmin, vmax, minpos): ...

class FuncScaleLog(LogScale):
    name: str
    subs: Incomplete
    def __init__(self, axis, functions, base: int = 10) -> None: ...
    @property
    def base(self): ...
    def get_transform(self): ...

class SymmetricalLogTransform(Transform):
    input_dims: int
    output_dims: int
    base: Incomplete
    linthresh: Incomplete
    linscale: Incomplete
    def __init__(self, base, linthresh, linscale) -> None: ...
    def transform_non_affine(self, values): ...
    def inverted(self): ...

class InvertedSymmetricalLogTransform(Transform):
    input_dims: int
    output_dims: int
    base: Incomplete
    linthresh: Incomplete
    invlinthresh: Incomplete
    linscale: Incomplete
    def __init__(self, base, linthresh, linscale) -> None: ...
    def transform_non_affine(self, values): ...
    def inverted(self): ...

class SymmetricalLogScale(ScaleBase):
    name: str
    subs: Incomplete
    def __init__(self, axis, *, base: int = 10, linthresh: int = 2, subs: Incomplete | None = None, linscale: int = 1) -> None: ...
    base: Incomplete
    linthresh: Incomplete
    linscale: Incomplete
    def set_default_locators_and_formatters(self, axis) -> None: ...
    def get_transform(self): ...

class AsinhTransform(Transform):
    input_dims: int
    output_dims: int
    linear_width: Incomplete
    def __init__(self, linear_width) -> None: ...
    def transform_non_affine(self, values): ...
    def inverted(self): ...

class InvertedAsinhTransform(Transform):
    input_dims: int
    output_dims: int
    linear_width: Incomplete
    def __init__(self, linear_width) -> None: ...
    def transform_non_affine(self, values): ...
    def inverted(self): ...

class AsinhScale(ScaleBase):
    name: str
    auto_tick_multipliers: Incomplete
    def __init__(self, axis, *, linear_width: float = 1.0, base: int = 10, subs: str = 'auto', **kwargs) -> None: ...
    linear_width: Incomplete
    def get_transform(self): ...
    def set_default_locators_and_formatters(self, axis) -> None: ...

class LogitTransform(Transform):
    input_dims: int
    output_dims: int
    def __init__(self, nonpositive: str = 'mask') -> None: ...
    def transform_non_affine(self, values): ...
    def inverted(self): ...

class LogisticTransform(Transform):
    input_dims: int
    output_dims: int
    def __init__(self, nonpositive: str = 'mask') -> None: ...
    def transform_non_affine(self, values): ...
    def inverted(self): ...

class LogitScale(ScaleBase):
    name: str
    def __init__(self, axis, nonpositive: str = 'mask', *, one_half: str = '\\frac{1}{2}', use_overline: bool = False) -> None: ...
    def get_transform(self): ...
    def set_default_locators_and_formatters(self, axis) -> None: ...
    def limit_range_for_scale(self, vmin, vmax, minpos): ...

def get_scale_names(): ...
def scale_factory(scale, axis, **kwargs): ...
def register_scale(scale_class) -> None: ...