from _typeshed import Incomplete

__all__ = ['TickHelper', 'Formatter', 'FixedFormatter', 'NullFormatter', 'FuncFormatter', 'FormatStrFormatter', 'StrMethodFormatter', 'ScalarFormatter', 'LogFormatter', 'LogFormatterExponent', 'LogFormatterMathtext', 'LogFormatterSciNotation', 'LogitFormatter', 'EngFormatter', 'PercentFormatter', 'Locator', 'IndexLocator', 'FixedLocator', 'NullLocator', 'LinearLocator', 'LogLocator', 'AutoLocator', 'MultipleLocator', 'MaxNLocator', 'AutoMinorLocator', 'SymmetricalLogLocator', 'AsinhLocator', 'LogitLocator']

class _DummyAxis:
    def __init__(self, minpos: int = 0) -> None: ...
    def get_view_interval(self): ...
    def set_view_interval(self, vmin, vmax) -> None: ...
    def get_minpos(self): ...
    def get_data_interval(self): ...
    def set_data_interval(self, vmin, vmax) -> None: ...
    def get_tick_space(self): ...

class TickHelper:
    axis: Incomplete
    def set_axis(self, axis) -> None: ...
    def create_dummy_axis(self, **kwargs) -> None: ...

class Formatter(TickHelper):
    locs: Incomplete
    def __call__(self, x, pos: Incomplete | None = None) -> None: ...
    def format_ticks(self, values): ...
    def format_data(self, value): ...
    def format_data_short(self, value): ...
    def get_offset(self): ...
    def set_locs(self, locs) -> None: ...
    @staticmethod
    def fix_minus(s): ...

class NullFormatter(Formatter):
    def __call__(self, x, pos: Incomplete | None = None): ...

class FixedFormatter(Formatter):
    seq: Incomplete
    offset_string: str
    def __init__(self, seq) -> None: ...
    def __call__(self, x, pos: Incomplete | None = None): ...
    def get_offset(self): ...
    def set_offset_string(self, ofs) -> None: ...

class FuncFormatter(Formatter):
    func: Incomplete
    offset_string: str
    def __init__(self, func) -> None: ...
    def __call__(self, x, pos: Incomplete | None = None): ...
    def get_offset(self): ...
    def set_offset_string(self, ofs) -> None: ...

class FormatStrFormatter(Formatter):
    fmt: Incomplete
    def __init__(self, fmt) -> None: ...
    def __call__(self, x, pos: Incomplete | None = None): ...

class StrMethodFormatter(Formatter):
    fmt: Incomplete
    def __init__(self, fmt) -> None: ...
    def __call__(self, x, pos: Incomplete | None = None): ...

class ScalarFormatter(Formatter):
    orderOfMagnitude: int
    format: str
    def __init__(self, useOffset: Incomplete | None = None, useMathText: Incomplete | None = None, useLocale: Incomplete | None = None) -> None: ...
    def get_useOffset(self): ...
    offset: int
    def set_useOffset(self, val) -> None: ...
    useOffset: Incomplete
    def get_useLocale(self): ...
    def set_useLocale(self, val) -> None: ...
    useLocale: Incomplete
    def get_useMathText(self): ...
    def set_useMathText(self, val) -> None: ...
    useMathText: Incomplete
    def __call__(self, x, pos: Incomplete | None = None): ...
    def set_scientific(self, b) -> None: ...
    def set_powerlimits(self, lims) -> None: ...
    def format_data_short(self, value): ...
    def format_data(self, value): ...
    def get_offset(self): ...
    locs: Incomplete
    def set_locs(self, locs) -> None: ...

class LogFormatter(Formatter):
    minor_thresholds: Incomplete
    def __init__(self, base: float = 10.0, labelOnlyBase: bool = False, minor_thresholds: Incomplete | None = None, linthresh: Incomplete | None = None) -> None: ...
    def set_base(self, base) -> None: ...
    labelOnlyBase: Incomplete
    def set_label_minor(self, labelOnlyBase) -> None: ...
    def set_locs(self, locs: Incomplete | None = None) -> None: ...
    def __call__(self, x, pos: Incomplete | None = None): ...
    def format_data(self, value): ...
    def format_data_short(self, value): ...

class LogFormatterExponent(LogFormatter): ...

class LogFormatterMathtext(LogFormatter):
    def __call__(self, x, pos: Incomplete | None = None): ...

class LogFormatterSciNotation(LogFormatterMathtext): ...

class LogitFormatter(Formatter):
    def __init__(self, *, use_overline: bool = False, one_half: str = '\\frac{1}{2}', minor: bool = False, minor_threshold: int = 25, minor_number: int = 6) -> None: ...
    def use_overline(self, use_overline) -> None: ...
    def set_one_half(self, one_half) -> None: ...
    def set_minor_threshold(self, minor_threshold) -> None: ...
    def set_minor_number(self, minor_number) -> None: ...
    locs: Incomplete
    def set_locs(self, locs): ...
    def __call__(self, x, pos: Incomplete | None = None): ...
    def format_data_short(self, value): ...

class EngFormatter(Formatter):
    ENG_PREFIXES: Incomplete
    unit: Incomplete
    places: Incomplete
    sep: Incomplete
    def __init__(self, unit: str = '', places: Incomplete | None = None, sep: str = ' ', *, usetex: Incomplete | None = None, useMathText: Incomplete | None = None) -> None: ...
    def get_usetex(self): ...
    def set_usetex(self, val) -> None: ...
    usetex: Incomplete
    def get_useMathText(self): ...
    def set_useMathText(self, val) -> None: ...
    useMathText: Incomplete
    def __call__(self, x, pos: Incomplete | None = None): ...
    def format_eng(self, num): ...

class PercentFormatter(Formatter):
    xmax: Incomplete
    decimals: Incomplete
    def __init__(self, xmax: int = 100, decimals: Incomplete | None = None, symbol: str = '%', is_latex: bool = False) -> None: ...
    def __call__(self, x, pos: Incomplete | None = None): ...
    def format_pct(self, x, display_range): ...
    def convert_to_pct(self, x): ...
    @property
    def symbol(self): ...
    @symbol.setter
    def symbol(self, symbol) -> None: ...

class Locator(TickHelper):
    MAXTICKS: int
    def tick_values(self, vmin, vmax) -> None: ...
    def set_params(self, **kwargs) -> None: ...
    def __call__(self) -> None: ...
    def raise_if_exceeds(self, locs): ...
    def nonsingular(self, v0, v1): ...
    def view_limits(self, vmin, vmax): ...

class IndexLocator(Locator):
    offset: Incomplete
    def __init__(self, base, offset) -> None: ...
    def set_params(self, base: Incomplete | None = None, offset: Incomplete | None = None) -> None: ...
    def __call__(self): ...
    def tick_values(self, vmin, vmax): ...

class FixedLocator(Locator):
    locs: Incomplete
    nbins: Incomplete
    def __init__(self, locs, nbins: Incomplete | None = None) -> None: ...
    def set_params(self, nbins: Incomplete | None = None) -> None: ...
    def __call__(self): ...
    def tick_values(self, vmin, vmax): ...

class NullLocator(Locator):
    def __call__(self): ...
    def tick_values(self, vmin, vmax): ...

class LinearLocator(Locator):
    presets: Incomplete
    def __init__(self, numticks: Incomplete | None = None, presets: Incomplete | None = None) -> None: ...
    @property
    def numticks(self): ...
    @numticks.setter
    def numticks(self, numticks) -> None: ...
    def set_params(self, numticks: Incomplete | None = None, presets: Incomplete | None = None) -> None: ...
    def __call__(self): ...
    def tick_values(self, vmin, vmax): ...
    def view_limits(self, vmin, vmax): ...

class MultipleLocator(Locator):
    def __init__(self, base: float = 1.0, offset: float = 0.0) -> None: ...
    def set_params(self, base: Incomplete | None = None, offset: Incomplete | None = None) -> None: ...
    def __call__(self): ...
    def tick_values(self, vmin, vmax): ...
    def view_limits(self, dmin, dmax): ...

class _Edge_integer:
    step: Incomplete
    def __init__(self, step, offset) -> None: ...
    def closeto(self, ms, edge): ...
    def le(self, x): ...
    def ge(self, x): ...

class MaxNLocator(Locator):
    default_params: Incomplete
    def __init__(self, nbins: Incomplete | None = None, **kwargs) -> None: ...
    def set_params(self, **kwargs) -> None: ...
    def __call__(self): ...
    def tick_values(self, vmin, vmax): ...
    def view_limits(self, dmin, dmax): ...

class LogLocator(Locator):
    numticks: Incomplete
    def __init__(self, base: float = 10.0, subs=(1.0,), numdecs: int = 4, numticks: Incomplete | None = None) -> None: ...
    def set_params(self, base: Incomplete | None = None, subs: Incomplete | None = None, numdecs: Incomplete | None = None, numticks: Incomplete | None = None) -> None: ...
    numdecs: Incomplete
    def __call__(self): ...
    def tick_values(self, vmin, vmax): ...
    def view_limits(self, vmin, vmax): ...
    def nonsingular(self, vmin, vmax): ...

class SymmetricalLogLocator(Locator):
    numticks: int
    def __init__(self, transform: Incomplete | None = None, subs: Incomplete | None = None, linthresh: Incomplete | None = None, base: Incomplete | None = None) -> None: ...
    def set_params(self, subs: Incomplete | None = None, numticks: Incomplete | None = None) -> None: ...
    def __call__(self): ...
    def tick_values(self, vmin, vmax): ...
    def view_limits(self, vmin, vmax): ...

class AsinhLocator(Locator):
    linear_width: Incomplete
    numticks: Incomplete
    symthresh: Incomplete
    base: Incomplete
    subs: Incomplete
    def __init__(self, linear_width, numticks: int = 11, symthresh: float = 0.2, base: int = 10, subs: Incomplete | None = None) -> None: ...
    def set_params(self, numticks: Incomplete | None = None, symthresh: Incomplete | None = None, base: Incomplete | None = None, subs: Incomplete | None = None) -> None: ...
    def __call__(self): ...
    def tick_values(self, vmin, vmax): ...

class LogitLocator(MaxNLocator):
    def __init__(self, minor: bool = False, *, nbins: str = 'auto') -> None: ...
    def set_params(self, minor: Incomplete | None = None, **kwargs) -> None: ...
    @property
    def minor(self): ...
    @minor.setter
    def minor(self, value) -> None: ...
    def tick_values(self, vmin, vmax): ...
    def nonsingular(self, vmin, vmax): ...

class AutoLocator(MaxNLocator):
    def __init__(self) -> None: ...

class AutoMinorLocator(Locator):
    ndivs: Incomplete
    def __init__(self, n: Incomplete | None = None) -> None: ...
    def __call__(self): ...
    def tick_values(self, vmin, vmax) -> None: ...
