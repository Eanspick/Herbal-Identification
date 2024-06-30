from ._color_data import BASE_COLORS as BASE_COLORS, CSS4_COLORS as CSS4_COLORS, TABLEAU_COLORS as TABLEAU_COLORS, XKCD_COLORS as XKCD_COLORS
from _typeshed import Incomplete
from collections.abc import Mapping
from matplotlib import cbook as cbook, scale as scale

class _ColorMapping(dict):
    cache: Incomplete
    def __init__(self, mapping) -> None: ...
    def __setitem__(self, key, value) -> None: ...
    def __delitem__(self, key) -> None: ...

def get_named_colors_mapping(): ...

class ColorSequenceRegistry(Mapping):
    def __init__(self) -> None: ...
    def __getitem__(self, item): ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def register(self, name, color_list) -> None: ...
    def unregister(self, name) -> None: ...

def is_color_like(c): ...
def same_color(c1, c2): ...
def to_rgba(c, alpha: Incomplete | None = None): ...
def to_rgba_array(c, alpha: Incomplete | None = None): ...
def to_rgb(c): ...
def to_hex(c, keep_alpha: bool = False): ...
cnames = CSS4_COLORS
hexColorPattern: Incomplete
rgb2hex = to_hex
hex2color = to_rgb

class ColorConverter:
    colors: Incomplete
    cache: Incomplete
    to_rgb: Incomplete
    to_rgba: Incomplete
    to_rgba_array: Incomplete

colorConverter: Incomplete

class Colormap:
    name: Incomplete
    N: Incomplete
    colorbar_extend: bool
    def __init__(self, name, N: int = 256) -> None: ...
    def __call__(self, X, alpha: Incomplete | None = None, bytes: bool = False): ...
    def __copy__(self): ...
    def __eq__(self, other): ...
    def get_bad(self): ...
    def set_bad(self, color: str = 'k', alpha: Incomplete | None = None) -> None: ...
    def get_under(self): ...
    def set_under(self, color: str = 'k', alpha: Incomplete | None = None) -> None: ...
    def get_over(self): ...
    def set_over(self, color: str = 'k', alpha: Incomplete | None = None) -> None: ...
    def set_extremes(self, *, bad: Incomplete | None = None, under: Incomplete | None = None, over: Incomplete | None = None) -> None: ...
    def with_extremes(self, *, bad: Incomplete | None = None, under: Incomplete | None = None, over: Incomplete | None = None): ...
    def is_gray(self): ...
    def resampled(self, lutsize): ...
    def reversed(self, name: Incomplete | None = None) -> None: ...
    def copy(self): ...

class LinearSegmentedColormap(Colormap):
    monochrome: bool
    def __init__(self, name, segmentdata, N: int = 256, gamma: float = 1.0) -> None: ...
    def set_gamma(self, gamma) -> None: ...
    @staticmethod
    def from_list(name, colors, N: int = 256, gamma: float = 1.0): ...
    def resampled(self, lutsize): ...
    def reversed(self, name: Incomplete | None = None): ...

class ListedColormap(Colormap):
    monochrome: bool
    colors: Incomplete
    def __init__(self, colors, name: str = 'from_list', N: Incomplete | None = None) -> None: ...
    def resampled(self, lutsize): ...
    def reversed(self, name: Incomplete | None = None): ...

class Normalize:
    callbacks: Incomplete
    def __init__(self, vmin: Incomplete | None = None, vmax: Incomplete | None = None, clip: bool = False) -> None: ...
    @property
    def vmin(self): ...
    @vmin.setter
    def vmin(self, value) -> None: ...
    @property
    def vmax(self): ...
    @vmax.setter
    def vmax(self, value) -> None: ...
    @property
    def clip(self): ...
    @clip.setter
    def clip(self, value) -> None: ...
    @staticmethod
    def process_value(value): ...
    def __call__(self, value, clip: Incomplete | None = None): ...
    def inverse(self, value): ...
    def autoscale(self, A) -> None: ...
    def autoscale_None(self, A) -> None: ...
    def scaled(self): ...

class TwoSlopeNorm(Normalize):
    def __init__(self, vcenter, vmin: Incomplete | None = None, vmax: Incomplete | None = None) -> None: ...
    @property
    def vcenter(self): ...
    @vcenter.setter
    def vcenter(self, value) -> None: ...
    vmin: Incomplete
    vmax: Incomplete
    def autoscale_None(self, A) -> None: ...
    def __call__(self, value, clip: Incomplete | None = None): ...
    def inverse(self, value): ...

class CenteredNorm(Normalize):
    def __init__(self, vcenter: int = 0, halfrange: Incomplete | None = None, clip: bool = False) -> None: ...
    def autoscale(self, A) -> None: ...
    def autoscale_None(self, A) -> None: ...
    @property
    def vmin(self): ...
    @vmin.setter
    def vmin(self, value) -> None: ...
    @property
    def vmax(self): ...
    @vmax.setter
    def vmax(self, value) -> None: ...
    @property
    def vcenter(self): ...
    @vcenter.setter
    def vcenter(self, vcenter) -> None: ...
    @property
    def halfrange(self): ...
    @halfrange.setter
    def halfrange(self, halfrange) -> None: ...

def make_norm_from_scale(scale_cls, base_norm_cls: Incomplete | None = None, *, init: Incomplete | None = None): ...

class FuncNorm(Normalize): ...

LogNorm: Incomplete

class SymLogNorm(Normalize):
    @property
    def linthresh(self): ...
    @linthresh.setter
    def linthresh(self, value) -> None: ...

class AsinhNorm(Normalize):
    @property
    def linear_width(self): ...
    @linear_width.setter
    def linear_width(self, value) -> None: ...

class PowerNorm(Normalize):
    gamma: Incomplete
    def __init__(self, gamma, vmin: Incomplete | None = None, vmax: Incomplete | None = None, clip: bool = False) -> None: ...
    def __call__(self, value, clip: Incomplete | None = None): ...
    def inverse(self, value): ...

class BoundaryNorm(Normalize):
    boundaries: Incomplete
    N: Incomplete
    Ncmap: Incomplete
    extend: Incomplete
    def __init__(self, boundaries, ncolors, clip: bool = False, *, extend: str = 'neither') -> None: ...
    def __call__(self, value, clip: Incomplete | None = None): ...
    def inverse(self, value) -> None: ...

class NoNorm(Normalize):
    def __call__(self, value, clip: Incomplete | None = None): ...
    def inverse(self, value): ...

def rgb_to_hsv(arr): ...
def hsv_to_rgb(hsv): ...

class LightSource:
    azdeg: Incomplete
    altdeg: Incomplete
    hsv_min_val: Incomplete
    hsv_max_val: Incomplete
    hsv_min_sat: Incomplete
    hsv_max_sat: Incomplete
    def __init__(self, azdeg: int = 315, altdeg: int = 45, hsv_min_val: int = 0, hsv_max_val: int = 1, hsv_min_sat: int = 1, hsv_max_sat: int = 0) -> None: ...
    @property
    def direction(self): ...
    def hillshade(self, elevation, vert_exag: int = 1, dx: int = 1, dy: int = 1, fraction: float = 1.0): ...
    def shade_normals(self, normals, fraction: float = 1.0): ...
    def shade(self, data, cmap, norm: Incomplete | None = None, blend_mode: str = 'overlay', vmin: Incomplete | None = None, vmax: Incomplete | None = None, vert_exag: int = 1, dx: int = 1, dy: int = 1, fraction: int = 1, **kwargs): ...
    def shade_rgb(self, rgb, elevation, fraction: float = 1.0, blend_mode: str = 'hsv', vert_exag: int = 1, dx: int = 1, dy: int = 1, **kwargs): ...
    def blend_hsv(self, rgb, intensity, hsv_max_sat: Incomplete | None = None, hsv_max_val: Incomplete | None = None, hsv_min_val: Incomplete | None = None, hsv_min_sat: Incomplete | None = None): ...
    def blend_soft_light(self, rgb, intensity): ...
    def blend_overlay(self, rgb, intensity): ...

def from_levels_and_colors(levels, colors, extend: str = 'neither'): ...
