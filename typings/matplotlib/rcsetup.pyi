import ast
from _typeshed import Incomplete
from matplotlib import cbook as cbook
from matplotlib._enums import CapStyle as CapStyle, JoinStyle as JoinStyle
from matplotlib._fontconfig_pattern import parse_fontconfig_pattern as parse_fontconfig_pattern
from matplotlib.cbook import ls_mapper as ls_mapper
from matplotlib.colors import Colormap as Colormap, is_color_like as is_color_like

interactive_bk: Incomplete
non_interactive_bk: Incomplete
all_backends: Incomplete

class ValidateInStrings:
    key: Incomplete
    ignorecase: Incomplete
    valid: Incomplete
    def __init__(self, key, valid, ignorecase: bool = False, *, _deprecated_since: Incomplete | None = None) -> None: ...
    def __call__(self, s): ...

def validate_any(s): ...

validate_anylist: Incomplete

def validate_bool(b): ...
def validate_axisbelow(s): ...
def validate_dpi(s): ...

validate_string: Incomplete
validate_string_or_None: Incomplete
validate_stringlist: Incomplete
validate_int: Incomplete
validate_int_or_None: Incomplete
validate_float: Incomplete
validate_float_or_None: Incomplete
validate_floatlist: Incomplete

def validate_fonttype(s): ...
def validate_backend(s): ...
def validate_color_or_inherit(s): ...
def validate_color_or_auto(s): ...
def validate_color_for_prop_cycle(s): ...
def validate_color(s): ...

validate_colorlist: Incomplete

def validate_aspect(s): ...
def validate_fontsize_None(s): ...
def validate_fontsize(s): ...

validate_fontsizelist: Incomplete

def validate_fontweight(s): ...
def validate_fontstretch(s): ...
def validate_font_properties(s): ...
def validate_whiskers(s): ...
def validate_ps_distiller(s): ...

validate_fillstyle: Incomplete
validate_fillstylelist: Incomplete

def validate_markevery(s): ...

validate_markeverylist: Incomplete

def validate_bbox(s): ...
def validate_sketch(s): ...
def validate_hatch(s): ...

validate_hatchlist: Incomplete
validate_dashlist: Incomplete

def cycler(*args, **kwargs): ...

class _DunderChecker(ast.NodeVisitor):
    def visit_Attribute(self, node) -> None: ...

def validate_cycler(s): ...
def validate_hist_bins(s): ...

class _ignorecase(list): ...