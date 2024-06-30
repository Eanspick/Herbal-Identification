from . import cbook as cbook
from .cm import ScalarMappable as ScalarMappable
from .colors import BoundaryNorm as BoundaryNorm
from .path import Path as Path
from .transforms import Bbox as Bbox, BboxBase as BboxBase, IdentityTransform as IdentityTransform, Transform as Transform, TransformedBbox as TransformedBbox, TransformedPatchPath as TransformedPatchPath, TransformedPath as TransformedPath
from _typeshed import Incomplete
from typing import NamedTuple

def allow_rasterization(draw): ...

class _XYPair(NamedTuple):
    x: Incomplete
    y: Incomplete

class _Unset: ...

class Artist:
    zorder: int
    def __init_subclass__(cls): ...
    stale_callback: Incomplete
    figure: Incomplete
    clipbox: Incomplete
    def __init__(self) -> None: ...
    def remove(self) -> None: ...
    def have_units(self): ...
    def convert_xunits(self, x): ...
    def convert_yunits(self, y): ...
    @property
    def axes(self): ...
    @axes.setter
    def axes(self, new_axes) -> None: ...
    @property
    def stale(self): ...
    @stale.setter
    def stale(self, val) -> None: ...
    def get_window_extent(self, renderer: Incomplete | None = None): ...
    def get_tightbbox(self, renderer: Incomplete | None = None): ...
    def add_callback(self, func): ...
    def remove_callback(self, oid) -> None: ...
    def pchanged(self) -> None: ...
    def is_transform_set(self): ...
    def set_transform(self, t) -> None: ...
    def get_transform(self): ...
    def get_children(self): ...
    def contains(self, mouseevent): ...
    def pickable(self): ...
    def pick(self, mouseevent) -> None: ...
    def set_picker(self, picker) -> None: ...
    def get_picker(self): ...
    def get_url(self): ...
    def set_url(self, url) -> None: ...
    def get_gid(self): ...
    def set_gid(self, gid) -> None: ...
    def get_snap(self): ...
    def set_snap(self, snap) -> None: ...
    def get_sketch_params(self): ...
    def set_sketch_params(self, scale: Incomplete | None = None, length: Incomplete | None = None, randomness: Incomplete | None = None) -> None: ...
    def set_path_effects(self, path_effects) -> None: ...
    def get_path_effects(self): ...
    def get_figure(self): ...
    def set_figure(self, fig) -> None: ...
    def set_clip_box(self, clipbox) -> None: ...
    def set_clip_path(self, path, transform: Incomplete | None = None) -> None: ...
    def get_alpha(self): ...
    def get_visible(self): ...
    def get_animated(self): ...
    def get_in_layout(self): ...
    def get_clip_on(self): ...
    def get_clip_box(self): ...
    def get_clip_path(self): ...
    def get_transformed_clip_path_and_affine(self): ...
    def set_clip_on(self, b) -> None: ...
    def get_rasterized(self): ...
    def set_rasterized(self, rasterized) -> None: ...
    def get_agg_filter(self): ...
    def set_agg_filter(self, filter_func) -> None: ...
    def draw(self, renderer) -> None: ...
    def set_alpha(self, alpha) -> None: ...
    def set_visible(self, b) -> None: ...
    def set_animated(self, b) -> None: ...
    def set_in_layout(self, in_layout) -> None: ...
    def get_label(self): ...
    def set_label(self, s) -> None: ...
    def get_zorder(self): ...
    def set_zorder(self, level) -> None: ...
    @property
    def sticky_edges(self): ...
    def update_from(self, other) -> None: ...
    def properties(self): ...
    def update(self, props): ...
    def set(self, **kwargs): ...
    def findobj(self, match: Incomplete | None = None, include_self: bool = True): ...
    def get_cursor_data(self, event) -> None: ...
    def format_cursor_data(self, data): ...
    def get_mouseover(self): ...
    def set_mouseover(self, mouseover) -> None: ...
    mouseover: Incomplete

class ArtistInspector:
    oorig: Incomplete
    o: Incomplete
    aliasd: Incomplete
    def __init__(self, o) -> None: ...
    def get_aliases(self): ...
    def get_valid_values(self, attr): ...
    def get_setters(self): ...
    @staticmethod
    def number_of_parameters(func): ...
    @staticmethod
    def is_alias(method): ...
    def aliased_name(self, s): ...
    def aliased_name_rest(self, s, target): ...
    def pprint_setters(self, prop: Incomplete | None = None, leadingspace: int = 2): ...
    def pprint_setters_rest(self, prop: Incomplete | None = None, leadingspace: int = 4): ...
    def properties(self): ...
    def pprint_getters(self): ...

def getp(obj, property: Incomplete | None = None): ...
get = getp

def setp(obj, *args, file: Incomplete | None = None, **kwargs): ...
def kwdoc(artist): ...
