from ._mathtext_data import uni2type1 as uni2type1
from _typeshed import Incomplete
from typing import NamedTuple

class CharMetrics(NamedTuple):
    width: Incomplete
    name: Incomplete
    bbox: Incomplete

class CompositePart(NamedTuple):
    name: Incomplete
    dx: Incomplete
    dy: Incomplete

class AFM:
    def __init__(self, fh) -> None: ...
    def get_bbox_char(self, c, isord: bool = False): ...
    def string_width_height(self, s): ...
    def get_str_bbox_and_descent(self, s): ...
    def get_str_bbox(self, s): ...
    def get_name_char(self, c, isord: bool = False): ...
    def get_width_char(self, c, isord: bool = False): ...
    def get_width_from_char_name(self, name): ...
    def get_height_char(self, c, isord: bool = False): ...
    def get_kern_dist(self, c1, c2): ...
    def get_kern_dist_from_name(self, name1, name2): ...
    def get_fontname(self): ...
    @property
    def postscript_name(self): ...
    def get_fullname(self): ...
    def get_familyname(self): ...
    @property
    def family_name(self): ...
    def get_weight(self): ...
    def get_angle(self): ...
    def get_capheight(self): ...
    def get_xheight(self): ...
    def get_underline_thickness(self): ...
    def get_horizontal_stem_width(self): ...
    def get_vertical_stem_width(self): ...
