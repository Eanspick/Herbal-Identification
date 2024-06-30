from _typeshed import Incomplete
from collections.abc import Mapping
from matplotlib import cbook as cbook, colors as colors, scale as scale
from matplotlib._cm import datad as datad

class ColormapRegistry(Mapping):
    def __init__(self, cmaps) -> None: ...
    def __getitem__(self, item): ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def __call__(self): ...
    def register(self, cmap, *, name: Incomplete | None = None, force: bool = False) -> None: ...
    def unregister(self, name) -> None: ...
    def get_cmap(self, cmap): ...

def register_cmap(name: Incomplete | None = None, cmap: Incomplete | None = None, *, override_builtin: bool = False) -> None: ...

get_cmap: Incomplete

def unregister_cmap(name): ...

class ScalarMappable:
    cmap: Incomplete
    colorbar: Incomplete
    callbacks: Incomplete
    def __init__(self, norm: Incomplete | None = None, cmap: Incomplete | None = None) -> None: ...
    def to_rgba(self, x, alpha: Incomplete | None = None, bytes: bool = False, norm: bool = True): ...
    def set_array(self, A) -> None: ...
    def get_array(self): ...
    def get_cmap(self): ...
    def get_clim(self): ...
    def set_clim(self, vmin: Incomplete | None = None, vmax: Incomplete | None = None) -> None: ...
    def get_alpha(self): ...
    def set_cmap(self, cmap) -> None: ...
    @property
    def norm(self): ...
    @norm.setter
    def norm(self, norm) -> None: ...
    def set_norm(self, norm) -> None: ...
    def autoscale(self) -> None: ...
    def autoscale_None(self) -> None: ...
    stale: bool
    def changed(self) -> None: ...
