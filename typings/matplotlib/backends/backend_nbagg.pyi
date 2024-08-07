from .backend_webagg_core import FigureCanvasWebAggCore as FigureCanvasWebAggCore, FigureManagerWebAgg as FigureManagerWebAgg, NavigationToolbar2WebAgg as NavigationToolbar2WebAgg, TimerAsyncio as TimerAsyncio, TimerTornado as TimerTornado
from _typeshed import Incomplete
from matplotlib import is_interactive as is_interactive
from matplotlib._pylab_helpers import Gcf as Gcf
from matplotlib.backend_bases import CloseEvent as CloseEvent, NavigationToolbar2 as NavigationToolbar2, _Backend

def connection_info(): ...

class NavigationIPy(NavigationToolbar2WebAgg):
    toolitems: Incomplete

class FigureManagerNbAgg(FigureManagerWebAgg):
    ToolbarCls = NavigationIPy
    def __init__(self, canvas, num) -> None: ...
    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num): ...
    def display_js(self) -> None: ...
    def show(self) -> None: ...
    def reshow(self) -> None: ...
    @property
    def connected(self): ...
    @classmethod
    def get_javascript(cls, stream: Incomplete | None = None): ...
    def destroy(self) -> None: ...
    web_sockets: Incomplete
    def clearup_closed(self) -> None: ...
    def remove_comm(self, comm_id) -> None: ...

class FigureCanvasNbAgg(FigureCanvasWebAggCore):
    manager_class = FigureManagerNbAgg

class CommSocket:
    supports_binary: Incomplete
    manager: Incomplete
    uuid: Incomplete
    comm: Incomplete
    def __init__(self, manager) -> None: ...
    def is_open(self): ...
    def on_close(self) -> None: ...
    def send_json(self, content) -> None: ...
    def send_binary(self, blob) -> None: ...
    def on_message(self, message) -> None: ...

class _BackendNbAgg(_Backend):
    FigureCanvas = FigureCanvasNbAgg
    FigureManager = FigureManagerNbAgg
