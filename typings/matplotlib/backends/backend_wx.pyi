import wx
from _typeshed import Incomplete
from matplotlib import backend_tools as backend_tools, cbook as cbook
from matplotlib._pylab_helpers import Gcf as Gcf
from matplotlib.backend_bases import CloseEvent as CloseEvent, FigureCanvasBase as FigureCanvasBase, FigureManagerBase as FigureManagerBase, GraphicsContextBase as GraphicsContextBase, KeyEvent as KeyEvent, LocationEvent as LocationEvent, MouseButton as MouseButton, MouseEvent as MouseEvent, NavigationToolbar2 as NavigationToolbar2, RendererBase as RendererBase, ResizeEvent as ResizeEvent, TimerBase as TimerBase, ToolContainerBase as ToolContainerBase, _Backend, cursors as cursors
from matplotlib.path import Path as Path
from matplotlib.transforms import Affine2D as Affine2D

PIXELS_PER_INCH: int

class TimerWx(TimerBase):
    def __init__(self, *args, **kwargs) -> None: ...

class RendererWx(RendererBase):
    fontweights: Incomplete
    fontangles: Incomplete
    fontnames: Incomplete
    width: Incomplete
    height: Incomplete
    bitmap: Incomplete
    fontd: Incomplete
    dpi: Incomplete
    gc: Incomplete
    def __init__(self, bitmap, dpi) -> None: ...
    def flipy(self): ...
    def get_text_width_height_descent(self, s, prop, ismath): ...
    def get_canvas_width_height(self): ...
    def handle_clip_rectangle(self, gc) -> None: ...
    @staticmethod
    def convert_path(gfx_ctx, path, transform): ...
    def draw_path(self, gc, path, transform, rgbFace: Incomplete | None = None) -> None: ...
    def draw_image(self, gc, x, y, im) -> None: ...
    def draw_text(self, gc, x, y, s, prop, angle, ismath: bool = False, mtext: Incomplete | None = None) -> None: ...
    def new_gc(self): ...
    def get_wx_font(self, s, prop): ...
    def points_to_pixels(self, points): ...

class GraphicsContextWx(GraphicsContextBase):
    bitmap: Incomplete
    dc: Incomplete
    gfx_ctx: Incomplete
    renderer: Incomplete
    def __init__(self, bitmap, renderer) -> None: ...
    IsSelected: bool
    def select(self) -> None: ...
    def unselect(self) -> None: ...
    def set_foreground(self, fg, isRGBA: Incomplete | None = None) -> None: ...
    def set_linewidth(self, w) -> None: ...
    def set_capstyle(self, cs) -> None: ...
    def set_joinstyle(self, js) -> None: ...
    def get_wxcolour(self, color): ...

class _FigureCanvasWxBase(FigureCanvasBase, wx.Panel):
    required_interactive_framework: str
    manager_class: Incomplete
    keyvald: Incomplete
    bitmap: Incomplete
    def __init__(self, parent, id, figure: Incomplete | None = None) -> None: ...
    def Copy_to_Clipboard(self, event: Incomplete | None = None) -> None: ...
    def draw_idle(self) -> None: ...
    def flush_events(self) -> None: ...
    def start_event_loop(self, timeout: int = 0) -> None: ...
    def stop_event_loop(self, event: Incomplete | None = None) -> None: ...
    def gui_repaint(self, drawDC: Incomplete | None = None) -> None: ...
    filetypes: Incomplete
    def set_cursor(self, cursor) -> None: ...

class FigureCanvasWx(_FigureCanvasWxBase):
    renderer: Incomplete
    def draw(self, drawDC: Incomplete | None = None) -> None: ...
    print_bmp: Incomplete
    print_jpeg: Incomplete
    print_jpg: Incomplete
    print_pcx: Incomplete
    print_png: Incomplete
    print_tiff: Incomplete
    print_tif: Incomplete
    print_xpm: Incomplete

class FigureFrameWx(wx.Frame):
    canvas: Incomplete
    def __init__(self, num, fig, *, canvas_class) -> None: ...

class FigureManagerWx(FigureManagerBase):
    frame: Incomplete
    def __init__(self, canvas, num, frame) -> None: ...
    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num): ...
    @classmethod
    def start_main_loop(cls) -> None: ...
    def show(self) -> None: ...
    def destroy(self, *args) -> None: ...
    def full_screen_toggle(self) -> None: ...
    def get_window_title(self): ...
    def set_window_title(self, title) -> None: ...
    def resize(self, width, height) -> None: ...

class NavigationToolbar2Wx(NavigationToolbar2, wx.ToolBar):
    wx_ids: Incomplete
    def __init__(self, canvas, coordinates: bool = True, *, style=...) -> None: ...
    def zoom(self, *args) -> None: ...
    def pan(self, *args) -> None: ...
    def save_figure(self, *args) -> None: ...
    def draw_rubberband(self, event, x0, y0, x1, y1) -> None: ...
    def remove_rubberband(self) -> None: ...
    def set_message(self, s) -> None: ...
    def set_history_buttons(self) -> None: ...

class ToolbarWx(ToolContainerBase, wx.ToolBar):
    def __init__(self, toolmanager, parent: Incomplete | None = None, style=...) -> None: ...
    def add_toolitem(self, name, group, position, image_file, description, toggle) -> None: ...
    def toggle_toolitem(self, name, toggled) -> None: ...
    def remove_toolitem(self, name) -> None: ...
    def set_message(self, s) -> None: ...

class ConfigureSubplotsWx(backend_tools.ConfigureSubplotsBase):
    def trigger(self, *args) -> None: ...

class SaveFigureWx(backend_tools.SaveFigureBase):
    def trigger(self, *args) -> None: ...

class RubberbandWx(backend_tools.RubberbandBase):
    def draw_rubberband(self, x0, y0, x1, y1) -> None: ...
    def remove_rubberband(self) -> None: ...

class _HelpDialog(wx.Dialog):
    headers: Incomplete
    widths: Incomplete
    def __init__(self, parent, help_entries) -> None: ...
    @classmethod
    def show(cls, parent, help_entries) -> None: ...

class HelpWx(backend_tools.ToolHelpBase):
    def trigger(self, *args) -> None: ...

class ToolCopyToClipboardWx(backend_tools.ToolCopyToClipboardBase):
    def trigger(self, *args, **kwargs) -> None: ...

class _BackendWx(_Backend):
    FigureCanvas = FigureCanvasWx
    FigureManager = FigureManagerWx
    mainloop: Incomplete
