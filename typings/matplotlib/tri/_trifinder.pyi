from matplotlib.tri import Triangulation as Triangulation

class TriFinder:
    def __init__(self, triangulation) -> None: ...
    def __call__(self, x, y) -> None: ...

class TrapezoidMapTriFinder(TriFinder):
    def __init__(self, triangulation) -> None: ...
    def __call__(self, x, y): ...