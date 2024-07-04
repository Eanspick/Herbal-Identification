from _typeshed import Incomplete
from matplotlib.tri._triangulation import Triangulation as Triangulation

class TriRefiner:
    def __init__(self, triangulation) -> None: ...

class UniformTriRefiner(TriRefiner):
    def __init__(self, triangulation) -> None: ...
    def refine_triangulation(self, return_tri_index: bool = False, subdiv: int = 3): ...
    def refine_field(self, z, triinterpolator: Incomplete | None = None, subdiv: int = 3): ...