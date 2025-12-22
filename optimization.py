"""
Simplest possible grammar: a connected path where you can insert vertices within line segments, or remove the vertices.
To compute the cost gradient in the signed distance field,
"""

class Path:
    def __init__(self, vertices: list[tuple[float, float]]):
        self.vertices = vertices

class Map:
    pass

def compute_cost_gradient():
    pass