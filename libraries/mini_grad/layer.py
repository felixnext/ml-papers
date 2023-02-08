from typing import List, Set, Tuple

from graphviz import Digraph

from .value import Differentiable
from .base_ops import Op
from .ops import UniaryOp
from .neuron import Neuron


class FCLayer(Differentiable):
    def __init__(
        self,
        inputs: List[Differentiable],
        size: int,
        activation: UniaryOp,
        name: str = None,
        freeze: bool = False,
    ):
        neurons = [
            Neuron(inputs, activation=activation, name=f"{name}.n{i}")
            for i in range(size)
        ]
        self._inputs = inputs
        super().__init__(children=neurons, name=name, freeze=freeze, _op=None)

    @property
    def data(self) -> List[float]:
        return [n.data for n in self._childs]

    @property
    def outs(self) -> List[Differentiable]:
        return self._childs

    def __repr__(self):
        return f"FCLayer({self.name}, neurons={len(self._childs)}, grad={self.grad})"

    def forward(self, _pull: bool = True) -> List[float]:
        return [n.forward(_pull=_pull) for n in self._childs]

    def backward(self):
        for n in self._childs:
            n.backward()

    def _plot(
        self, dot: Digraph, visited: Set[str] = set()
    ) -> Tuple[Digraph, Set[int]]:
        # double lined node with the name and shape
        dot.node(
            str(self.id),
            label=f"{self.name} | d={self.data} | ins={len(self._childs)}",
            shape="record",
            style="filled",
            fillcolor="lightgreen",
        )

        return dot, visited

    def _iter_plot(
        self,
        dot: Digraph,
        visited: Set[str],
        level: int,
        max_level: int,
        rec_dot: Digraph = None,
    ) -> Tuple[Digraph, Set[int]]:
        dot, visited = super()._iter_plot(dot, visited, level, max_level, rec_dot)

        # iterate inputs
        for inp in self._inputs:
            dot = rec_dot or dot
            dot, visited = inp._iter_plot(dot, visited, level, max_level, rec_dot)

        # add edges
        if self._over_level(level, max_level):
            for inp in self._inputs:
                dot.edge(
                    self._check_id(inp.id, visited), self._check_id(self.id, visited)
                )

        return dot, visited
