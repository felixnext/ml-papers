from typing import List, Tuple, Set, Union, Type
from typing_extensions import Self

import numpy as np
from graphviz import Digraph

from .base import Base
from .value import Value, Variable
from .base_ops import Add, Op
from .ops import multi_apply, UniaryOp


class Neuron(Base):
    def __init__(self, inputs: List[Value], activation: UniaryOp, name: str = None):
        self._inputs = inputs
        self._weights = [
            Variable(np.random.randn(), name=f"{name}.w{i}") for i in range(len(inputs))
        ]
        self._bias = Variable(np.random.randn(), name=f"{name}.b")
        sum_list = [w * x for w, x in zip(self._weights, inputs)]
        summed = multi_apply(sum_list, Add) + self._bias
        self._out = activation.apply(summed)
        super().__init__(
            data=None, children=self._weights + [self._bias, self._out], name=name
        )

    @property
    def data(self):
        return self._out.data

    @property
    def _grad(self):
        return self._out._grad

    @_grad.setter
    def _grad(self, value):
        self._out._grad = value

    def forward(self, _pull: bool = True) -> float:
        return self._out.forward(_pull=_pull)

    def backward(self):
        self._out.backward()

    def _elevate(self, other: Union[Self, float]) -> Self:
        if not isinstance(other, Base):
            return Value(other, name=str(other))
        return other

    def _apply_op(self, other: Self, op: Type[Op]) -> Self:
        other = self._elevate(other)
        out = Value(0, name=f"{self.name} {op.name} {other.name}")
        op = op([self, other], out, name=f"{op.__name__}({self.name}, {other.name})")
        out._op = op
        out.forward()
        return out

    def __add__(self, other):
        return self._apply_op(other, Add)

    def _plot(
        self, dot: Digraph, visited: Set[int] = set()
    ) -> Tuple[Digraph, Set[int]]:
        # double lined node with the name and shape
        dot.node(
            str(self.id),
            label=f"{self.name} | d={self.data} | ins={len(self._weights)}",
            shape="record",
            style="filled",
            fillcolor="lightblue",
        )

        # add input edges
        for inp in self._inputs:
            dot.edge(str(inp.id), str(self.id))
        return dot, visited

    def _iter_plot(
        self, dot: Digraph, visited: Set[int], level: int, max_level: int
    ) -> Tuple[Digraph, Set[int]]:
        dot, visited = super()._iter_plot(dot, visited, level, max_level)

        # iterate inputs
        for inp in self._inputs:
            dot, visited = inp._iter_plot(dot, visited, level, max_level)

        return dot, visited
