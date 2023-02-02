from abc import abstractmethod
import logging
from typing import List, Set, Tuple

from graphviz import Digraph
import numpy as np

from .base import Base


class Op(Base):
    """Defines basic operator"""

    def __init__(self, inputs: List[Base], output: Base, name: str, op_name: str):
        super().__init__(data=None, children=[], name=name)
        self._values = inputs
        self._output = output
        self._op_name = op_name

    @property
    def op_name(self):
        return self._op_name

    def __repr__(self):
        return f"Op({self.name} dim={len(self._values)} op={self.op_name})"

    @abstractmethod
    def _forward(self, _pull: bool = False):
        raise NotImplementedError

    def forward(self, _pull: bool = True):
        res = self._forward(_pull=_pull)
        self._output._data = res
        return res

    @abstractmethod
    def _backward(self):
        raise NotImplementedError

    def backward(self):
        # prepare gradients of the values
        for v in self._values:
            if not hasattr(v, "_grad"):
                logging.warning(f"Value {v} has no grad")
                continue
            if v._grad is None:
                v._grad = 0

        # iterate the backward pass
        self._backward()

        # execute sub operations
        for v in self._values:
            v.backward()

    def _plot(
        self, dot: Digraph, visited: Set[int] = set()
    ) -> Tuple[Digraph, Set[int]]:
        dot.node(str(self.id), label=self.op_name, shape="circle")
        return dot, visited

    def _iter_plot(
        self, dot: Digraph, visited: Set[int], level: int, max_level: int
    ) -> Tuple[Digraph, Set[int]]:
        # call super function
        dot, visited = super()._iter_plot(dot, visited, level, max_level)

        # plot current edges
        for v in self._values:
            dot, visited = v._iter_plot(dot, visited, level, max_level)
            dot.edge(str(v.id), str(self.id))

        return dot, visited


class Add(Op):
    """Defines addition operator"""

    def __init__(self, values: List[Base], output: Base, name: str = None):
        if len(values) != 2:
            raise ValueError(f"Add operator expects 2 values, got {len(values)}")
        super().__init__(values, output, name, "+")

    def _forward(self, _pull: bool = True):
        return sum(v.forward(_pull=_pull) for v in self._values)

    def _backward(self):
        # distribute the gradient to the inputs
        for v in self._values:
            v._grad += self._output.grad


class Mul(Op):
    """Defines multiplication operator"""

    def __init__(self, values: List[Base], output: Base, name: str = None):
        if len(values) != 2:
            raise ValueError(f"Mul operator expects 2 values, got {len(values)}")
        super().__init__(values, output, name, "*")

    def _forward(self, _pull: bool = True):
        return np.prod([v.forward(_pull=_pull) for v in self._values])

    def _backward(self):
        # distribute the gradient to the inputs
        self._values[0]._grad += self._output.grad * self._values[1].forward(
            _pull=False
        )
        self._values[1]._grad += self._output.grad * self._values[0].forward(
            _pull=False
        )


class Sub(Op):
    """Defines subtraction operator"""

    def __init__(self, values: List[Base], output: Base, name: str = None):
        if len(values) != 2:
            raise ValueError(f"Sub operator expects 2 values, got {len(values)}")
        super().__init__(values, output, name, "-")

    def _forward(self, _pull: bool = True):
        return self._values[0].forward(_pull=_pull) - self._values[1].forward(
            _pull=_pull
        )

    def _backward(self):
        # distribute the gradient to the inputs
        self._values[0]._grad += self._output.grad
        self._values[1]._grad -= self._output.grad


class Div(Op):
    """Defines division operator"""

    def __init__(self, values: List[Base], output: Base, name: str = None):
        if len(values) != 2:
            raise ValueError(f"Div operator expects 2 values, got {len(values)}")
        super().__init__(values, output, name, "/")

    def _forward(self, _pull: bool = True):
        return self._values[0].forward(_pull=_pull) / self._values[1].forward(
            _pull=_pull
        )

    def _backward(self):
        # distribute the gradient to the inputs
        self._values[0]._grad += self._output.grad / self._values[1].forward(
            _pull=False
        )
        self._values[1]._grad -= (
            self._output.grad * self._values[0].forward(_pull=False)
        ) / self._values[1].forward(_pull=False) ** 2
