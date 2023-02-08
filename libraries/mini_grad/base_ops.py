from abc import abstractmethod
import logging
from typing import List, Set, Tuple

from graphviz import Digraph
import numpy as np

from .base import Base


class Op(Base):
    """Defines basic operator"""

    def __init__(self, inputs: List[Base], output: Base, name: str, op_name: str):
        super().__init__(children=[], name=name)

        # check inputs
        for v in inputs:
            if not isinstance(v, Base):
                raise ValueError(f"Input {v} is not a Base")
            if not hasattr(v, "data") or not hasattr(v, "_grad"):
                raise ValueError(f"Input {v} is not a Differentiable")
        # check output
        if (
            not hasattr(output, "data")
            or not hasattr(output, "_grad")
            or not hasattr(output, "_data")
        ):
            raise ValueError(f"Output {output} is not a Differentiable")

        # apply items
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
        self, dot: Digraph, visited: Set[str] = set()
    ) -> Tuple[Digraph, Set[str]]:
        dot.node(str(self.id), label=self.op_name, shape="circle")
        return dot, visited

    def _iter_plot(
        self,
        dot: Digraph,
        visited: Set[str],
        level: int,
        max_level: int,
        rec_dot: Digraph = None,
    ) -> Tuple[Digraph, Set[str]]:
        # call super function
        dot, visited = super()._iter_plot(dot, visited, level, max_level, rec_dot)

        # plot current edges
        for v in self._values:
            # FIXME: handle out of scope
            # dot = rec_dot or dot
            dot, visited = v._iter_plot(dot, visited, level, max_level, rec_dot)
            dot.edge(self._check_id(v.id, visited), self._check_id(self.id, visited))

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
