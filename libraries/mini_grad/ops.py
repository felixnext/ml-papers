"""Defines more advanced ops"""

from typing import List

import numpy as np

from .base import Base
from .base_ops import Op
from .value import Value


class UniaryOp(Op):
    def __init__(self, value: Base, output: Base, name: str, op_name: str):
        if isinstance(value, Base):
            value = [value]
        if len(value) != 1:
            raise ValueError(f"Uniary operator expects 1 value, got {len(value)}")
        super().__init__(value, output, name, op_name)

    @classmethod
    def apply(cls, x: Base, y: Value = None, name: str = None) -> Value:
        if y is None:
            y = Value(0)
        op = cls([x], y, name)
        y._op = op
        y.forward()
        return y


class Tanh(UniaryOp):
    """Defines tanh operator"""

    def __init__(self, values: List[Base], output: Base, name: str):
        super().__init__(values, output, name, "tanh")

    def _forward(self, _pull: bool = True):
        return np.tanh(self._values[0].forward(_pull=_pull))

    def _backward(self):
        # distribute the gradient to the inputs
        self._values[0]._grad += self._output.grad * (
            1 - self._output.forward(False) ** 2
        )


def tanh(x: Base, name: str = None) -> Value:
    """Defines tanh function"""
    return Tanh.apply(x, name=name)


class Sigmoid(UniaryOp):
    """Defines sigmoid operator"""

    def __init__(self, values: List[Base], output: Base, name: str):
        super().__init__(values, output, name, "sigmoid")

    def _forward(self, _pull: bool = True):
        return 1 / (1 + np.exp(-self._values[0].forward(_pull=_pull)))

    def _backward(self):
        # distribute the gradient to the inputs
        self._values[0]._grad += self._output.grad * (
            self._output.forward(False) * (1 - self._output.forward(False))
        )


def sigmoid(x: Base, name: str = None) -> Value:
    """Defines sigmoid function"""
    return Sigmoid.apply(x, name=name)


def multi_apply(x: List[Value], op: Op):
    """Applies an op to a list of values"""
    if len(x) == 1:
        return x[0]
    if len(x) == 2:
        inner = x[1]
    if len(x) > 2:
        inner = multi_apply(x[1:], op)
    y = Value(0)
    _op = op([x[0], inner], y)
    y._op = _op
    y.forward()
    return y
