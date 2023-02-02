from abc import abstractclassmethod
from typing import List, Set, Tuple, Callable, Union, Type
from typing_extensions import Self
from graphviz import Digraph

from .base import Base
from .base_ops import Op, Add, Mul, Sub, Div


class Value(Base):
    """A singular value in the system.

    Args:
        data (float): The data of the value
        name (str, optional): Name of the value. Defaults to None.
        freeze (bool, optional): Whether the value should be frozen. Defaults to False.
        _op (Op, optional): The operation that generated the value. Defaults to None.
    """

    def __init__(
        self, data: float, name: str = None, freeze: bool = False, _op: Op = None
    ):
        super().__init__(data=data, children=[], name=name)
        self._grad = None
        self._op = _op
        self._freeze = freeze

    @property
    def grad(self):
        if self._grad is not None:
            return self._grad
        # in case of origin data set the grad to 0
        return 0

    def __repr__(self):
        return f"Value({self.name}, data={self.data:.4f}, grad={self.grad:.4f})"

    def forward(self, _pull: bool = True) -> float:
        # check if pulled
        if _pull:
            self._grad = None

            # check for operation to update the data
            if self._op is not None:
                self._op.forward(_pull=_pull)

        # return current data
        return self.data

    def backward(self):
        # in case this is first (set gradient to 1)
        if self._grad is None:
            self._grad = 1

        # iterate the backward pass (going through creational op)
        if self._op is not None:
            self._op.backward()

        # return personal gradient (will be 1)
        return self._grad

    def _plot(
        self, dot: Digraph, visited: Set[int] = set()
    ) -> Tuple[Digraph, Set[int]]:
        dot.node(
            str(self.id),
            label=f"{self.name or ''} | v={self.data:.4f} | g={self.grad:.4f}",
            shape="record",
        )
        return dot, visited

    def _iter_plot(
        self, dot: Digraph, visited: Set[int], level: int, max_level: int
    ) -> Tuple[Digraph, Set[int]]:
        # call super function
        dot, visited = super()._iter_plot(dot, visited, level, max_level)

        # check for previous items
        if self._op is not None:
            dot, visited = self._op._iter_plot(dot, visited, level, max_level)
            dot.edge(str(self._op.id), str(self.id))

        return dot, visited

    @abstractclassmethod
    def create(
        cls,
        shape: Tuple[int, ...],
        init: Callable[[Tuple[int, ...]], float] = None,
        name: str = None,
    ) -> List[Self]:
        """Abstract method to create a multi-shaped Value object"""
        raise NotImplementedError

    def _elevate(self, other: Union[Self, float]) -> Self:
        if isinstance(other, Value):
            return other
        return Value(other, name=str(other))

    def _apply_op(self, other: Self, op: Type[Op]) -> Self:
        other = self._elevate(other)
        out = Value(0, name=f"{self.name} {op.name} {other.name}")
        op = op([self, other], out, name=f"{op.__name__}({self.name}, {other.name})")
        out._op = op
        out.forward()
        return out

    def __add__(self, other: Self) -> Self:
        return self._apply_op(other, Add)

    def __mul__(self, other: Self) -> Self:
        return self._apply_op(other, Mul)

    def __sub__(self, other: Self) -> Self:
        return self._apply_op(other, Sub)

    def __truediv__(self, other: Self) -> Self:
        return self._apply_op(other, Div)


class Parameter(Value):
    def __init__(self, data: float, name: str = None):
        super().__init__(data, name=name, freeze=True)
        self._step = 0

    @property
    def step(self):
        return self._step

    def prepare(self, data: float):
        """Should be called after each iteration to update the parameters"""
        self._step += 1
        self._data = data

    def _plot(
        self, dot: Digraph, visited: Set[int] = set()
    ) -> Tuple[Digraph, Set[int]]:
        dot.node(
            str(self.id),
            label=f"{self.name} | v={self.data:.4f}",
            shape="record",
            color="blue",
            # attrs={"color": "blue", "fontstyle": "italic"},
        )
        return dot, visited


class Variable(Value):
    def __init__(self, data: float, name: str = None, freeze: bool = False):
        super().__init__(data, name=name, freeze=freeze)

    def _plot(
        self, dot: Digraph, visited: Set[int] = set()
    ) -> Tuple[Digraph, Set[int]]:
        dot.node(
            str(self.id),
            label=f"{self.name or ''} | v={self.data:.4f} | g={self.grad:.4f}",
            shape="record",
            # attrs={"color": "orange", "fontstyle": "bold"},
        )
        return dot, visited
