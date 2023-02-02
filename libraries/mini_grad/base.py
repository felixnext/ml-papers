"""Basic Operations"""


from abc import abstractmethod
from typing import List, Set, Tuple, Any
from typing_extensions import Self

from graphviz import Digraph


class Base:
    """Base class for all the operations.

    Args:
        children (List[Self]): List of children nodes (used for plotting and reference)
        name (str, optional): Name of the node. Defaults to None.
    """

    def __init__(self, data: Any, children: List[Self], name=None):
        self._name = name
        self._childs = children or []
        self._data = data

    @property
    def data(self):
        return self._data

    @property
    def id(self):
        return id(self)

    @property
    def name(self):
        if self._name is not None:
            return self._name
        return f"{self.__class__.__name__}({self.id})"

    @name.setter
    def name(self, value):
        self._name = value

    @abstractmethod
    def forward(self, _pull: bool = True):
        """Evaluates the forward pass"""
        raise NotImplementedError

    @abstractmethod
    def backward(self):
        """Evaluates the backward pass"""
        raise NotImplementedError

    @abstractmethod
    def _plot(
        self,
        dot: Digraph,
        visited: Set[int] = set(),
    ) -> Tuple[Digraph, Set[int]]:
        """Internal implementation of the graph plotter.

        This only plots the current item as
        """
        raise NotImplementedError

    def _iter_plot(
        self, dot: Digraph, visited: Set[int], level: int, max_level: int
    ) -> Tuple[Digraph, Set[int]]:
        """Iterates over the children of the current node and plots them."""
        # if max recursion is reached, plot the current node
        if level >= max_level or len(self._childs) == 0:
            if self.id not in visited:
                visited.add(self.id)
                dot, visited = self._plot(dot, visited)
        else:
            # otherwise create a subgraph with a label and dashed border
            with dot.subgraph(name=f"cluster_{self.id}") as sub:
                sub.attr(label=self.name, style="dashed")

                # plot all children in current subgraph
                for child in self._childs:
                    if child.id not in visited:
                        visited.add(child.id)
                        # plot the child
                        sub, visited = child._iter_plot(
                            sub, visited, level + 1, max_level
                        )

        return dot, visited

    def plot(self, max_level: int = 3) -> Digraph:
        """Plots the current branch of data.

        Args:
            max_level (int, optional): Maximum depth of the graph. Defaults to 3.

        Returns:
            Digraph: Graphviz object
        """
        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})
        dot, _ = self._iter_plot(dot, set(), 0, max_level)
        return dot


class Derived(Base):
    def __init__(self, data: Any, children: List[Self], name=None):
        super().__init__(data, children, name)

    def base(self):
        """Should return the base value"""
        raise NotImplementedError
