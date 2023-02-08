"""Basic Operations"""


from abc import abstractmethod
from typing import List, Set, Tuple, Any, Optional
from typing_extensions import Self

from graphviz import Digraph


class Base:
    """Base class for all the operations.

    Args:
        children (List[Self]): List of children nodes (used for plotting and reference)
        name (str, optional): Name of the node. Defaults to None.
    """

    def __init__(self, children: List[Self], name=None):
        self._name = name
        self._childs = children or []

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
        visited: Set[str] = set(),
    ) -> Tuple[Digraph, Set[str]]:
        """Internal implementation of the graph plotter.

        This only plots the current item as
        """
        raise NotImplementedError

    def _check_id(self, id: int, visited: Set[str]) -> Optional[str]:
        """Checks if the id is already visited and returns connection name"""
        if str(id) in visited:
            return str(id)
        elif f"cluster_{id}" in visited:
            return f"cluster_{id}"
        return str(id)
        # raise ValueError(f"ID {id} not found in visited nodes")

    def _over_level(self, level: int, max_level: int) -> bool:
        """Checks if the current level is over the max level"""
        return level >= max_level or len(self._childs) == 0

    def _iter_plot(
        self,
        dot: Digraph,
        visited: Set[str],
        level: int,
        max_level: int,
        rec_dot: Digraph = None,
    ) -> Tuple[Digraph, Set[int]]:
        """Iterates over the children of the current node and plots them."""
        # if max recursion is reached, plot the current node
        if self._over_level(level, max_level):
            if self.id not in visited:
                visited.add(str(self.id))
                dot, visited = self._plot(dot, visited)
        else:
            # check for recursive:
            dot = rec_dot or dot

            # otherwise create a subgraph with a label and dashed border
            sg_name = f"cluster_{self.id}"
            with dot.subgraph(name=sg_name) as sub:
                visited.add(sg_name)
                # create label, dashed border
                sub.attr(label=self.name, style="dashed")

                # plot all children in current subgraph
                for child in self._childs:
                    if str(child.id) not in visited:
                        visited.add(str(child.id))
                        # plot the child
                        sub, visited = child._iter_plot(
                            sub, visited, level + 1, max_level, rec_dot=dot
                        )

        return dot, visited

    def plot(self, max_level: int = 3) -> Digraph:
        """Plots the current branch of data.

        Args:
            max_level (int, optional): Maximum depth of the graph. Defaults to 3.

        Returns:
            Digraph: Graphviz object
        """
        dot = Digraph(format="svg", graph_attr={"rankdir": "LR"}, engine="dot")
        dot, _ = self._iter_plot(dot, set(), 0, max_level)
        return dot


class Derived(Base):
    def __init__(self, data: Any, children: List[Self], name=None):
        super().__init__(data, children, name)

    def base(self):
        """Should return the base value"""
        raise NotImplementedError
