"""Example file to test the system"""

from .base import Parameter, Value, Op, UniaryOp


def main():
    """Main function"""
    # generate a simple example
    a = Value(2.0, name="a")
    b = Value(3.0, name="b")
    c = a + b
    c.name = "c"

    res = c.forward()
    print(res)

    # generate backward

    # plot the graph
