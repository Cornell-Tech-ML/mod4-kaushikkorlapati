"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, List, Iterable


# Mathematical functions:
def mul(a: float, b: float) -> float:
    """Multiply two float values.

    Args:
    ----
    a : float
        The first number.
    b : float
        The second number.

    Returns:
    -------
    float
        The product of `a` and `b`.

    """
    return a * b


def id(a: float) -> float:
    """Identity function that returns its input.

    Args:
    ----
    a : float
        Input number.

    Returns:
    -------
    float
        The same input `a`.

    """
    return a


def add(a: float, b: float) -> float:
    """Add two float values.

    Args:
    ----
    a : float
        The first number.
    b : float
        The second number.

    Returns:
    -------
    float
        The sum of `a` and `b`.

    """
    return a + b


def neg(a: float) -> float:
    """Negate a float value.

    Args:
    ----
    a : float
        Input number.

    Returns:
    -------
    float
        The negation of `a`.

    """
    return -1 * a


def lt(a: float, b: float) -> bool:
    """Determine if one float is less than another.

    Args:
    ----
    a : float
        The first number.
    b : float
        The second number.

    Returns:
    -------
    bool
        True if `a` is less than `b`, False otherwise.

    """
    if a < b:
        return True
    else:
        return False


def eq(a: float, b: float) -> bool:
    """Determine if two floats are equal.

    Args:
    ----
    a : float
        The first number.
    b : float
        The second number.

    Returns:
    -------
    bool
        True if `a` equals `b`, False otherwise.

    """
    if a == b:
        return True
    else:
        return False


def max(a: float, b: float) -> float:
    """Return the maximum of two floats.

    Args:
    ----
    a : float
        The first number.
    b : float
        The second number.

    Returns:
    -------
    float
        The maximum of `a` and `b`.

    """
    if a < b:
        return b
    else:
        return a


def is_close(x: float, y: float) -> bool:
    """Check if two floats are close to each other within a small tolerance.

    Args:
    ----
    x : float
        The first number.
    y : float
        The second number.

    Returns:
    -------
    bool
        True if `x` and `y` are within `1e-2` of each other, False otherwise.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the sigmoid function.

    Args:
    ----
    x : float
        Input number.

    Returns:
    -------
    float
        The sigmoid of `x`.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU (Rectified Linear Unit) function.

    Args:
    ----
    x : float
        Input number.

    Returns:
    -------
    float
        `x` if `x` > 0, otherwise 0.

    """
    return x if x > 0 else 0


def log(x: float) -> float:
    """Compute the logarithm of a number with a specified base.

    Args:
    ----
    x : float
        The number to take the logarithm of.
    base : float, optional
        The base of the logarithm, default is the natural logarithm (e).

    Returns:
    -------
    float
        The logarithm of `x` to the given base.

    """
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential function.

    Args:
    ----
    x : float
        Input number.

    Returns:
    -------
    float
        The exponential of `x`.

    """
    return math.exp(x)


def log_back(x: float, k: float) -> float:
    """Compute the gradient of the log function.

    Args:
    ----
    x : float
        Input number.
    k : float
        The gradient value to backpropagate.

    Returns:
    -------
    float
        The gradient of the log function at `x`.

    Raises:
    ------
    ValueError
        If `x` is less than or equal to 0.

    """
    if x <= 0:
        raise ValueError("Math domain error: log(x) is undefined for x <= 0.")
    return k / x


def inv(x: float) -> float:
    """Compute the reciprocal (inverse) of a number.

    Args:
    ----
    x : float
        Input number.

    Returns:
    -------
    float
        The reciprocal of `x`.

    Raises:
    ------
    ValueError
        If `x` is zero.

    """
    if x == 0:
        raise ValueError("Cannot compute reciprocal of zero.")
    return 1 / x


def inv_back(x: float, k: float) -> float:
    """Compute the gradient of the inverse function.

    Args:
    ----
    x : float
        Input number.
    k : float
        The gradient value to backpropagate.

    Returns:
    -------
    float
        The gradient of the inverse function at `x`.

    Raises:
    ------
    ValueError
        If `x` is zero.

    """
    if x == 0:
        raise ValueError("Cannot compute derivative at x = 0.")
    return -k / (x**2)


def relu_back(x: float, k: float) -> float:
    """Compute the gradient of the ReLU function.

    Args:
    ----
    x : float
        Input number.
    k : float
        The gradient value to backpropagate.

    Returns:
    -------
    float
        `k` if `x` > 0, otherwise 0.

    """
    return k if x > 0 else 0


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce


def map(func: Callable[[float], float], lst: List[float]) -> List[float]:
    """Apply function `func` to each element of `lst`."""
    return [func(x) for x in lst]


def zipWith(
    func: Callable[[float, float], float], lst1: List[float], lst2: List[float]
) -> List[float]:
    """Combine `lst1` and `lst2` element-wise using function `func`."""
    return [func(x, y) for x, y in zip(lst1, lst2)]


def reduce(
    func: Callable[[float, float], float], lst: Iterable[float], initial: float
) -> float:
    """Reduce `lst` to a single value using binary function `func` with an initial value."""
    result = initial
    for x in lst:
        result = func(result, x)
    return result


#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def negList(lst: List[float]) -> List[float]:
    """Negate each element of the list."""
    return map(lambda x: -x, lst)


def addLists(lst1: List[float], lst2: List[float]) -> List[float]:
    """Add corresponding elements of two lists."""
    return zipWith(lambda x, y: x + y, lst1, lst2)


def sum(lst: List[float]) -> float:
    """Sum all elements in the list."""
    return reduce(lambda x, y: x + y, lst, 0.0)


def prod(lst: Iterable[float]) -> float:
    """Compute the product of all elements in the list."""
    return reduce(lambda x, y: x * y, lst, 1.0)
