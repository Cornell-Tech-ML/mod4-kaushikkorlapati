from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

from dataclasses import field
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import (
    EQ,
    LT,
    Add,
    Exp,
    Inv,
    Log,
    Mul,
    Neg,
    ReLU,
    ScalarFunction,
    Sigmoid,
)

ScalarLike = Union[float, int, "Scalar"]


@dataclass
class ScalarHistory:
    """`ScalarHistory` stores the history of `Function` operations that was
    used to construct the current Variable.

    Attributes
    ----------
        last_fn : The last Function that was called.
        ctx : The context for that Function.
        inputs : The inputs that were given when `last_fn.forward` was called.

    """

    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()


# ## Task 1.2 and 1.4
# Scalar Forward and Backward

_var_count = 0


@dataclass
class Scalar:
    """A reimplementation of scalar values for autodifferentiation
    tracking. Scalar Variables behave as close as possible to standard
    Python numbers while also tracking the operations that led to the
    number's creation. They can only be manipulated by
    `ScalarFunction`.

    Attributes
    ----------
        data : The underlying value stored in the Scalar.
        history : Optional history of the function and inputs used to
                  create the scalar.
        derivative : The derivative of the scalar during backpropagation.
        name : Unique name identifier for this scalar.
        unique_id : Unique identifier for this scalar instance.

    """

    data: float
    history: Optional[ScalarHistory] = field(default_factory=ScalarHistory)
    derivative: Optional[float] = None
    name: str = field(default="")
    unique_id: int = field(default=0)

    def __post_init__(self):
        """Post-initialization to assign a unique ID and ensure the data is a float."""
        global _var_count
        _var_count += 1
        object.__setattr__(self, "unique_id", _var_count)
        object.__setattr__(self, "name", str(self.unique_id))
        object.__setattr__(self, "data", float(self.data))

    def __hash__(self):
        """Return a hash of the scalar data."""
        return hash(self.data)

    def __repr__(self) -> str:
        """Return a string representation of the scalar."""
        return f"Scalar({self.data})"

    def __mul__(self, b: ScalarLike) -> Scalar:
        """Perform element-wise multiplication between two scalars."""
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        """Perform element-wise division between two scalars."""
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        """Perform element-wise division in reverse order."""
        return Mul.apply(b, Inv.apply(self))

    def __bool__(self) -> bool:
        """Return the boolean representation of the scalar data."""
        return bool(self.data)

    def __radd__(self, b: ScalarLike) -> Scalar:
        """Perform element-wise addition between scalars."""
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        """Perform element-wise multiplication between scalars in reverse order."""
        return self * b

    def __lt__(self, other: ScalarLike) -> Scalar:
        """Return the result of less-than comparison with another scalar."""
        return LT.apply(self, other)

    def __gt__(self, other: ScalarLike) -> Scalar:
        """Return the result of greater-than comparison with another scalar."""
        return LT.apply(other, self)

    def __sub__(self, other: ScalarLike) -> Scalar:
        """Return the result of subtracting another scalar."""
        return Add.apply(self, Neg.apply(other))

    def __neg__(self) -> Scalar:
        """Return the negative of the scalar."""
        return Neg.apply(self)

    def __add__(self, other: ScalarLike) -> Scalar:
        """Return the result of adding another scalar."""
        return Add.apply(self, other)

    def __eq__(self, other: ScalarLike) -> Scalar:
        """Return the result of equality comparison with another scalar."""
        return EQ.apply(self, other)

    def log(self) -> Scalar:
        """Return the logarithm of the scalar."""
        return Log.apply(self)

    def exp(self) -> Scalar:
        """Return the exponential of the scalar."""
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        """Return the sigmoid activation of the scalar."""
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        """Return the ReLU activation of the scalar."""
        return ReLU.apply(self)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate a value `x` to the derivative of this scalar.

        This should be called only during autodifferentiation on leaf variables.

        Args:
        ----
            x : The value to accumulate into the derivative.

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.__setattr__("derivative", 0.0)
        self.__setattr__("derivative", self.derivative + x)

    def is_leaf(self) -> bool:
        """Return True if this variable was created by the user (i.e., has no `last_fn`)."""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Return True if the scalar is constant (i.e., no history or function applied)."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Return the input variables (parents) used to create this scalar."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule for backpropagation, returning a list of (parent, gradient) pairs.

        Args:
        ----
            d_output : The derivative output at this scalar.

        Returns:
        -------
            Iterable of (parent scalar, corresponding gradient) tuples.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        grads = h.last_fn._backward(h.ctx, d_output)

        if not isinstance(grads, tuple):
            grads = (grads,)

        result = []
        for input_var, grad in zip(h.inputs, grads):
            if not input_var.is_constant():
                result.append((input_var, grad))

        return result

    def backward(self, d_output: Optional[float] = None) -> None:
        """Perform backpropagation to compute the gradients for this scalar.

        This function calls the autodiff `backpropagate` method to propagate the
        derivatives through the computational graph.

        Args:
        ----
            d_output : The initial derivative to propagate. If None, defaults to 1.0.

        """
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)


def derivative_check(f: Any, *scalars: Scalar) -> None:
    """Checks that autodiff works on a Python function by comparing
    the calculated derivatives with numerical approximations.

    Asserts False if the derivative is incorrect.

    Parameters
    ----------
    f : Callable
        A function that takes n Scalar arguments and returns a single Scalar value.
    *scalars : Scalar
        A variable-length argument list of Scalar objects to check the derivatives for.

    """
    out = f(*scalars)
    out.backward()

    err_msg = """
Derivative check at arguments f(%s) and received derivative f'=%f for argument %d,
but was expecting derivative f'=%f from central difference."""
    for i, x in enumerate(scalars):
        check = central_difference(f, *scalars, arg=i)
        print(str([x.data for x in scalars]), x.derivative, i, check)
        assert x.derivative is not None
        np.testing.assert_allclose(
            x.derivative,
            check.data,
            1e-2,
            1e-2,
            err_msg=err_msg
            % (str([x.data for x in scalars]), x.derivative, i, check.data),
        )
