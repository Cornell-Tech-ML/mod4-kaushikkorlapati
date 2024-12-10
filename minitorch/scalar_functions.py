from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Converts the given value into a tuple if it is not already.

    Args:
    ----
        x: A single float or a tuple of floats.

    Returns:
    -------
        A tuple containing the value(s).

    """
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that operates on Scalar variables.

    This static class is used to group together the forward and backward methods
    used to compute the value and the gradient of the function during automatic differentiation.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Wraps the backward method to return a tuple of gradients.

        Args:
        ----
            ctx: The context storing information from the forward pass.
            d_out: The derivative from the next layer.

        Returns:
        -------
            Tuple of gradients for each input.

        """
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Wraps the forward method for computing the value of the function.

        Args:
        ----
            ctx: The context used to store information during forward pass.
            *inps: The input values for the function.

        Returns:
        -------
            The computed value.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Applies the scalar function to input Scalar values.

        Args:
        ----
            *vals: Scalars or values that can be converted to Scalars.

        Returns:
        -------
            A new Scalar object resulting from applying the function.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), f"Expected return type float, got {type(c)}."

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the sum of two values.

        Args:
        ----
            ctx: The context used to store information for backpropagation.
            a: The first input.
            b: The second input.

        Returns:
        -------
            The sum of the inputs.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the gradient of the addition operation.

        Args:
        ----
            ctx: The context storing information from the forward pass.
            d_output: The derivative from the next layer.

        Returns:
        -------
            Tuple of gradients for each input (same as `d_output`).

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Logarithmic function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the natural logarithm of the input.

        Args:
        ----
            ctx: The context used to store information for backpropagation.
            a: The input value.

        Returns:
        -------
            The natural logarithm of the input.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the logarithm operation.

        Args:
        ----
            ctx: The context storing information from the forward pass.
            d_output: The derivative from the next layer.

        Returns:
        -------
            The gradient of the logarithm function with respect to its input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the product of two values.

        Args:
        ----
            ctx: The context used to store information for backpropagation.
            a: The first input.
            b: The second input.

        Returns:
        -------
            The product of the inputs.

        """
        ctx.save_for_backward(a, b)
        return float(a * b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the gradient of the multiplication operation.

        Args:
        ----
            ctx: The context storing information from the forward pass.
            d_output: The derivative from the next layer.

        Returns:
        -------
            Tuple of gradients for each input.

        """
        (a, b) = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1 / x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the multiplicative inverse of the input.

        Args:
        ----
            ctx: The context used to store information for backpropagation.
            a: The input value.

        Returns:
        -------
            The inverse of the input.

        """
        ctx.save_for_backward(a)
        return float(1 / a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the inverse operation.

        Args:
        ----
            ctx: The context storing information from the forward pass.
            d_output: The derivative from the next layer.

        Returns:
        -------
            The gradient of the inverse function with respect to its input.

        """
        (a,) = ctx.saved_values
        return -d_output / (a**2)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the negation of the input.

        Args:
        ----
            ctx: The context used to store information for backpropagation.
            a: The input value.

        Returns:
        -------
            The negation of the input.

        """
        return float(-a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the negation operation.

        Args:
        ----
            ctx: The context storing information from the forward pass.
            d_output: The derivative from the next layer.

        Returns:
        -------
            The gradient of the negation function with respect to its input.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + exp(-x))$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the sigmoid of the input.

        Args:
        ----
            ctx: The context used to store information for backpropagation.
            a: The input value.

        Returns:
        -------
            The sigmoid of the input.

        """
        res = operators.sigmoid(a)
        ctx.save_for_backward(res)
        return float(res)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the sigmoid operation.

        Args:
        ----
            ctx: The context storing information from the forward pass.
            d_output: The derivative from the next layer.

        Returns:
        -------
            The gradient of the sigmoid function with respect to its input.

        """
        (sigmoid_a,) = ctx.saved_values
        return d_output * sigmoid_a * (1 - sigmoid_a)


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the ReLU (rectified linear unit) of the input.

        Args:
        ----
            ctx: The context used to store information for backpropagation.
            a: The input value.

        Returns:
        -------
            The ReLU of the input.

        """
        ctx.save_for_backward(a)
        return max(0.0, a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the gradient of the ReLU operation.

        Args:
        ----
            ctx: The context storing information from the forward pass.
        d_output: The derivative from the next layer.

        """
        (a,) = ctx.saved_values
        grad_a = d_output if a > 0 else 0.0
        return grad_a


class Exp(ScalarFunction):
    """Exponential function $f(x) = exp(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Computes the forward pass of the exponential function.

        Args:
        ----
            ctx : Context
                The context object to save values for backpropagation.
            a : float
                The input value for the exponential function.

        Returns:
        -------
            float
                The result of the exponential function evaluated at `a`.

        """
        ctx.save_for_backward(a)
        return float(operators.exp(a))

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Computes the backward pass of the exponential function.

        Args:
        ----
            ctx : Context
                The context object containing saved values from the forward pass.
            d_output : float
                The gradient of the output with respect to some loss.

        Returns:
        -------
            float
                The gradient of the input `a` for the backward pass.

        """
        (a,) = ctx.saved_values
        return d_output * operators.exp(a)


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the less than function.

        Args:
        ----
            ctx : Context
                The context object to save values for backpropagation.
            a : float
                The first input value for the comparison.
            b : float
                The second input value for the comparison.

        Returns:
        -------
            float
                The result of the comparison, which is 1.0 if `a < b` else 0.0.

        """
        return float(a < b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass of the less than function.

        Args:
        ----
            ctx : Context
                The context object containing saved values from the forward pass.
            d_output : float
                The gradient of the output with respect to some loss.

        Returns:
        -------
            Tuple[float, ...]
                Gradients with respect to inputs `a` and `b`. Both gradients are zero since
                the derivative of a comparison is not defined.

        """
        return 0, 0  # Derivative of a comparison is not defined


class EQ(ScalarFunction):
    """Equal function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Computes the forward pass of the equality function.

        Args:
        ----
            ctx : Context
                The context object to save values for backpropagation.
            a : float
                The first input value for the comparison.
            b : float
                The second input value for the comparison.

        Returns:
        -------
            float
                The result of the comparison, which is 1.0 if `a == b` else 0.0.

        """
        return float(a == b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Computes the backward pass of the equality function.

        Args:
        ----
            ctx : Context
                The context object containing saved values from the forward pass.
            d_output : float
                The gradient of the output with respect to some loss.

        Returns:
        -------
            Tuple[float, ...]
                Gradients with respect to inputs `a` and `b`, both set to zero since
                the derivative of an equality is not defined.

        """
        grad_a = 0.0
        grad_b = 0.0
        return grad_a, grad_b  # Derivative of an equality is not defined
