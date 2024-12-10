"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for the negation operation.

        This method negates the input tensor element-wise.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            t1 (Tensor): The input tensor to be negated.

        Returns:
        -------
            Tensor: A new tensor containing the negated values of the
            input tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for the negation operation.

        This method calculates the gradient of the negation operation
        during backpropagation.

        Args:
        ----
            ctx (Context): The context object containing saved information
                from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect
                to the output tensor.

        Returns:
        -------
            Tensor: The gradient of the loss with respect to the input tensor,
            which is equal to the gradient output due to the nature of the
            negation operation.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the forward pass for the inversion operation.

        This method computes the element-wise inverse of the input tensor.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            t1 (Tensor): The input tensor to be inverted.

        Returns:
        -------
            Tensor: A new tensor containing the inverses of the input tensor values.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the backward pass for the inversion operation.

        This method calculates the gradient of the inversion operation
        during backpropagation.

        Args:
        ----
            ctx (Context): The context object containing saved information
                from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect
                to the output tensor.

        Returns:
        -------
            Tensor: The gradient of the loss with respect to the input tensor.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the forward pass for the addition operation.

        This method computes the element-wise sum of two input tensors.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the element-wise sum of the
            input tensors.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the backward pass for the addition operation.

        This method calculates the gradient of the addition operation
        during backpropagation.

        Args:
        ----
            ctx (Context): The context object containing saved information
                from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect
                to the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: A tuple containing the gradients of
            the loss with respect to each input tensor, which are both
            equal to the gradient output due to the nature of the addition operation.

        """
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Compute the logical AND operation over the input tensor.

        This method returns a tensor indicating whether all elements
        along the specified dimension are true (non-zero). If no dimension
        is specified, it checks all elements in the tensor.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            a (Tensor): The input tensor containing boolean values
                (or values interpreted as boolean).
            dim (Tensor): The dimension along which to perform the
                logical AND operation. If None, the operation is performed
                over all elements in the tensor.

        Returns:
        -------
            Tensor: A tensor with the result of the logical AND operation.
            If `dim` is specified, the result will have one fewer dimension
            than the input tensor along that dimension. If no dimension is
            specified, the result is a scalar indicating if all elements
            in the tensor are true.

        """
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Perform element-wise multiplication of two tensors.

        This method computes the product of two input tensors,
        `t1` and `t2`, and saves them for use in the backward
        pass.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the element-wise product
            of `t1` and `t2`.

        """
        ctx.save_for_backward(t1, t2)
        return t1.f.mul_zip(t1, t2)  # Element-wise multiplication using zip

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute gradients for the input tensors during backpropagation.

        This method calculates the gradients of the input tensors
        with respect to the output tensor using the chain rule.

        Args:
        ----
            ctx (Context): The context object containing the saved
                values from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect
                to the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: A tuple containing the gradients
            of `t1` and `t2` respectively.

        """
        t1, t2 = ctx.saved_values
        return grad_output.f.mul_zip(t2, grad_output), grad_output.f.mul_zip(
            t1, grad_output
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor) -> Tensor:
        """Compute the sigmoid of the input tensor.

        This method applies the sigmoid activation function to the
        input tensor `t` and saves the result for use in the
        backward pass.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            t (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the sigmoid values of `t`.

        """
        sig_a = t.f.sigmoid_map(t)
        ctx.save_for_backward(sig_a)
        return sig_a

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the sigmoid function during backpropagation.

        This method calculates the gradient of the input tensor with
        respect to the output tensor based on the saved sigmoid values.

        Args:
        ----
            ctx (Context): The context object containing the saved
                values from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect
                to the output tensor.

        Returns:
        -------
            Tensor: The gradient of the input tensor `t` with respect
            to the output tensor.

        """
        sigma: Tensor = ctx.saved_values[0]
        return sigma * (-sigma + 1.0) * grad_output


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor) -> Tensor:
        """Apply the ReLU activation function to the input tensor.

        This method computes the rectified linear unit (ReLU) of the input
        tensor `t`, which sets negative values to zero.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            t (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the ReLU values of `t`,
            where negative values are set to zero.

        """
        ctx.save_for_backward(t)
        return t.f.relu_map(t)  # Using relu_map for ReLU operation

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the ReLU function during backpropagation.

        This method calculates the gradient of the input tensor with
        respect to the output tensor based on the saved values from the
        forward pass.

        Args:
        ----
            ctx (Context): The context object containing the saved
                values from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect
                to the output tensor.

        Returns:
        -------
            Tensor: The gradient of the input tensor `t` with respect
            to the output tensor.

        """
        (t1,) = ctx.saved_values
        return t1.f.relu_back_zip(t1, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor) -> Tensor:
        """Compute the natural logarithm of the input tensor.

        This method applies the logarithm function to the input tensor
        `t`.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            t (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the logarithm values of `t`.

        """
        ctx.save_for_backward(t)
        return t.f.log_map(t)  # Using the log method of Tensor

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the logarithm function during backpropagation.

        This method calculates the gradient of the input tensor with
        respect to the output tensor based on the saved values from the
        forward pass.

        Args:
        ----
            ctx (Context): The context object containing the saved
                values from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect
                to the output tensor.

        Returns:
        -------
            Tensor: The gradient of the input tensor `t` with respect
            to the output tensor.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor) -> Tensor:
        """Compute the exponential of the input tensor.

        This method applies the exponential function to the input
        tensor `t`.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            t (Tensor): The input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the exponential values of `t`.

        """
        exp_t = t.f.exp_map(t)  # Using the exp method of Tensor
        ctx.save_for_backward(exp_t)
        return exp_t

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient of the exponential function during backpropagation.

        This method calculates the gradient of the input tensor with
        respect to the output tensor based on the saved values from the
        forward pass.

        Args:
        ----
            ctx (Context): The context object containing the saved
                values from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect
                to the output tensor.

        Returns:
        -------
            Tensor: The gradient of the input tensor `t` with respect
            to the output tensor.

        """
        (exp_t1,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, exp_t1)


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, t: Tensor, dim: Optional[Tensor] | None) -> Tensor:
        """Compute the sum of the elements of the input tensor along a specified dimension.

        This method reduces the input tensor `t` by summing its elements
        along the specified dimension `dim`. If no dimension is specified,
        the entire tensor is summed.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            t (Tensor): The input tensor.
            dim (int, optional): The dimension along which to sum the elements.
                If None, sums all elements in the tensor.

        Returns:
        -------
            Tensor: A new tensor containing the sum of the elements of `t`.

        """
        # Saving the shape of 't' and 'dim' in the context for backward pass
        ctx.save_for_backward(t.shape, dim)
        if dim is not None:
            return t.f.add_reduce(t, int(dim.item()))
        else:
            return t.f.add_reduce(t.contiguous().view(int(operators.prod(t.shape))), 0)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the sum operation during backpropagation.

        This method returns the gradient of the input tensor with respect
        to the output tensor and a zero tensor for the dimension gradient.

        Args:
        ----
            ctx (Context): The context object containing the saved
                values from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect
                to the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: The gradient of the input tensor and
            a zero tensor for the dimension gradient.

        """
        shape, dim = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Perform element-wise less-than comparison between two tensors.

        This method computes the result of the element-wise less-than
        operation between the input tensors `t1` and `t2`.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the boolean results of the
            less-than comparisons.

        """
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.lt_zip(t1, t2)  # Using lt_map for less-than operation

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the less-than operation during backpropagation.

        This method returns zero tensors for the gradients of both input
        tensors since the less-than operation is non-differentiable.

        Args:
        ----
            ctx (Context): The context object containing the saved
                values from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect
                to the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Two zero tensors for the gradients
            of the input tensors.

        """
        zero_grad = grad_output * 0
        return zero_grad, zero_grad


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Perform element-wise equality comparison between two tensors.

        This method computes the result of the element-wise equality
        operation between the input tensors `t1` and `t2`.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            t1 (Tensor): The first input tensor.
            t2 (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A new tensor containing the boolean results of the
            equality comparisons.

        """
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.eq_zip(t1, t2)  # Using eq_map for equality operation

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the equality operation during backpropagation.

        This method returns zero tensors for the gradients of both input
        tensors since the equality operation is non-differentiable.

        Args:
        ----
            ctx (Context): The context object containing the saved
                values from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect
                to the output tensor.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Two zero tensors for the gradients
            of the input tensors.

        """
        (t1_shape, t2_shape) = ctx.saved_values
        return zeros(t1_shape), zeros(t2_shape)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Check element-wise closeness between two tensors.

        This method returns a tensor indicating whether each element
        of tensor `a` is close to the corresponding element of tensor `b`.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            a (Tensor): The first input tensor.
            b (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: A new tensor containing boolean values, where each
            value indicates if the corresponding elements in `a` and `b`
            are close to each other.

        """
        return a.f.is_close_zip(a, b)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Permute the dimensions of a tensor according to the specified order.

        This method rearranges the dimensions of the input tensor `a`
        based on the indices specified in `order`.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            a (Tensor): The input tensor to permute.
            order (Tensor): A tensor containing the new order of dimensions.

        Returns:
        -------
            Tensor: A new tensor with its dimensions permuted according
            to the specified order.

        """
        order_list = order.to_numpy().astype(int).tolist()
        ctx.save_for_backward(order_list)
        return a._new(a._tensor.permute(*order_list))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the permute operation during backpropagation.

        This method returns the gradient of the input tensor
        with respect to the output tensor, undoing the permute operation.

        Args:
        ----
            ctx (Context): The context object containing the saved
                values from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect
                to the output tensor.

        Returns:
        -------
            Tuple[Tensor, float]: The gradient of the input tensor and
            a float (zero) for the dimension gradient.

        """
        (order_list,) = ctx.saved_values
        undo_permute_order = [order_list.index(i) for i in range(len(order_list))]
        return grad_output._new(grad_output._tensor.permute(*undo_permute_order)), 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Change the shape of a tensor without changing its data.

        This method returns a new tensor with the same data as `a` but
        reshaped according to the dimensions specified in `shape`.
        The original tensor must be contiguous.

        Args:
        ----
            ctx (Context): The context object to save any information
                needed for backpropagation.
            a (Tensor): The input tensor to reshape.
            shape (Tensor): A tensor specifying the new shape.

        Returns:
        -------
            Tensor: A new tensor with the specified shape.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the view operation during backpropagation.

        This method returns a tensor with the same data as `grad_output`
        but reshaped back to the original shape of the input tensor.

        Args:
        ----
            ctx (Context): The context object containing the saved
                values from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect
                to the output tensor.

        Returns:
        -------
            Tuple[Tensor, float]: The gradient of the input tensor and
            a float (zero) for the dimension gradient.

        """
        (og,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, og, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors
def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Calculate the numerical gradient of a function using central difference.

    This function approximates the derivative of the function `f` with
    respect to the `arg`-th input tensor, using the central difference
    method. It perturbs the input tensor slightly in both directions and
    computes the difference in the function outputs to estimate the gradient.

    Args:
    ----
        f (Any): The function for which to compute the gradient.
        *vals (Tensor): The input tensors to the function `f`.
        arg (int): The index of the tensor with respect to which the gradient is computed (default is 0).
        epsilon (float): The small perturbation value used in the central difference approximation (default is 1e-6).
        ind (UserIndex): The index at which to perturb the input tensor.

    Returns:
    -------
        float: The estimated gradient of the function at the specified input tensor.

    """
    # Extract the tensor to perturb
    x = vals[arg]

    # Create a perturbation tensor
    up = zeros(x.shape)
    up[ind] = epsilon

    # Create perturbed input tensors
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]

    # Compute the difference in function outputs
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    # Return the estimated gradient
    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
