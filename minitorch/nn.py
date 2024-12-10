from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    height_res = height // kh
    width_res = width // kw
    input = input.contiguous()
    new_tensor = input.view(batch, channel, height_res, kh, width_res, kw)
    reshaped = new_tensor.permute(0, 1, 2, 4, 3, 5)
    output = reshaped.contiguous()
    output = output.view(batch, channel, height_res, width_res, kh * kw)
    return output, height_res, width_res


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling 2D over a tensor using a given kernel size

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width that has been averaged over the pooling kernel

    """
    batch = input.shape[0]
    channel = input.shape[1]
    # input_height = input.shape[2]
    # input_width = input.shape[3]

    tiled, new_height, new_width = tile(input, kernel)
    mean_tiled = tiled.mean(dim=-1)
    output = mean_tiled.view(batch, channel, new_height, new_width)

    return output


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to reduce over

    Returns:
    -------
        Tensor of size batch x channel x height x width x 1 that is 1 in the location of the max value and 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    """Max operator for the maxpool2d function"""

    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Performs the forward pass for the max operation.

        Args:
        ----
        ctx: A context object used to store information required for the backward pass.
        input: A tensor with shape (batch, channel, height, width).
        dim: The dimension along which the maximum operation is performed.

        Returns:
        -------
        A tensor of the same shape as the input (batch, channel, height, width),
        where each element is 1 if it corresponds to the maximum value along the specified dimension,
        and 0 otherwise.

        """
        dim_int = int(dim.item())
        ctx.save_for_backward(input, dim_int)
        return max_reduce(input, dim_int)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass for the max operation"""
        input, dim = ctx.saved_values
        return argmax(input, dim), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction over a tensor"""
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute softmax along the specified dimension."""
    val_exp = input.exp()
    res = val_exp.sum(dim=dim)
    return val_exp / res


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute log of softmax along the specified dimension."""
    exp_sum = input.exp().sum(dim=dim)
    log_exp_sum = exp_sum.log()
    return input - log_exp_sum


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply 2D max pooling with the given kernel size."""
    batch, channel, height, width = input.shape
    tiled, new_height, new_width = tile(input, kernel)
    return max(tiled, dim=-1).view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the tensor with the specified rate."""
    if ignore or rate == 0:
        return input
    else:
        out = rand(input.shape) > rate
        return out * input
