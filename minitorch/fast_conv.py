from typing import Tuple, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """Just-in-time (JIT) compile a function for performance optimization.

    Args:
    ----
        fn (Fn): The function to be JIT-compiled.
        **kwargs (Any): Additional keyword arguments passed to the JIT compiler.

    Returns:
    -------
        Fn: The JIT-compiled version of the input function.

    """
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """1D Convolution implementation."""
    batch, out_channels, out_width = out_shape
    _, in_channels, input_width = input_shape
    _, _, kernel_width = weight_shape

    input_s = input_strides
    weight_s = weight_strides

    for i in prange(out_size):
        out_idx = np.empty(MAX_DIMS, np.int32)
        to_index(i, out_shape, out_idx)
        b, oc, ow = out_idx[:3]

        result = 0.0
        out_pos = index_to_position(out_idx, out_strides)

        for ic in range(in_channels):
            for k in range(kernel_width):
                offset = kernel_width - 1 - k if reverse else k
                weight_pos = oc * weight_s[0] + ic * weight_s[1] + offset * weight_s[2]
                iw = ow - offset if reverse else ow + offset

                if 0 <= iw < input_width:
                    input_pos = b * input_s[0] + ic * input_s[1] + iw * input_s[2]
                    result += input[input_pos] * weight[weight_pos]
        out[out_pos] = result


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
        -------
            batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradients of input and weight tensors during backpropagation.

        Args:
        ----
            ctx (Context): The context object containing saved values from the forward pass.
            grad_output (Tensor): The gradient of the output tensor from the subsequent layer.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to input and weight tensors.

        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """2D Convolution implementation."""
    batch, out_channels, out_height, out_width = out_shape
    _, in_channels, input_height, input_width = input_shape
    _, _, kernel_height, kernel_width = weight_shape

    input_s = input_strides
    weight_s = weight_strides

    for i in prange(out_size):
        out_idx = np.empty(MAX_DIMS, np.int32)
        to_index(i, out_shape, out_idx)
        b, oc, oh, ow = out_idx[:4]

        result = 0.0
        out_pos = index_to_position(out_idx, out_strides)

        for ic in range(in_channels):
            for kh in range(kernel_height):
                for kw in range(kernel_width):
                    h_offset = kernel_height - 1 - kh if reverse else kh
                    w_offset = kernel_width - 1 - kw if reverse else kw

                    weight_pos = (
                        oc * weight_s[0]
                        + ic * weight_s[1]
                        + h_offset * weight_s[2]
                        + w_offset * weight_s[3]
                    )

                    ih = oh - h_offset if reverse else oh + h_offset
                    iw = ow - w_offset if reverse else ow + w_offset

                    if 0 <= ih < input_height and 0 <= iw < input_width:
                        input_pos = (
                            b * input_s[0]
                            + ic * input_s[1]
                            + ih * input_s[2]
                            + iw * input_s[3]
                        )
                        result += input[input_pos] * weight[weight_pos]

        out[out_pos] = result


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D Convolution

        Args:
        ----
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
        -------
            (:class:`Tensor`) : batch x out_channel x h x w

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes the gradients of input and weight tensors during backpropagation.

        Args:
        ----
            ctx (Context): The context object containing saved values from the forward pass.
            grad_output (Tensor): The gradient of the output tensor from the subsequent layer.

        Returns:
        -------
            Tuple[Tensor, Tensor]: Gradients with respect to input and weight tensors.

        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(  # type: ignore
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
