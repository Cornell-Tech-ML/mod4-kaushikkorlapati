"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Sets whether the variable requires gradient computation.

        This method initializes the history of the variable. If `x` is True,
        the variable will track operations for gradient computation.

        Args:
        ----
            x (bool): A flag indicating whether the variable should track
                    gradients (True) or not (False).

        Returns:
        -------
            None

        """
        self.history = History()

    def requires_grad(self) -> bool:
        """Checks if the variable requires gradient computation.

        Returns
        -------
            bool: True if the variable is set to track gradients,
                False otherwise.

        """
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Returns
        Converted to numpy array

        """
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Turns a python number into a tensor with the same backend."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)  # type: ignore
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float"""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data"""
        return Copy.apply(self)

    def __repr__(self) -> str:
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from data"""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Method used to allow for backprop over broadcasting.
        This method is called when the output of `backward`
        is a different size than the input of `forward`.


        Args:
        ----
            other : backward tensor (must broadcast with self)

        Returns:
        -------
            Expanded version of `other` with the right derivatives

        """
        # Case 1: Both the same shape.
        if self.shape == other.shape:
            return other

        # Case 2: Backward is a smaller than self. Broadcast up.
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf

        # Case 3: Still different, reduce extra dims.
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        # START CODE CHANGE (2021)
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)
        # END CODE CHANGE (2021)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Creates a tensor filled with zeros.

        This method creates a new tensor of the specified shape, initialized
        with zeros. If no shape is provided, it uses the shape of the
        current tensor.

        Args:
        ----
            shape (Optional[UserShape]): The shape of the tensor to be created.
                                        If None, the shape of the current
                                        tensor will be used.

        Returns:
        -------
            Tensor: A new tensor filled with zeros, having the specified shape
                    or the shape of the current tensor if none is provided.

        """

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Get the tensor data info as a tuple."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Detach from backprop"""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Add `val` to the the derivative accumulated on this variable.
        Should only be called during autodifferentiation on leaf variables.

        Args:
        ----
            x : value to be accumulated

        """
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """True if this variable created by the user (no `last_fn`)"""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Determines if the current variable is a constant.

        A variable is considered constant if it does not have a history
        of operations (i.e., it was not created from other variables).

        Returns
        -------
            bool: True if the variable is a constant, False otherwise.

        """
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Returns the parent variables of the current variable.

        This property retrieves the input variables that contributed to
        the creation of this variable in the computation graph.

        Returns
        -------
            Iterable[Variable]: An iterable of parent variables.

        """
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Applies the chain rule to compute the gradients for parent variables.

        Args:
        ----
            d_output (Any): The gradient of the output with respect to the
                            current variable.

        Returns:
        -------
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, each containing
                                            a parent variable and its corresponding
                                            gradient.

        Raises:
        ------
            AssertionError: If there is an inconsistency in the computation graph.

        """
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Computes the gradients of the current variable with respect to the inputs.

        Args:
        ----
            grad_output (Optional[Tensor]): The gradient of the output with respect
                                            to this variable. If None and this
                                            variable is not a scalar, a default
                                            gradient of 1.0 will be used.

        Raises:
        ------
            AssertionError: If grad_output is None and this variable is not scalar.

        """
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    # Functions
    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor.

        Returns
        -------
            int: The total number of elements contained in the tensor.

        """
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Returns the number of dimensions of the tensor.

        Returns
        -------
            int: The number of dimensions of the tensor (also known as
                the tensor's rank).

        """
        return self._tensor.dims

    def zero_grad_(self) -> None:
        """Resets the gradient of the tensor to None.

        This method is typically used in the training process of neural networks
        to clear the gradients from the previous iteration before performing
        backpropagation.

        Returns
        -------
            None

        """
        self.grad = None

    def __add__(self, other: TensorLike) -> Tensor:
        """Implements the addition operation for tensors.

        Args:
        ----
            other (TensorLike): The tensor or scalar to add to this tensor.

        Returns:
        -------
            Tensor: A new tensor resulting from the addition of this tensor
                    and the specified tensor or scalar.

        """
        other = self._ensure_tensor(other)
        return Add.apply(self, other)

    def __sub__(self, other: TensorLike) -> Tensor:
        """Implements the subtraction operation for tensors.

        Args:
        ----
            other (TensorLike): The tensor or scalar to subtract from this tensor.

        Returns:
        -------
            Tensor: A new tensor resulting from the subtraction of the specified
                    tensor or scalar from this tensor.

        """
        other = self._ensure_tensor(other)
        return Add.apply(self, -other)

    def __mul__(self, other: TensorLike) -> Tensor:
        """Implements the multiplication operation for tensors.

        Args:
        ----
            other (TensorLike): The tensor or scalar to multiply with this tensor.

        Returns:
        -------
            Tensor: A new tensor resulting from the multiplication of this tensor
                    and the specified tensor or scalar.

        """
        other = self._ensure_tensor(other)
        return Mul.apply(self, other)

    def __neg__(self) -> Tensor:
        """Implements the unary negation operation for tensors.

        Returns
        -------
            Tensor: A new tensor containing the negated values of the original tensor.

        """
        return Neg.apply(self)

    def __lt__(self, other: TensorLike) -> Tensor:
        """Implements the less-than comparison operation for tensors.

        Args:
        ----
            other (TensorLike): The tensor or scalar to compare against.

        Returns:
        -------
            Tensor: A new tensor containing boolean values indicating whether
                    each element of this tensor is less than the corresponding
                    element of the specified tensor or scalar.

        """
        other = self._ensure_tensor(other)
        return LT.apply(self, other)

    def __gt__(self, other: TensorLike) -> Tensor:
        """Element-wise greater-than comparison."""
        return LT.apply(self._ensure_tensor(other), self)

    def __eq__(self, other: TensorLike) -> Tensor:
        """Implements the equality comparison operation for tensors.

        Args:
        ----
            other (TensorLike): The tensor or scalar to compare against.

        Returns:
        -------
            Tensor: A new tensor containing boolean values indicating whether
                    each element of this tensor is equal to the corresponding
                    element of the specified tensor or scalar.

        """
        other = self._ensure_tensor(other)
        return EQ.apply(self, other)

    def __radd__(self, other: TensorLike) -> Tensor:
        """Implements the right addition operation for tensors.

        Args:
        ----
            other (TensorLike): The tensor or scalar to add to this tensor.

        Returns:
        -------
            Tensor: A new tensor resulting from the addition of this tensor
                    to the specified tensor or scalar.

        """
        return self + other

    def __rmul__(self, other: TensorLike) -> Tensor:
        """Implements the right multiplication operation for tensors.

        Args:
        ----
            other (TensorLike): The tensor or scalar to multiply with.

        Returns:
        -------
            Tensor: A new tensor resulting from the multiplication of
                    this tensor with the specified tensor or scalar.

        """
        return self * other

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Computes the logical AND of all elements in the tensor along the specified dimension.

        Args:
        ----
            dim (Optional[int]): The dimension along which to compute the logical AND.
                                If None, computes the AND across all elements.

        Returns:
        -------
            Tensor: A new tensor containing the result of the logical AND
                    operation applied along the specified dimension.

        """
        if dim is None:
            return All.apply(self.view(self.size), self._ensure_tensor(0))
        else:
            return All.apply(self, self._ensure_tensor(dim))

    def is_close(self, b: Tensor) -> Tensor:
        """Checks if the elements of this tensor are close to those of another tensor.

        Args:
        ----
            b (Tensor): The tensor to compare against.

        Returns:
        -------
            Tensor: A new tensor containing boolean values indicating whether
                    each element is close to the corresponding element in the
                    specified tensor.

        """
        return IsClose.apply(self, self._ensure_tensor(b))

    def sigmoid(self) -> Tensor:
        """Applies the sigmoid activation function element-wise.

        Returns
        -------
            Tensor: A new tensor with the sigmoid function applied, where
                    each element is transformed to a value between 0 and 1.

        """
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Applies the ReLU (Rectified Linear Unit) activation function element-wise.

        Returns
        -------
            Tensor: A new tensor with ReLU applied, where all negative values
                    are set to zero.

        """
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Computes the natural logarithm of each element in the tensor.

        Returns
        -------
            Tensor: A new tensor containing the natural logarithm of each
                    element from the original tensor.

        """
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Computes the exponential of each element in the tensor.

        Returns
        -------
            Tensor: A new tensor containing the exponential of each element
                    from the original tensor.

        """
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Computes the sum of elements in the tensor along the specified dimension.

        Args:
        ----
            dim (Optional[int]): The dimension along which to compute the sum.
                                If None, computes the sum of all elements.

        Returns:
        -------
            Tensor: A new tensor containing the sum value(s) computed along the
                    specified dimension.

        """
        if dim is None:
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Calculates the mean of the tensor along the specified dimension.

        Args:
        ----
            dim (Optional[int]): The dimension along which to compute the mean.
                                If None, computes the mean of all elements.

        Returns:
        -------
            Tensor: A new tensor containing the mean value(s) computed.

        """
        if dim is None:
            return self.sum() / self.size
        return self.sum(dim) / self.shape[dim]

    def permute(self, *order: int) -> Tensor:
        """Rearranges the dimensions of the tensor according to the specified order.

        Args:
        ----
            *order (int): The new order of the dimensions. Can be a sequence of
                        integers specifying the desired arrangement.

        Returns:
        -------
            Tensor: A new tensor with its dimensions permuted.

        """
        if isinstance(order, int):
            order = (order,)
        else:
            order = tuple(order)

        return Permute.apply(self, tensor(order))

    def view(self, *shape: UserShape | int) -> Tensor:
        """Returns a new tensor with the same data but a different shape.

        The new shape must be compatible with the original shape.

        Args:
        ----
            *shape (UserShape | int): The desired shape for the new tensor.
                                    If a single integer is provided, it will be treated
                                    as the first dimension, and the rest will maintain
                                    their sizes.

        Returns:
        -------
            Tensor: A new tensor with the specified shape.

        """
        return View.apply(self, tensor(list(shape)))
