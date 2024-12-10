from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # Create a list of values for the function call
    vals = tuple(vals)

    # Convert vals to a list if it's not already
    vals_list = list(vals)

    vals_plus_epsilon = vals_list.copy()  # Create a copy of the list
    vals_plus_epsilon[arg] += epsilon
    f_plus = f(*vals_plus_epsilon)

    vals_minus_epsilon = vals_list.copy()  # Create another copy of the list
    vals_minus_epsilon[arg] -= epsilon
    f_minus = f(*vals_minus_epsilon)

    derivative = (f_plus - f_minus) / (2 * epsilon)

    return derivative


variable_count = 1


class Variable(Protocol):
    """Protocol for Variable interface used for autodifferentiation."""

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the given derivative value `x` onto the variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Return the unique identifier of the variable."""
        ...

    def is_leaf(self) -> bool:
        """Return True if the variable is a leaf node (created by the user)."""
        ...

    def is_constant(self) -> bool:
        """Return True if the variable is a constant (no gradients required)."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return an iterable of parent variables."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to compute gradients with respect to parents."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable.

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = []
    stack = []

    def dfs(var: Variable) -> None:
        """Depth-first search to visit all the nodes in the computation graph.

        Args:
        ----
            var: The current variable to visit.

        """
        if var.unique_id not in visited and not var.is_constant():
            visited.append(var.unique_id)
            for parent in var.parents:
                dfs(parent)
            stack.insert(0, var)

    dfs(variable)
    return stack


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to compute derivatives for the leaf nodes.

    Args:
    ----
        variable (Variable): The right-most variable for which the derivative is being computed.
        deriv (Any): The derivative to propagate backward to the leaves.

    Returns:
    -------
        None: The function does not return anything. It updates the derivative values of each leaf
        through `accumulate_derivative`.

    """
    grad_table = {variable.unique_id: deriv}

    # Get the list of variables in topological order
    sorted_variables = topological_sort(variable)

    # Traverse the variables in topological order
    for var in sorted_variables:
        d_output = grad_table[var.unique_id]

        if var.is_leaf():
            var.accumulate_derivative(d_output)
        else:
            # Apply the chain rule to compute derivatives with respect to inputs
            for parent, d_input in var.chain_rule(d_output):
                if parent.unique_id in grad_table:
                    grad_table[parent.unique_id] += d_input
                else:
                    grad_table[parent.unique_id] = d_input


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation.

        Args:
        ----
            *values: Values that will be saved for use in backpropagation.

        """
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the tensors that were saved for backpropagation."""
        return self.saved_values
