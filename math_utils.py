"""Lightweight math utility functions."""

from functools import lru_cache
from typing import List

# A 2x2 matrix represented as a list of two rows, each a list of two ints.
Matrix = List[List[int]]


@lru_cache(maxsize=128)
def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number using the standard 0-indexed sequence.

    The sequence is defined as:
        fibonacci(0) = 0
        fibonacci(1) = 1
        fibonacci(n) = fibonacci(n - 1) + fibonacci(n - 2) for n >= 2

    This uses the straightforward iterative (O(n) time, O(1) space) algorithm.
    Results are memoized with :func:`functools.lru_cache` (an LRU cache holding
    up to 128 entries), so a repeated call with the same ``n`` is served from
    the cache in O(1) time instead of recomputing.

    Args:
        n: A non-negative integer index into the Fibonacci sequence.

    Returns:
        The n-th Fibonacci number.

    Raises:
        ValueError: If ``n`` is negative.
        TypeError: If ``n`` is not an integer.
    """
    if not isinstance(n, int):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _matrix_multiply(a: Matrix, b: Matrix) -> Matrix:
    """Multiply two 2x2 integer matrices."""
    return [
        [a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]],
        [a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]],
    ]


def _matrix_power(matrix: Matrix, n: int) -> Matrix:
    """Raise a 2x2 matrix to the ``n``-th power using exponentiation by squaring.

    Runs in O(log n) matrix multiplications.
    """
    result: Matrix = [[1, 0], [0, 1]]  # 2x2 identity matrix
    base = matrix
    while n > 0:
        if n % 2 == 1:
            result = _matrix_multiply(result, base)
        base = _matrix_multiply(base, base)
        n //= 2
    return result


# Verified by Kiro
def fibonacci_fast(n: int) -> int:
    """Return the n-th Fibonacci number using O(log n) matrix exponentiation.

    Uses exponentiation by squaring on the matrix ``[[1, 1], [1, 0]]``, which
    satisfies::

        [[F(n+1), F(n)  ],
         [F(n),   F(n-1)]] = [[1, 1], [1, 0]] ** n

    so ``fibonacci(n)`` is the top-right entry of that matrix raised to the
    ``n``-th power. This performs only O(log n) matrix multiplications versus
    the O(n) additions of :func:`fibonacci`, making it asymptotically faster
    for large ``n``.

    Args:
        n: A non-negative integer index into the Fibonacci sequence.

    Returns:
        The n-th Fibonacci number.

    Raises:
        ValueError: If ``n`` is negative.
        TypeError: If ``n`` is not an integer.

    Examples:
        >>> fibonacci_fast(0)
        0
        >>> fibonacci_fast(1)
        1
        >>> fibonacci_fast(10)
        55
        >>> fibonacci_fast(20) == fibonacci(20)
        True
    """
    if not isinstance(n, int):
        raise TypeError(f"n must be an integer, got {type(n).__name__}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    # _matrix_power returns the identity matrix for n == 0, whose top-right
    # entry is 0, so no special case is required.
    powered = _matrix_power([[1, 1], [1, 0]], n)
    return powered[0][1]
