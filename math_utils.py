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
    """Raise a 2x2 matrix to the ``n``-th power via binary exponentiation.

    Binary exponentiation (also called "exponentiation by squaring") computes
    ``matrix ** n`` using only O(log n) 2x2 matrix multiplications instead of
    the naive O(n) repeated-multiply approach. The key insight is that any
    power can be expressed in terms of the bits of ``n``.

    The algorithm writes the exponent ``n`` in binary as
    ``b_k b_{k-1} ... b_1 b_0`` and processes its bits from least-significant
    to most-significant, accumulating factors of ``matrix ** (2 ** i)`` for
    each set bit ``b_i``:

    1. Start with ``result`` as the identity matrix ``[[1, 0], [0, 1]]`` (the
       multiplicative neutral element) and ``base = matrix``.
    2. While ``n > 0``:
       a. If the current least-significant bit of ``n`` is 1 (i.e. ``n`` is
          odd), fold ``base`` into the running product:
          ``result = result * base``.
       b. Square the running base for the next higher bit position:
          ``base = base * base``.
       c. Shift ``n`` right by one bit: ``n //= 2``.

    Why this is correct: because matrix multiplication is associative, we may
    group the ``n``-fold product ``matrix * matrix * ... * matrix`` freely.
    Using the binary expansion::

        matrix ** n = product over i where b_i == 1 of (matrix ** (2 ** i))

    Each loop iteration ``i`` supplies the factor ``matrix ** (2 ** i)``
    through ``base``, which has been squared ``i`` times. The bit test
    ``n % 2 == 1`` decides whether that factor is multiplied into ``result``.
    This mirrors the way Horner's method builds up a value from its binary
    digits, only here the operations are 2x2 matrix multiplications.

    Complexity: ``base`` is squared every iteration and the loop runs once per
    bit of ``n``, so the number of 2x2 matrix multiplications is at most
    ``2 * floor(log2(n)) + 1`` -- one squaring per bit plus one multiply per
    set bit. Since each 2x2 multiply is a constant-time (O(1)) arithmetic
    operation, the overall time complexity is O(log n) with O(1) extra space
    (only ``result`` and ``base`` are held).

    This is the engine behind :func:`fibonacci_fast`, which raises the
    Fibonacci Q-matrix ``[[1, 1], [1, 0]]`` to the ``n``-th power: the result
    satisfies ``Q ** n == [[F(n+1), F(n)], [F(n), F(n-1)]]``, so the top-right
    entry equals ``fibonacci(n)``.

    Args:
        matrix: A 2x2 matrix, i.e. a list of two rows each of two ints,
            written as ``[[a, b], [c, d]]``.
        n: A non-negative integer exponent.

    Returns:
        The matrix product ``matrix`` raised to the ``n``-th power, as a fresh
        2x2 matrix. For ``n == 0`` the 2x2 identity matrix ``[[1, 0], [0, 1]]``
        is returned regardless of the input.

    Examples:
        The identity matrix is the neutral element, so any power returns it:

        >>> _matrix_power([[1, 0], [0, 1]], 5)
        [[1, 0], [0, 1]]

        Raising to the 0th power yields the identity matrix:

        >>> _matrix_power([[2, 3], [4, 5]], 0)
        [[1, 0], [0, 1]]

        Raising to the 1st power returns a copy of the matrix itself:

        >>> _matrix_power([[2, 0], [0, 3]], 1)
        [[2, 0], [0, 3]]

        A diagonal matrix raised to a power keeps its diagonal form, with each
        diagonal element raised to that power:

        >>> _matrix_power([[2, 0], [0, 3]], 4)
        [[16, 0], [0, 81]]

        The Fibonacci Q-matrix raised to the n-th power has ``fibonacci(n)``
        in its top-right entry:

        >>> _matrix_power([[1, 1], [1, 0]], 10)
        [[89, 55], [55, 34]]
        >>> _matrix_power([[1, 1], [1, 0]], 10)[0][1] == fibonacci(10)
        True
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
