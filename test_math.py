"""Pytest unit tests and benchmarks for math_utils.fibonacci and fibonacci_fast."""

import time

import pytest

from math_utils import fibonacci, fibonacci_fast


# ---------------------------------------------------------------------------
# Correctness tests for the iterative fibonacci()
# ---------------------------------------------------------------------------
def test_base_case_zero():
    assert fibonacci(0) == 0


def test_base_case_one():
    assert fibonacci(1) == 1


@pytest.mark.parametrize("n,expected", [
    (2, 1),
    (3, 2),
    (4, 3),
    (5, 5),
    (6, 8),
    (7, 13),
    (10, 55),
    (20, 6765),
])
def test_positive_integers(n, expected):
    assert fibonacci(n) == expected


def test_negative_input_raises_value_error():
    with pytest.raises(ValueError):
        fibonacci(-1)


def test_negative_input_message():
    with pytest.raises(ValueError, match="non-negative"):
        fibonacci(-5)


def test_non_integer_input_raises_type_error():
    with pytest.raises(TypeError):
        fibonacci(3.5)


# ---------------------------------------------------------------------------
# Correctness tests for fibonacci_fast()
# ---------------------------------------------------------------------------
def test_fibonacci_fast_base_case_zero():
    assert fibonacci_fast(0) == 0


def test_fibonacci_fast_base_case_one():
    assert fibonacci_fast(1) == 1


@pytest.mark.parametrize("n,expected", [
    (2, 1),
    (3, 2),
    (4, 3),
    (5, 5),
    (6, 8),
    (7, 13),
    (10, 55),
    (20, 6765),
    (30, 832040),
    (50, 12586269025),
])
def test_fibonacci_fast_positive_integers(n, expected):
    assert fibonacci_fast(n) == expected


def test_fibonacci_fast_negative_input_raises_value_error():
    with pytest.raises(ValueError):
        fibonacci_fast(-1)


def test_fibonacci_fast_negative_input_message():
    with pytest.raises(ValueError, match="non-negative"):
        fibonacci_fast(-5)


def test_fibonacci_fast_non_integer_input_raises_type_error():
    with pytest.raises(TypeError):
        fibonacci_fast("10")


# ---------------------------------------------------------------------------
# Comparative correctness: the two implementations must agree
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", list(range(0, 50)) + [100, 500, 1000, 5000])
def test_fast_matches_iterative(n):
    assert fibonacci_fast(n) == fibonacci(n)


# ---------------------------------------------------------------------------
# Comparative benchmark: fibonacci_fast should be faster for large n
# ---------------------------------------------------------------------------
def _measure(func, n, repeats=5):
    """Return the minimum wall-clock time over ``repeats`` runs of ``func(n)``.

    If ``func`` is an LRU-cached callable (exposes ``cache_clear``), the cache
    is cleared before each repeat so every run measures the real cost of
    computation rather than an instant cache hit. Non-cached functions (e.g.
    :func:`fibonacci_fast`) are unaffected.
    """
    best = float("inf")
    for _ in range(repeats):
        # Reset any memoization so each repeat reflects a cold computation.
        if hasattr(func, "cache_clear"):
            func.cache_clear()
        start = time.perf_counter()
        func(n)
        best = min(best, time.perf_counter() - start)
    return best


def test_benchmark_fibonacci_fast_faster_for_large_n():
    """For a large ``n``, the O(log n) implementation should outperform O(n).

    A generous tolerance factor is used so this stays robust on slow or noisy
    machines while still catching a genuine algorithmic regression (e.g. if
    fibonacci_fast were accidentally O(n)).
    """
    n = 200_000
    iter_time = _measure(fibonacci, n)
    fast_time = _measure(fibonacci_fast, n)

    # The two implementations must agree on the result for this n.
    assert fibonacci_fast(n) == fibonacci(n)

    # fibonacci_fast must be meaningfully faster; allow up to 75% of the
    # iterative time to absorb measurement noise while still requiring a win.
    assert fast_time < 0.75 * iter_time, (
        f"fibonacci_fast ({fast_time:.6f}s) was not sufficiently faster than "
        f"fibonacci ({iter_time:.6f}s) for n={n}"
    )


def test_benchmark_reports_timings(capfd):
    """A documented benchmark that prints comparative timings for inspection."""
    n = 100_000
    iter_time = _measure(fibonacci, n, repeats=3)
    fast_time = _measure(fibonacci_fast, n, repeats=3)
    print(
        f"\n[benchmark] n={n}: iterative={iter_time:.6f}s, "
        f"fast={fast_time:.6f}s, speedup={iter_time / fast_time:.1f}x"
    )
    # Sanity: both produced the same value.
    assert fibonacci(n) == fibonacci_fast(n)


# ---------------------------------------------------------------------------
# LRU cache (memoization) tests for fibonacci()
# ---------------------------------------------------------------------------
def test_fibonacci_is_cached():
    """fibonacci should be memoized via functools.lru_cache."""
    assert hasattr(fibonacci, "cache_info"), "fibonacci should be lru_cached"
    assert hasattr(fibonacci, "cache_clear")


def test_fibonacci_cache_registers_hits():
    """A repeated call for the same n registers a cache hit, not a new miss."""
    fibonacci.cache_clear()
    first = fibonacci(10)
    info_after_first = fibonacci.cache_info()
    assert info_after_first.misses == 1
    assert info_after_first.hits == 0

    second = fibonacci(10)
    info_after_second = fibonacci.cache_info()
    assert info_after_second.misses == 1
    assert info_after_second.hits == 1

    # The cached value must equal the freshly computed one.
    assert first == second == 55


def test_fibonacci_cache_misses_once_per_distinct_n():
    """Each distinct argument misses exactly once; repeats are cache hits."""
    fibonacci.cache_clear()
    values = [0, 1, 2, 3, 5, 8, 13, 21]
    for n in values:
        fibonacci(n)

    info = fibonacci.cache_info()
    assert info.misses == len(values)
    assert info.currsize == len(values)

    # Re-call two previously seen arguments: hits accumulate, misses do not.
    fibonacci(5)
    fibonacci(8)
    info2 = fibonacci.cache_info()
    assert info2.hits == 2
    assert info2.misses == len(values)
    assert info2.currsize == len(values)


def test_fibonacci_cached_results_match_expected_sequence():
    """Cached results must still match the canonical Fibonacci sequence."""
    fibonacci.cache_clear()
    expected = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
    for n, exp in enumerate(expected):
        assert fibonacci(n) == exp, f"fibonacci({n}) != {exp}"


def test_fibonacci_cached_matches_fibonacci_fast():
    """Cached fibonacci results must agree with the independent fibonacci_fast."""
    fibonacci.cache_clear()
    for n in [0, 1, 2, 10, 20, 50, 100]:
        assert fibonacci(n) == fibonacci_fast(n), f"mismatch at n={n}"


def test_fibonacci_cache_clear_resets_state():
    """cache_clear() empties the cache and resets hit/miss counters."""
    fibonacci.cache_clear()
    fibonacci(10)
    fibonacci(10)
    assert fibonacci.cache_info().currsize == 1

    fibonacci.cache_clear()
    info = fibonacci.cache_info()
    assert info.currsize == 0
    assert info.hits == 0
    assert info.misses == 0
