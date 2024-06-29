"""Contexts for performance timing."""

import contextlib
import time


@contextlib.contextmanager
def capture_time():
    """Capture the time elapsed in a context.

    Yields:
        A function that returns the time elapsed since the context was entered.

    Example:
        >>> with capture_time() as elapsed:
        ...     time.sleep(1)
        ...
        >>> elapsed()
        1.0
    """
    start = time.perf_counter()
    done = False

    def fn():
        if done:
            return end - start
        else:
            return time.perf_counter() - start

    yield fn
    end = time.time()
