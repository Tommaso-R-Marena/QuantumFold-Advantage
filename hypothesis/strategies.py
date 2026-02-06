"""Minimal strategy objects for local tests."""


class _Strategy:
    def __init__(self, factory):
        self._factory = factory

    def example(self):
        return self._factory()


def integers(min_value=0, max_value=100):
    return _Strategy(lambda: min_value)


def floats(min_value=-1.0, max_value=1.0, allow_nan=False):
    return _Strategy(lambda: float(min_value if min_value is not None else 0.0))


def lists(strategy, min_size=0, max_size=10):
    size = min_size
    return _Strategy(lambda: [strategy.example() for _ in range(size)])
