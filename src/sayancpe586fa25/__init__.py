from sayancpe586fa25._core import hello_from_bin


def hello() -> str:
    return hello_from_bin()

from . import differential

__all__ = ["differential"]

