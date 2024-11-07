from typing import Tuple


_USE_XDIT = False

def set_use_xdit(use_dit: bool) -> None:
    """Set whether to use DIT model.
    
    Args:
        use_dit: Boolean flag indicating whether to use xdur
    """
    global _USE_XDIT
    _USE_XDIT = use_dit
    print(f"The xDiT flag use_xdit={use_dit}")

def is_use_xdit() -> bool:
    return _USE_XDIT

_ULYSSES_DEGREE = None
_RING_DEGREE = None

def set_usp_config(ulysses_degree : int, ring_degree : int) -> None:
    global _ULYSSES_DEGREE, _RING_DEGREE
    _ULYSSES_DEGREE = ulysses_degree
    _RING_DEGREE = ring_degree
    print(f"Now we use xdit with ulysses degree {ulysses_degree} and ring degree {ring_degree}")

def get_usp_config() -> Tuple[int, int]:
    return _ULYSSES_DEGREE, _RING_DEGREE
