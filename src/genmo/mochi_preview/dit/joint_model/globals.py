_USE_XDIT = False

def set_use_xdit(use_dit: bool) -> None:
    """Set whether to use DIT model.
    
    Args:
        use_dit: Boolean flag indicating whether to use xdur
    """
    global _USE_XDIT
    _USE_XDIT = use_dit

def is_use_xdit() -> bool:
    return _USE_XDIT
