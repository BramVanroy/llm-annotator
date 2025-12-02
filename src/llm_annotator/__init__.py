def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies for utils-only usage."""
    if name == "Annotator":
        from .annotator import Annotator

        return Annotator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Annotator"]

