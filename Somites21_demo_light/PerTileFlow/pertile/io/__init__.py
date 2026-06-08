"""I/O utilities (slim build) — only TIFF loading is shipped."""
from .tiff import load_tiff_stack
__all__ = ["load_tiff_stack"]
