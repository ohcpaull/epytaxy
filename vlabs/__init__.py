from .spm import scanningprobe
from .xray import xrr, xrd


try:
    from refnx.version import version as __version__
except ImportError:
    __version__ = "version string not created yet"
