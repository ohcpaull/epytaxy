from .spm import scanningprobe
from .xray import xrr, xrd

from vlabs.spm.utils import(
    gaussian,
    skewed_gauss,
    rayleigh,
    exp_dist,
    lorentz,
    line,
    parabola,
    second_poly,
    cubic,
    exp,
    log,
    sine,
    cosine,
)

try:
    from refnx.version import version as __version__
except ImportError:
    __version__ = "version string not created yet"
