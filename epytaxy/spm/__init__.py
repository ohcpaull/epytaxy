"""
Scanning probe microscopy reading and analysis.

Quick start
-----------
>>> from epytaxy.spm import load_spm
>>> scan = load_spm("P0125OP0000.h5", data_dir="PFM analysis/data")
>>> scan.keys()
['Topography', 'Amplitude_1', 'Amplitude_2', 'Phase_1', 'Phase_2', 'Frequency']
>>> fig, axes = scan.plot_channels(["Topography", "Amplitude_1", "Phase_1"])

:func:`load_spm` picks the class from the file contents.  Construct one
directly (:class:`AsylumDART`, :class:`AsylumSF`, :class:`BrukerScan`,
:class:`Innova`) when you know what you have and want the channel layout
checked.  ``.ibw`` files are read directly -- there is no conversion step.
"""

from epytaxy.spm.channels import (
    Channel,
    DEFAULT_CMAPS,
    QUANTITY_UNITS,
    canonical_key,
    parse_channel_name,
)

from epytaxy.spm.igor_ibw import (
    channel_labels,
    is_ibw_file,
    parse_note,
    read_ibw,
)

from epytaxy.spm.scanningprobe import (
    AsylumDART,
    AsylumSF,
    AsylumScan,
    BrukerScan,
    Innova,
    SPMScan,
    convert_to_h5,
    load_files,
    load_spm,
    read_usid_h5,
)

from epytaxy.spm.widgets import (
    Innova_edge_detection_widget,
    edge_detection_widget,
    file_preview,
)

from epytaxy.spm.utils import (
    Gauss2d,
    LineProfile,
    cosine,
    cubic,
    exp,
    exp_dist,
    gaussian,
    get_2DFFT,
    line,
    line_profile,
    log,
    lorentz,
    parabola,
    plot_map,
    rayleigh,
    second_poly,
    sine,
    skewed_gauss,
)

__all__ = [
    # scans
    "SPMScan",
    "AsylumScan",
    "AsylumDART",
    "AsylumSF",
    "BrukerScan",
    "Innova",
    "load_spm",
    "load_files",
    "read_usid_h5",
    "convert_to_h5",
    # channels
    "Channel",
    "QUANTITY_UNITS",
    "DEFAULT_CMAPS",
    "parse_channel_name",
    "canonical_key",
    # readers
    "read_ibw",
    "parse_note",
    "channel_labels",
    "is_ibw_file",
    # widgets
    "file_preview",
    "edge_detection_widget",
    "Innova_edge_detection_widget",
    # utils
    "LineProfile",
    "line_profile",
    "plot_map",
    "get_2DFFT",
    "gaussian",
    "Gauss2d",
    "skewed_gauss",
    "rayleigh",
    "exp_dist",
    "lorentz",
    "line",
    "parabola",
    "second_poly",
    "cubic",
    "exp",
    "log",
    "sine",
    "cosine",
]
