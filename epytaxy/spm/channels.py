"""
Channel abstraction shared by every scanning probe reader in :mod:`epytaxy.spm`.

The original ``AsylumDART``/``AsylumSF`` classes carried a data channel around as
a bare 2D array plus a handful of parallel dictionaries (``channels``,
``factors``, ``line_slice``, ``px``, ``py``, ...) that all had to be kept in
sync by hand and were hard-coded to one particular set of channel names.  A
:class:`Channel` bundles the array with everything needed to plot and interpret
it, so a scan is simply a mapping of name -> :class:`Channel` and any channel a
microscope happens to save is supported without editing a lookup table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "Channel",
    "QUANTITY_UNITS",
    "DEFAULT_CMAPS",
    "parse_channel_name",
    "canonical_key",
]


#: ``quantity -> (SI unit, display unit, SI -> display divisor)``.
#:
#: SPM data is stored in SI units (metres, degrees, hertz, ...) but is only ever
#: *looked* at in a scaled unit -- nanometres for topography, picometres for a
#: piezo response amplitude and so on.  Each channel keeps both so plots can be
#: labelled correctly without the caller juggling factors of 1e-9.
QUANTITY_UNITS: dict[str, tuple[str, str, float]] = {
    "Topography": ("m", "nm", 1e-9),
    "Height": ("m", "nm", 1e-9),
    "ZSensor": ("m", "nm", 1e-9),
    "Amplitude": ("m", "pm", 1e-12),
    "Phase": ("deg", "deg", 1.0),
    "Frequency": ("Hz", "kHz", 1e3),
    "Deflection": ("V", "V", 1.0),
    "Current": ("A", "pA", 1e-12),
    "Potential": ("V", "V", 1.0),
    "Lateral": ("m", "nm", 1e-9),
    "Friction": ("V", "V", 1.0),
    "Count": ("", "", 1.0),
    "UserIn": ("V", "V", 1.0),
    "FFT": ("", "", 1.0),
}

#: Default colour map per quantity.  ``afmhot`` everywhere matches the
#: behaviour of the original implementation; override entries here (or pass
#: ``cmap=`` explicitly) to change it globally.
DEFAULT_CMAPS: dict[str, str] = {
    "Topography": "afmhot",
    "Amplitude": "afmhot",
    "Phase": "afmhot",
    "Frequency": "viridis",
}

#: Quantities for which a trailing digit in the channel label is the drive
#: harmonic (``Amplitude1``, ``Phase2``) rather than part of the name
#: (``UserIn0``).
_HARMONIC_QUANTITIES = frozenset({"Amplitude", "Phase", "Frequency", "Freq"})

#: Labels that all mean "the height image".
_TOPOGRAPHY_ALIASES = frozenset({"Height", "Topo", "Topography", "HeightSensor"})


def parse_channel_name(label: str) -> tuple[str, int | None, str | None]:
    """
    Split a microscope channel label into its parts.

    Asylum Research encodes three things in one string: the physical quantity,
    the drive harmonic used to measure it (DART/dual-AC scans drive at two
    frequencies) and the scan direction.

    Parameters
    ----------
    label : str
        Channel label as saved by the microscope, e.g. ``"Amplitude2Retrace"``.

    Returns
    -------
    quantity : str
        Physical quantity, e.g. ``"Amplitude"``.  ``"Height"`` is normalised to
        ``"Topography"``.
    harmonic : int or None
        Drive harmonic, or ``None`` for channels that do not have one.
    direction : {"Trace", "Retrace", None}
        Scan direction.

    Examples
    --------
    >>> parse_channel_name("Amplitude2Retrace")
    ('Amplitude', 2, 'Retrace')
    >>> parse_channel_name("HeightTrace")
    ('Topography', None, 'Trace')
    >>> parse_channel_name("UserIn0")
    ('UserIn0', None, None)
    """
    name = str(label).strip()

    direction = None
    for suffix in ("Retrace", "Trace"):
        if name.endswith(suffix):
            direction = suffix
            name = name[: -len(suffix)]
            break

    harmonic = None
    if len(name) > 1 and name[-1].isdigit() and name[:-1] in _HARMONIC_QUANTITIES:
        harmonic = int(name[-1])
        name = name[:-1]

    if name in _TOPOGRAPHY_ALIASES:
        name = "Topography"

    return name, harmonic, direction


def canonical_key(quantity: str, harmonic: int | None = None) -> str:
    """
    Build the dictionary key a channel is stored under.

    Keys are ``"Topography"``, ``"Amplitude_1"``, ``"Phase_2"``, ... which keeps
    the names used by the pre-existing notebooks working.
    """
    if harmonic is None:
        return quantity
    return f"{quantity}_{harmonic}"


@dataclass
class Channel:
    """
    One 2D data channel of a scanning probe image.

    Parameters
    ----------
    name : str
        Label as saved by the microscope, e.g. ``"Amplitude1Retrace"``.
    data : numpy.ndarray
        2D array of the channel in SI units, indexed ``data[row, col]`` with
        ``row`` along the slow (Y) axis and ``col`` along the fast (X) axis.
        This is the standard image convention, so the array can be handed
        straight to :func:`matplotlib.pyplot.imshow` or
        :func:`skimage.measure.profile_line` without transposing.
    quantity : str, optional
        Physical quantity.  Inferred from ``name`` when not given.
    units, display_units : str, optional
        SI and plotting units.  Inferred from ``quantity`` when not given.
    scale : float, optional
        Divisor taking SI values to display values.  Inferred from ``quantity``
        when not given.
    harmonic : int, optional
        Drive harmonic the channel was measured at.
    direction : {"Trace", "Retrace"}, optional
        Scan direction.
    metadata : dict, optional
        Anything the reader wants to keep, e.g. the units string as recorded in
        the file (which for Asylum HDF5 files is frequently wrong).

    Attributes
    ----------
    display : numpy.ndarray
        ``data / scale`` -- the channel in ``display_units``.
    """

    name: str
    data: np.ndarray
    quantity: str = ""
    units: str = ""
    display_units: str = ""
    scale: float = float("nan")
    harmonic: int | None = None
    direction: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Key the channel is stored under.  Defaults to the canonical key; a scan
    #: overrides it when a channel is stored elsewhere (a levelled copy of
    #: ``"Topography"`` is stored as ``"Topography_levelled"``, and must not
    #: report the same key as the channel it came from).
    _key: str | None = None

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError(
                f"channel {self.name!r} data should be 2D, got shape {self.data.shape}"
            )

        parsed_quantity, parsed_harmonic, parsed_direction = parse_channel_name(
            self.name
        )
        if not self.quantity:
            self.quantity = parsed_quantity
        if self.harmonic is None:
            self.harmonic = parsed_harmonic
        if self.direction is None:
            self.direction = parsed_direction

        si, display, scale = QUANTITY_UNITS.get(self.quantity, ("", "", 1.0))
        if not self.units:
            self.units = si
        if not self.display_units:
            self.display_units = display
        if not np.isfinite(self.scale):
            self.scale = scale

    # -- basic properties -------------------------------------------------
    @property
    def key(self) -> str:
        """Key this channel is stored under in :attr:`SPMScan.channels`."""
        if self._key is not None:
            return self._key
        return canonical_key(self.quantity, self.harmonic)

    @key.setter
    def key(self, value: str | None) -> None:
        self._key = value

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def display(self) -> np.ndarray:
        """The data converted to :attr:`display_units`."""
        return self.data / self.scale

    @property
    def label(self) -> str:
        """Axis/colourbar label, e.g. ``"Amplitude 1 (pm)"``."""
        name = self.quantity
        if self.harmonic is not None:
            name = f"{name} {self.harmonic}"
        if self.display_units:
            return f"{name} ({self.display_units})"
        return name

    @property
    def cmap(self) -> str:
        """Default colour map for this quantity."""
        return DEFAULT_CMAPS.get(self.quantity, "afmhot")

    def to_display(self, values):
        """Convert SI ``values`` to :attr:`display_units`."""
        return np.asarray(values, dtype=float) / self.scale

    def from_display(self, values):
        """Convert ``values`` given in :attr:`display_units` to SI."""
        return np.asarray(values, dtype=float) * self.scale

    def __array__(self, dtype=None, copy=None):
        """Allow a Channel to be used directly wherever an array is expected."""
        if copy is False:
            return self.data if dtype is None else self.data.astype(dtype, copy=False)
        if dtype is None:
            return self.data
        return self.data.astype(dtype)

    def __repr__(self) -> str:
        return (
            f"<Channel {self.key!r} ({self.name}) shape={self.data.shape} "
            f"units={self.units!r}>"
        )
