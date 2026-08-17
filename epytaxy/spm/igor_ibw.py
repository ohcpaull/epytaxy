"""
Reader for Igor Binary Wave (``.ibw``) files written by Asylum Research AFMs.

This replaces the old ``IgorIBWTranslator``, which was a ``pyUSID`` translator
that wrote a USID-formatted HDF5 file to disk before anything could be read back
out of it.  That indirection is what broke when ``pycroscopy``/``pyUSID`` moved
on, and it was never necessary: an ``.ibw`` file is a small container holding a
stack of 2D images, a text "note" of scan parameters and a list of channel
labels.  :func:`read_ibw` parses it straight into numpy arrays with
:mod:`igor2`, so no conversion step, no HDF5 file and no USID dependency.

The HDF5 files produced by the old translator can still be read -- see
:func:`epytaxy.spm.scanningprobe.read_usid_h5`.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

__all__ = ["read_ibw", "parse_note", "channel_labels", "is_ibw_file"]


def _load_igor(path):
    """Import :mod:`igor2` lazily so the rest of the package works without it."""
    try:
        from igor2 import binarywave
    except ImportError:  # pragma: no cover - depends on the environment
        try:
            from igor import binarywave  # type: ignore[no-redef]
        except ImportError:
            raise ImportError(
                "Reading Igor binary wave (.ibw) files requires the 'igor2' "
                "package. Install it with `pip install igor2`."
            ) from None
    return binarywave.load(os.fspath(path))


def is_ibw_file(path) -> bool:
    """Whether ``path`` looks like an Igor binary wave file."""
    return os.path.splitext(os.fspath(path))[1].lower() == ".ibw"


def _decode(value: Any, codec: str = "utf-8") -> Any:
    """Decode a bytestring, falling back to latin-1 for older AR software."""
    if isinstance(value, bytes):
        try:
            return value.decode(codec)
        except UnicodeDecodeError:
            return value.decode("ISO-8859-1")
    return value


def parse_note(wave: dict, codec: str = "utf-8") -> dict[str, Any]:
    """
    Parse the scan parameters out of the ``note`` field of an Igor wave.

    The note is a carriage-return separated list of ``key: value`` pairs holding
    the full state of the microscope at the time of the scan (several hundred
    entries).  Numeric values are converted to :class:`int`/:class:`float`.

    Parameters
    ----------
    wave : dict
        The ``"wave"`` entry of a loaded ``.ibw`` file.
    codec : str, optional
        Codec used to decode bytestrings.

    Returns
    -------
    dict
        Scan parameters, plus ``creationDate``, ``modDate`` and ``bname`` from
        the wave header when present.
    """
    note = _decode(wave.get("note", b""), codec)
    parameters: dict[str, Any] = {}

    for pair in note.rstrip("\r").split("\r"):
        key, sep, value = pair.partition(":")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        if not key:
            continue
        parameters[key] = _coerce(value)

    header = wave.get("wave_header", {})
    for key in ("creationDate", "modDate", "bname"):
        if key in header:
            parameters[key] = _decode(header[key], codec)

    return parameters


def _coerce(value: str) -> Any:
    """Turn a parameter string into an int/float where that is lossless."""
    try:
        number = float(value)
    except ValueError:
        return value
    if not np.isfinite(number):
        return number
    try:
        if number == int(number):
            return int(number)
    except (OverflowError, ValueError):
        pass
    return number


def channel_labels(wave: dict, codec: str = "utf-8") -> list[str]:
    """
    Retrieve the channel labels of an Igor wave.

    Parameters
    ----------
    wave : dict
        The ``"wave"`` entry of a loaded ``.ibw`` file.
    codec : str, optional
        Codec used to decode bytestrings.

    Returns
    -------
    list of str
        One label per channel, e.g.
        ``["HeightRetrace", "Amplitude1Retrace", "Phase1Retrace"]``.
    """
    labels: list[str] = []
    for dimension in wave.get("labels", []):
        for item in dimension:
            item = _decode(item, codec)
            if item:
                labels.append(item)

    # Older AR software appends a redundant "trace"/"retrace" tail to some
    # labels; keep everything up to and including the suffix.
    cleaned = []
    for label in labels:
        position = label.lower().rfind("trace")
        cleaned.append(label[: position + 5] if position > 0 else label)
    return cleaned


def read_ibw(
    path, codec: str = "utf-8"
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    """
    Read an Asylum Research ``.ibw`` file.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the ``.ibw`` file.
    codec : str, optional
        Codec used to decode the parameter and label bytestrings.

    Returns
    -------
    channels : dict of str to numpy.ndarray
        Channel label -> data.  For images each array is 2D and indexed
        ``[row, col]`` with ``row`` along the slow (Y) axis, matching the
        convention documented on :class:`epytaxy.spm.channels.Channel`.  For a
        force curve each array is 1D.
    metadata : dict
        Scan parameters from the wave note (see :func:`parse_note`).
    kind : {"image", "force"}
        Whether the file holds an image stack or a single force curve.

    Raises
    ------
    ImportError
        If :mod:`igor2` is not installed.
    """
    wave = _load_igor(path)["wave"]

    metadata = parse_note(wave, codec)
    labels = channel_labels(wave, codec)
    data = np.asarray(wave["wData"])

    # Some AR versions write an empty leading layer whose label is a null set.
    if len(labels) == data.shape[-1] + 1:
        labels = labels[1:]
    if len(labels) != data.shape[-1]:
        labels = [
            labels[i] if i < len(labels) else f"Channel_{i}"
            for i in range(data.shape[-1])
        ]

    if data.ndim == 3:
        kind = "image"
        # wData is stored [X, Y, channel]; transpose each layer to [Y, X].
        channels = {
            label: np.ascontiguousarray(data[:, :, index].T, dtype=float)
            for index, label in enumerate(labels)
        }
    elif data.ndim == 2:
        kind = "force"
        channels = {
            label: np.asarray(data[:, index], dtype=float)
            for index, label in enumerate(labels)
        }
    else:
        raise ValueError(
            f"unexpected wave shape {data.shape} in {os.fspath(path)!r}; "
            "expected an image stack or a force curve"
        )

    return channels, metadata, kind
