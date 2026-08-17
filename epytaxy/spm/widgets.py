"""
Interactive Jupyter widgets for scanning probe data.

These need ``ipywidgets`` and an interactive matplotlib backend
(``%matplotlib widget``).  They work with any :class:`~epytaxy.spm.SPMScan`
subclass rather than only with the Asylum classes, and the sliders operate in
the channel's display units so the numbers on them match the colourbar.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from epytaxy.spm.scanningprobe import load_spm

__all__ = ["file_preview", "edge_detection_widget", "Innova_edge_detection_widget"]


def _widgets():
    """Import :mod:`ipywidgets` lazily with a helpful error message."""
    try:
        import ipywidgets
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the interactive widgets need 'ipywidgets'. Install it with "
            "`pip install ipywidgets`."
        ) from None
    return ipywidgets


def file_preview(directory, extensions=(".ibw", ".h5"), **kwargs):
    """
    Browse the scans in a directory with dropdown menus.

    Parameters
    ----------
    directory : str or os.PathLike
        Directory to preview.
    extensions : tuple of str, optional
        File extensions to offer.
    **kwargs
        Passed to :meth:`~epytaxy.spm.SPMScan.plot_channels`.

    Returns
    -------
    ipywidgets.Widget
        The interactive control, displayed automatically in a notebook.

    Notes
    -----
    The old implementation guessed the file type with a bare ``try``/``except``
    around ``AsylumDART``, which hid genuine read errors and reported every
    failure as "SF detected".  The type now comes from the scan metadata via
    :func:`~epytaxy.spm.load_spm`.
    """
    widgets = _widgets()

    filenames = sorted(
        name
        for name in os.listdir(os.fspath(directory))
        if name.lower().endswith(tuple(extensions))
    )
    if not filenames:
        raise FileNotFoundError(f"no {extensions} files in {os.fspath(directory)!r}")

    file_dropdown = widgets.Dropdown(options=filenames, description="file")
    channel_dropdown = widgets.Dropdown(options=[], description="channels")
    output = widgets.Output()

    # Channel groupings offered per file, so a DART scan can be flipped between
    # its two drive frequencies the way the old widget did.
    def channel_sets(scan):
        sets = {}
        for harmonic in (1, 2):
            keys = [
                key
                for key in scan.keys()
                if scan.channels[key].harmonic == harmonic
            ]
            if keys:
                sets[f"Channel {harmonic}"] = ["Topography", *keys]
        sets["All"] = scan.keys()
        for key in scan.keys():
            sets[key] = [key]
        return sets

    state: dict = {}

    def load(name):
        scan = load_spm(name, data_dir=directory)
        state["scan"] = scan
        state["sets"] = channel_sets(scan)
        channel_dropdown.options = list(state["sets"])
        return scan

    def draw(*_):
        scan, sets = state.get("scan"), state.get("sets")
        if scan is None:
            return
        selection = channel_dropdown.value
        if selection not in sets:
            return

        # Close the previous figure so browsing a directory does not leave one
        # open per file viewed.
        previous = state.pop("fig", None)
        if previous is not None:
            plt.close(previous)

        with output:
            output.clear_output(wait=True)
            fig, _ = scan.plot_channels(sets[selection], **kwargs)
            fig.suptitle(f"{scan.filename} ({type(scan).__name__})")
            state["fig"] = fig
            plt.show()

    def on_file_change(change):
        load(change["new"])
        draw()

    file_dropdown.observe(on_file_change, names="value")
    channel_dropdown.observe(draw, names="value")

    load(file_dropdown.value)
    draw()

    return widgets.VBox([widgets.HBox([file_dropdown, channel_dropdown]), output])


def edge_detection_widget(
    scan,
    channel="Topography",
    zrange=None,
    sigma: float = 0.2,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    figsize=(5, 5),
    **kwargs,
):
    """
    Tune Canny edge detection on a channel interactively.

    Parameters
    ----------
    scan : SPMScan
        Scan to work on.  Any subclass works, including :class:`Innova`.
    channel : str, optional
        Channel to run edge detection on.
    zrange : tuple of float, optional
        ``(zmin, zmax)`` for the image, in the channel's display units.  Also
        sets the range of the two threshold sliders.  Chosen by
        :meth:`~epytaxy.spm.SPMScan.get_map_range` when not given.
    sigma : float, optional
        Initial Gaussian smoothing width.
    low_threshold, high_threshold : float, optional
        Initial hysteresis thresholds, in display units.  Default to 10% and 90%
        of ``zrange``.
    figsize : tuple of float, optional
        Figure size.
    **kwargs
        Passed to :meth:`~epytaxy.spm.SPMScan.plot`.

    Returns
    -------
    ipywidgets.Widget
        The interactive control.  The last mask computed is left in
        ``scan.channel_edges``.

    Notes
    -----
    Fixes three faults in the original: the update callback shadowed the scan
    object with a local of the same name (so moving a slider raised
    ``UnboundLocalError``), the overlay was hard-coded to 512x512 pixels, and
    ``cmap`` was read from a global that was never set.
    """
    widgets = _widgets()

    # Accept the argument names used by the pre-0.1 version of this function.
    for old, new in (
        ("sigmainit", "sigma"),
        ("lowthresinit", "low_threshold"),
        ("highthresinit", "high_threshold"),
    ):
        if old in kwargs:
            value = kwargs.pop(old)
            if new == "sigma":
                sigma = value
            elif new == "low_threshold":
                low_threshold = value
            else:
                high_threshold = value

    data = scan.channel(channel)
    if zrange is None:
        zrange = scan.get_map_range(data, nsig=5)
    zmin, zmax = zrange
    span = zmax - zmin

    if low_threshold is None:
        low_threshold = zmin + 0.1 * span
    if high_threshold is None:
        high_threshold = zmin + 0.9 * span

    kwargs.setdefault("cmap", data.cmap)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    overlay: dict = {}

    def render(sigma, low_threshold, high_threshold):
        ax.clear()
        scan.plot(data, fig=fig, ax=ax, zrange=(zmin, zmax), show_cbar=False, **kwargs)
        mask = scan.edge_detection(
            data,
            sigma=sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        overlay["mask"] = mask
        ax.imshow(mask, cmap="binary_r", interpolation="none", origin="lower")
        fig.canvas.draw_idle()

    step = span / 200 if span else 0.01
    control = widgets.interactive(
        render,
        sigma=widgets.FloatSlider(description="sigma", min=0, max=10, value=sigma),
        low_threshold=widgets.FloatSlider(
            description="low threshold",
            min=zmin,
            max=zmax,
            step=step,
            value=low_threshold,
        ),
        high_threshold=widgets.FloatSlider(
            description="high threshold",
            min=zmin,
            max=zmax,
            step=step,
            value=high_threshold,
        ),
    )
    return control


def Innova_edge_detection_widget(scan, **kwargs):
    """
    Deprecated alias of :func:`edge_detection_widget`.

    :func:`edge_detection_widget` handles :class:`~epytaxy.spm.Innova` scans
    directly now, so the separate function is no longer needed.
    """
    kwargs.setdefault("channel", "Topography")
    return edge_detection_widget(scan, **kwargs)
