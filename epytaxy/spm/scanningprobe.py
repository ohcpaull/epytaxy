"""
Scanning probe microscopy image classes.

Every file format is read into the same :class:`SPMScan` container -- a mapping
of channel name to :class:`~epytaxy.spm.channels.Channel` plus the scan
metadata -- and all of the analysis and plotting lives on that base class.  The
format-specific subclasses (:class:`AsylumDART`, :class:`AsylumSF`,
:class:`Innova`, :class:`BrukerScan`) therefore only have to answer "what
channels are in this file and how big was the scan".

Notes
-----
Compared with the original implementation:

* ``.ibw`` files are read directly with :mod:`igor2` (see
  :mod:`epytaxy.spm.igor_ibw`); there is no ``pycroscopy``/``pyUSID``
  dependency and no need to convert anything to HDF5 first.  HDF5 files
  produced by the old ``convert_to_h5`` are still readable via
  :func:`read_usid_h5`, using plain :mod:`h5py`.
* Channels are looked up by the quantity recorded in the file rather than by
  positional index.  The old reader assumed exactly six channels in a fixed
  order, so it raised on files holding a different set (trace *and* retrace, a
  Z-sensor channel, a plain AC scan), and in the branch for files with more
  than ten groups it returned a second topography channel in the frequency
  slot -- the equivalent branch in ``AsylumSF`` referenced an undefined
  variable and always raised ``NameError``.
* X and Y are treated independently instead of assuming a square scan.
* Data is held as ``[row, col] = [Y, X]``, the standard image convention, so
  images and line profiles are no longer transposed relative to each other.
* Automatic z-scaling falls back to a robust percentile range when the
  histogram fit fails instead of raising.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Iterator, Sequence

import h5py
import numpy as np
import scipy.stats
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from epytaxy.spm.channels import Channel, canonical_key, parse_channel_name
from epytaxy.spm.igor_ibw import read_ibw
from epytaxy.spm.utils import LineProfile, gaussian, plot_map

__all__ = [
    "SPMScan",
    "AsylumScan",
    "AsylumDART",
    "AsylumSF",
    "Innova",
    "BrukerScan",
    "load_spm",
    "load_files",
    "read_usid_h5",
    "convert_to_h5",
]


class SPMScan:
    """
    A scanning probe microscopy image with one or more data channels.

    Parameters
    ----------
    channels : dict of str to Channel
        Data channels, keyed by :func:`~epytaxy.spm.channels.canonical_key`
        (``"Topography"``, ``"Amplitude_1"``, ``"Phase_2"``, ...).
    metadata : dict, optional
        Scan parameters as recorded by the microscope.
    x_size, y_size : float, optional
        Physical extent of the scan along the fast (X) and slow (Y) axis, in
        metres.  Defaults to the pixel count when unknown.
    filename : str, optional
        Name of the file the scan was read from.

    Attributes
    ----------
    channels : dict of str to Channel
        The data channels.
    metadata : dict
        Scan parameters.
    x_vec, y_vec : numpy.ndarray
        Axis coordinates in micrometres, of length ``n_cols``/``n_rows``.
    channel_edges : dict of str to numpy.ndarray
        Masked arrays populated by :meth:`edge_detection`.
    line_slices : dict of str to LineProfile
        Line profiles populated by :meth:`line_profile`.

    Notes
    -----
    ``zrange`` arguments and the values returned by :meth:`get_map_range` are
    always in the channel's *display* units (nm for topography, pm for
    amplitude, degrees for phase) -- the units written on the colourbar.
    """

    def __init__(
        self,
        channels: dict[str, Channel],
        metadata: dict[str, Any] | None = None,
        x_size: float | None = None,
        y_size: float | None = None,
        filename: str = "",
    ) -> None:
        if not channels:
            raise ValueError("a scan needs at least one data channel")

        self.channels: dict[str, Channel] = dict(channels)
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.filename = filename

        shapes = {channel.data.shape for channel in self.channels.values()}
        if len(shapes) > 1:
            raise ValueError(f"channels have inconsistent shapes: {sorted(shapes)}")
        self.n_rows, self.n_cols = shapes.pop()

        # Scan extents in metres; fall back to pixel counts when unknown.
        self.x_size = float(x_size) if x_size else float(self.n_cols)
        self.y_size = float(y_size) if y_size else float(self.n_rows)

        # Axis vectors in micrometres, for plotting.
        self.x_vec = np.linspace(0, self.x_size / 1e-6, self.n_cols)
        self.y_vec = np.linspace(0, self.y_size / 1e-6, self.n_rows)

        self.channel_edges: dict[str, np.ndarray] = {}
        self.line_slices: dict[str, LineProfile] = {}
        self.mask: np.ndarray | None = None

    # ------------------------------------------------------------------
    # mapping-style access
    # ------------------------------------------------------------------
    def keys(self) -> list[str]:
        """Names of the available channels."""
        return list(self.channels)

    def __iter__(self) -> Iterator[str]:
        return iter(self.channels)

    def __len__(self) -> int:
        return len(self.channels)

    def __contains__(self, key) -> bool:
        try:
            self._resolve_key(key)
        except (KeyError, TypeError):
            return False
        return True

    def __getitem__(self, key) -> Channel:
        return self.channel(key)

    def _resolve_key(self, key) -> str:
        """
        Map a user-supplied channel name onto a key of :attr:`channels`.

        Accepts the canonical key (``"Amplitude_1"``), the label as written by
        the microscope (``"Amplitude1Retrace"``), a case-insensitive variant, and
        the un-suffixed quantity (``"Amplitude"``) when only one harmonic of it
        was recorded.
        """
        if not isinstance(key, str):
            raise TypeError(f"channel name should be a string, got {type(key)}")

        if key in self.channels:
            return key

        # The label as saved by the microscope.
        for name, channel in self.channels.items():
            if channel.name == key:
                return name

        # Case-insensitive match on either.
        lowered = key.lower()
        for name, channel in self.channels.items():
            if lowered in (name.lower(), channel.name.lower()):
                return name

        # "Amplitude" when the file only holds one amplitude harmonic, or
        # "Amplitude_1" when the file holds an un-numbered amplitude.
        quantity, harmonic, _ = parse_channel_name(key)
        matches = [
            name
            for name, channel in self.channels.items()
            if channel.quantity == quantity
            and (harmonic is None or channel.harmonic == harmonic)
        ]
        if len(matches) == 1:
            return matches[0]

        raise KeyError(
            f"unknown channel {key!r}; available channels are {self.keys()}"
        )

    def channel(self, channel) -> Channel:
        """
        Return a :class:`~epytaxy.spm.channels.Channel`.

        Parameters
        ----------
        channel : str or Channel or numpy.ndarray
            A channel name (see :meth:`_resolve_key`), a channel, or a bare 2D
            array -- which is wrapped in an unnamed channel so that arrays
            derived from a scan can be passed to the plotting methods.

        Returns
        -------
        Channel
        """
        if isinstance(channel, Channel):
            return channel
        if isinstance(channel, np.ndarray):
            for existing in self.channels.values():
                if existing.data is channel:
                    return existing
            return Channel(name="Data", data=channel, quantity="", scale=1.0)
        return self.channels[self._resolve_key(channel)]

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        """The channel data as plain arrays, keyed like :attr:`channels`."""
        return {name: channel.data for name, channel in self.channels.items()}

    def add_channel(self, channel: Channel, key: str | None = None) -> Channel:
        """
        Add ``channel`` to the scan, returning it.

        The channel's :attr:`~epytaxy.spm.channels.Channel.key` is set to the
        key it is stored under, so a derived channel (a levelled or transformed
        copy) does not report the same key as the channel it came from.
        """
        if channel.data.shape != (self.n_rows, self.n_cols):
            raise ValueError(
                f"channel shape {channel.data.shape} does not match scan shape "
                f"{(self.n_rows, self.n_cols)}"
            )
        key = key or channel.key
        channel.key = key
        self.channels[key] = channel
        return channel

    def __getattr__(self, name: str) -> Any:
        """
        Expose channels as lowercase attributes, e.g. ``scan.amplitude_1``.

        Keeps ``scan.topography``, ``scan.phase_2`` etc. from the original API
        working without hard-coding one channel layout.
        """
        # __getattr__ is only called when normal lookup fails, but guard against
        # recursion during unpickling/construction.
        if name.startswith("_") or "channels" not in self.__dict__:
            raise AttributeError(name)
        for key, channel in self.__dict__["channels"].items():
            if name == key.lower():
                return channel.data
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    # ------------------------------------------------------------------
    # back-compatible aliases
    # ------------------------------------------------------------------
    @property
    def axis_length(self) -> int:
        """Number of pixels along the fast (X) axis."""
        return self.n_cols

    @property
    def pos_max(self) -> float:
        """Scan size along the fast (X) axis, in micrometres."""
        return self.x_size / 1e-6

    @property
    def factors(self) -> dict[str, float]:
        """SI -> display scale factor of each channel."""
        return {name: channel.scale for name, channel in self.channels.items()}

    @property
    def line_slice(self) -> dict[str, LineProfile]:
        """Alias of :attr:`line_slices`."""
        return self.line_slices

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} {self.filename or 'unnamed'}: "
            f"{self.n_rows}x{self.n_cols} px, "
            f"{self.pos_max:.3g}x{self.y_size / 1e-6:.3g} um, "
            f"channels={self.keys()}>"
        )

    def __str__(self) -> str:
        lines = [
            f"{type(self).__name__}: {self.filename or 'unnamed'}",
            f"  scan size : {self.pos_max:.4g} x {self.y_size / 1e-6:.4g} um",
            f"  resolution: {self.n_cols} x {self.n_rows} px",
        ]
        for key in ("ImagingMode", "MicroscopeModel", "ScanRate", "Date"):
            if key in self.metadata:
                lines.append(f"  {key:10}: {self.metadata[key]}")
        lines.append("  channels  :")
        for name, channel in self.channels.items():
            lines.append(f"    {name:16} {channel.name:22} [{channel.display_units}]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # z-scaling
    # ------------------------------------------------------------------
    def get_map_range(
        self,
        channel="Topography",
        func=gaussian,
        plot: bool = False,
        nsig: float = 4.0,
        method: str = "hist",
        percentile: tuple[float, float] = (1.0, 99.0),
        bins: int = 200,
        p0: Sequence[float] | None = None,
    ) -> tuple[float, float]:
        """
        Choose a sensible z-range for plotting a channel.

        Parameters
        ----------
        channel : str or Channel or numpy.ndarray
            Channel to analyse.
        func : callable, optional
            Peak function fitted to the data histogram when ``method="hist"``.
            Must have the signature ``func(x, amplitude, centre, width, offset)``
            unless ``p0`` is given as well.
        plot : bool, optional
            Plot the histogram and the fit.
        nsig : float, optional
            Number of fitted standard deviations to keep either side of the
            histogram peak.
        method : {"hist", "percentile", "minmax"}, optional
            How to pick the range.  ``"hist"`` fits ``func`` to the histogram
            and takes ``centre +/- nsig * width``, which gives good contrast on
            topography; it silently falls back to ``"percentile"`` if the fit
            does not converge.  ``"percentile"`` is robust for any channel and
            ``"minmax"`` uses the full data range.
        percentile : tuple of float, optional
            Lower and upper percentile used by ``method="percentile"``.
        bins : int, optional
            Number of histogram bins.
        p0 : sequence of float, optional
            Initial parameters for the fit.  Estimated from the histogram when
            not given.

        Returns
        -------
        zmin, zmax : tuple of float
            Range in the channel's display units, clipped to the data extent.
        """
        data = self.channel(channel)
        values = data.display.ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"channel {data.key!r} contains no finite values")

        low, high = float(values.min()), float(values.max())

        if method == "minmax":
            zmin, zmax = low, high
        elif method == "percentile":
            zmin, zmax = (float(v) for v in np.percentile(values, percentile))
        elif method == "hist":
            counts, edges = np.histogram(values, bins=bins)
            centres = 0.5 * (edges[:-1] + edges[1:])
            if p0 is None:
                p0 = [
                    float(counts.max()),
                    float(centres[np.argmax(counts)]),
                    float(np.std(values)) or 1.0,
                    float(counts.min()),
                ]
            try:
                popt, _ = curve_fit(func, centres, counts, p0=list(p0), maxfev=10000)
            except (RuntimeError, ValueError, TypeError) as exc:
                warnings.warn(
                    f"histogram fit failed for channel {data.key!r} ({exc}); "
                    "falling back to a percentile range",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self.get_map_range(
                    channel, method="percentile", percentile=percentile
                )
            centre, width = float(popt[1]), abs(float(popt[2]))
            zmin, zmax = centre - nsig * width, centre + nsig * width

            if plot:
                _, ax = plt.subplots()
                ax.hist(values, bins=bins)
                ax.plot(centres, func(centres, *popt), "k-")
                ax.set(xlabel=data.label, ylabel="Counts")
        else:
            raise ValueError(
                f"method should be 'hist', 'percentile' or 'minmax', got {method!r}"
            )

        # An amplitude cannot be negative.
        if data.quantity == "Amplitude":
            zmin = max(zmin, 0.0)

        # Never claim a range wider than the data.
        zmin, zmax = max(zmin, low), min(zmax, high)
        if zmin >= zmax:  # degenerate (e.g. a constant channel)
            zmin, zmax = low, high if high > low else low + 1.0
        return float(zmin), float(zmax)

    def describe(self, channel="Topography"):
        """Summary statistics of a channel, in display units."""
        return scipy.stats.describe(self.channel(channel).display.ravel())

    def roughness(self, channel="Topography") -> dict[str, float]:
        """
        Surface roughness of a channel, in its display units.

        Returns
        -------
        dict
            ``Ra`` (mean absolute deviation), ``Rq``/RMS (standard deviation)
            and ``Rz`` (peak-to-valley).
        """
        values = self.channel(channel).display
        values = values[np.isfinite(values)]
        centred = values - values.mean()
        return {
            "Ra": float(np.abs(centred).mean()),
            "Rq": float(np.sqrt((centred**2).mean())),
            "Rz": float(values.max() - values.min()),
        }

    # ------------------------------------------------------------------
    # processing
    # ------------------------------------------------------------------
    def level(self, channel="Topography", order: int = 1, lines: bool = False, key=None):
        """
        Remove a polynomial background from a channel.

        Parameters
        ----------
        channel : str or Channel
            Channel to level.
        order : int, optional
            Polynomial order.  ``1`` removes a tilted plane, ``2`` also removes
            bow.
        lines : bool, optional
            Fit and subtract each scan line independently instead of a single
            2D surface.  Use this to remove line-to-line offsets.
        key : str, optional
            Key to store the result under.  Defaults to
            ``"<channel>_levelled"``.

        Returns
        -------
        Channel
            The levelled channel, also added to :attr:`channels`.
        """
        source = self.channel(channel)
        data = source.data

        if lines:
            columns = np.arange(self.n_cols)
            levelled = np.empty_like(data)
            for row, values in enumerate(data):
                finite = np.isfinite(values)
                if finite.sum() <= order:
                    levelled[row] = values
                    continue
                coefficients = np.polyfit(columns[finite], values[finite], order)
                levelled[row] = values - np.polyval(coefficients, columns)
        else:
            rows, columns = np.mgrid[: self.n_rows, : self.n_cols]
            terms = [
                rows.ravel() ** i * columns.ravel() ** j
                for i in range(order + 1)
                for j in range(order + 1 - i)
            ]
            design = np.stack(terms, axis=1)
            flat = data.ravel()
            finite = np.isfinite(flat)
            coefficients, *_ = np.linalg.lstsq(design[finite], flat[finite], rcond=None)
            levelled = (flat - design @ coefficients).reshape(data.shape)

        result = Channel(
            name=f"{source.name}Levelled",
            data=levelled,
            quantity=source.quantity,
            units=source.units,
            display_units=source.display_units,
            scale=source.scale,
            harmonic=source.harmonic,
            direction=source.direction,
            metadata={"levelled_from": source.key, "order": order, "lines": lines},
        )
        return self.add_channel(result, key or f"{source.key}_levelled")

    def fft(self, channel="Topography", key=None):
        """
        Add the magnitude of the 2D Fourier transform of a channel to the scan.

        Parameters
        ----------
        channel : str or Channel
            Channel to transform.
        key : str, optional
            Key to store the result under.  Defaults to ``"FFT_<channel>"``.

        Returns
        -------
        Channel
            The transformed channel, also added to :attr:`channels`.
        """
        source = self.channel(channel)
        # Transform the data in display units so the magnitudes are of order
        # one rather than of order 1e-9.
        magnitude = np.abs(np.fft.fftshift(np.fft.fft2(source.display)))
        result = Channel(
            name=f"FFT_{source.name}",
            data=magnitude,
            quantity="FFT",
            units=f"{source.display_units} px",
            display_units="a.u.",
            scale=1.0,
            metadata={"fft_of": source.key},
        )
        return self.add_channel(result, key or f"FFT_{source.key}")

    def get_2D_fft(self, channel="Topography") -> np.ndarray:
        """Deprecated alias of :meth:`fft` that returns the array."""
        return self.fft(channel).data

    def edge_detection(
        self,
        channel="Topography",
        sigma: float = 0.2,
        low_threshold: float | None = None,
        high_threshold: float | None = None,
    ) -> np.ndarray:
        """
        Find edges in a channel with a Canny filter.

        Parameters
        ----------
        channel : str or Channel
            Channel to run edge detection on.
        sigma : float, optional
            Width of the Gaussian smoothing applied before differentiating.
        low_threshold, high_threshold : float, optional
            Hysteresis thresholds, in the channel's display units.  Default to
            the 10th/90th percentile of the data gradient chosen by
            :mod:`skimage`.

        Returns
        -------
        numpy.ndarray
            A masked array of zeros, masked everywhere except on an edge, ready
            to overlay on an image plot.  Also stored in
            :attr:`channel_edges`.
        """
        import skimage.feature

        data = self.channel(channel)
        edges = skimage.feature.canny(
            image=data.display,
            sigma=sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        overlay = np.ma.masked_array(np.zeros(edges.shape), np.invert(edges))
        self.channel_edges[data.key] = overlay
        return overlay

    # ------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------
    def plot(
        self,
        channel="Topography",
        cmap=None,
        units: str | None = None,
        zrange: tuple[float, float] | None = None,
        ax=None,
        fig=None,
        rotate: bool = False,
        log: bool = False,
        show_edges: bool = False,
        **kwargs,
    ):
        """
        Plot a single data channel.

        Parameters
        ----------
        channel : str or Channel or numpy.ndarray
            Channel to plot.
        cmap : str or matplotlib.colors.Colormap, optional
            Colour map.  Defaults to the channel's
            :attr:`~epytaxy.spm.channels.Channel.cmap`.
        units : str, optional
            Units to plot in.  Defaults to the channel's display units.
            Understood values are ``"m"``, ``"um"``, ``"nm"``, ``"pm"``,
            ``"deg"``, ``"Hz"``, ``"kHz"``.
        zrange : tuple of float, optional
            ``(zmin, zmax)`` in ``units``.  Chosen by :meth:`get_map_range`
            when not given.
        ax, fig : matplotlib axes and figure, optional
            Where to draw.  Created when not given.
        rotate : bool, optional
            Rotate the image by 90 degrees.
        log : bool, optional
            Plot ``log10`` of the data -- useful for Fourier transforms.
        show_edges : bool, optional
            Overlay the edges found by :meth:`edge_detection`.
        **kwargs
            Passed to :func:`~epytaxy.spm.utils.plot_map`, and from there to
            :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        fig, ax, im_handle
        """
        data = self.channel(channel)
        scale = data.scale if units is None else _unit_scale(units, data)
        label = data.display_units if units is None else units

        image = data.data / scale
        x_vec, y_vec = self.x_vec, self.y_vec
        if rotate:
            image = np.rot90(image)
            x_vec, y_vec = y_vec, x_vec
        if log:
            with np.errstate(divide="ignore", invalid="ignore"):
                image = np.log10(np.abs(image))
            image = np.where(np.isfinite(image), image, np.nan)
            label = f"log10({label})" if label else "log10"

        if ax is None:
            fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
        elif fig is None:
            fig = ax.get_figure()

        if zrange is None and not log:
            zmin, zmax = self.get_map_range(data)
            zmin, zmax = zmin * data.scale / scale, zmax * data.scale / scale
        elif zrange is None:
            zmin, zmax = np.nanmin(image), np.nanmax(image)
        else:
            zmin, zmax = zrange

        kwargs.setdefault("cmap", cmap if cmap is not None else data.cmap)
        im_handle, _ = plot_map(
            ax,
            image,
            cbar_label=label,
            vmin=zmin,
            vmax=zmax,
            x_vec=x_vec,
            y_vec=y_vec,
            **kwargs,
        )
        ax.set_xlabel("X ($\\mathrm{\\mu}$m)")
        ax.set_ylabel("Y ($\\mathrm{\\mu}$m)")

        if show_edges and data.key in self.channel_edges:
            ax.imshow(
                self.channel_edges[data.key],
                cmap="binary_r",
                interpolation="none",
                origin=kwargs.get("origin", "lower"),
            )

        return fig, ax, im_handle

    def single_image_plot(self, channel="Topography", **kwargs):
        """Alias of :meth:`plot`, kept for backwards compatibility."""
        return self.plot(channel, **kwargs)

    def plot_channels(
        self,
        channels: Sequence[str] | None = None,
        titles: Sequence[str] | None = None,
        cmap=None,
        zrange=None,
        fig=None,
        axes=None,
        gs=None,
        figsize=None,
        **kwargs,
    ):
        """
        Plot several channels side by side, each with its own colourbar.

        Parameters
        ----------
        channels : sequence of str, optional
            Channels to plot.  Defaults to every channel in the scan.
        titles : sequence of str, optional
            Title for each subplot.  Defaults to the channel labels.
        cmap : str, optional
            Colour map for every subplot.  Defaults to each channel's own.
        zrange : dict or sequence, optional
            ``(zmin, zmax)`` per channel, either keyed by channel name or in the
            same order as ``channels``.  Chosen by :meth:`get_map_range` when
            not given.
        fig, axes : matplotlib figure and sequence of axes, optional
            Where to draw.  Created when not given.
        gs : matplotlib.gridspec.GridSpec, optional
            Grid to place the axes on when ``fig`` is given but ``axes`` is not.
        figsize : tuple of float, optional
            Figure size.  Defaults to ``(4 * n_channels, 4)``.
        **kwargs
            Passed to :meth:`plot`.

        Returns
        -------
        fig, axes
        """
        if channels is None:
            channels = self.keys()
        channels = list(channels)
        resolved = [self.channel(channel) for channel in channels]

        if isinstance(zrange, dict):
            ranges = [
                zrange.get(name, zrange.get(channel.key))
                for name, channel in zip(channels, resolved)
            ]
        elif zrange is None:
            ranges = [None] * len(resolved)
        else:
            ranges = list(zrange)
            if len(ranges) != len(resolved):
                raise ValueError(
                    f"got {len(ranges)} z-ranges for {len(resolved)} channels"
                )

        if axes is None:
            if fig is None:
                fig = plt.figure(
                    figsize=figsize or (4 * len(resolved), 4), constrained_layout=True
                )
            if gs is None:
                gs = GridSpec(1, len(resolved), figure=fig)
            axes = [fig.add_subplot(gs[index]) for index in range(len(resolved))]
        else:
            axes = list(np.ravel(axes))
            if fig is None:
                fig = axes[0].get_figure()

        for index, (channel, z) in enumerate(zip(resolved, ranges)):
            self.plot(channel, ax=axes[index], fig=fig, cmap=cmap, zrange=z, **kwargs)
            title = titles[index] if titles is not None else channel.label
            axes[index].set_title(title)

        return fig, axes

    def multi_image_plot(self, channels=None, titles=None, **kwargs):
        """Alias of :meth:`plot_channels`, kept for backwards compatibility."""
        kwargs.pop("axis", None)
        kwargs.pop("posn", None)
        kwargs.pop("ntick", None)
        return self.plot_channels(channels, titles=titles, **kwargs)

    def line_profile(self, channel="Topography", width: float = 50.0, **kwargs):
        """
        Interactively take a line profile across a channel.

        Click once on the image to set the start of the profile and a second
        time to set the end; the integrated profile is drawn in the right hand
        panel and stored in :attr:`line_slices`.  A third click clears it.
        Requires an interactive matplotlib backend (``%matplotlib widget`` or
        ``%matplotlib qt`` in a notebook).

        Parameters
        ----------
        channel : str or Channel
            Channel to profile.
        width : float, optional
            Width in nanometres to integrate the profile over, perpendicular to
            the line.
        **kwargs
            Passed to :meth:`plot`.

        Returns
        -------
        fig, axes
        """
        data = self.channel(channel)
        color = kwargs.pop("color", "black")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        self.plot(data, fig=fig, ax=axes[0], **kwargs)

        # Width from nanometres to pixels along the fast axis.
        nm_per_pixel = self.x_size / 1e-9 / self.n_cols
        width_px = max(int(round(width / nm_per_pixel)), 1)

        scatters: list = []
        artists: list = []
        clicks: list[tuple[float, float]] = []

        def onclick(event):
            if event.inaxes is not axes[0] or event.xdata is None:
                return

            if len(clicks) >= 2:
                for scatter in scatters:
                    scatter.remove()
                for artist in artists:
                    artist.remove()
                scatters.clear()
                artists.clear()
                clicks.clear()
                axes[1].clear()
                fig.canvas.draw_idle()
                return

            scatters.append(axes[0].scatter(event.xdata, event.ydata, s=5, color=color))
            clicks.append((event.xdata, event.ydata))

            if len(clicks) == 2:
                profile = LineProfile(
                    clicks[0], clicks[1], width=width_px, color=color
                )
                profile.cut_channel(data.data)
                profile.s_dist = np.linspace(
                    0, self._pixel_distance(clicks[0], clicks[1]), len(profile.line_profile)
                )

                axes[1].plot(
                    profile.s_dist,
                    profile.line_profile / data.scale,
                    color="k",
                    label=f"{data.key} line profile",
                )
                axes[1].set(
                    xlabel="Distance ($\\mathrm{\\mu}$m)",
                    ylabel=data.label,
                    title="Line profile",
                )
                profile._plot_over_channel(axes[0])
                artists.extend(
                    [profile.cpatch_line, profile.cpatch_i, profile.cpatch_f]
                )
                self.line_slices[data.key] = profile

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", onclick)
        return fig, axes

    def _pixel_distance(self, start, end) -> float:
        """Distance in micrometres between two points given in pixels."""
        dx = (end[0] - start[0]) / self.n_cols * self.x_size / 1e-6
        dy = (end[1] - start[1]) / self.n_rows * self.y_size / 1e-6
        return float(np.hypot(dx, dy))

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def to_hdf5(self, path, overwrite: bool = False) -> str:
        """
        Write the scan to a plain, self-describing HDF5 file.

        One dataset per channel, with the units and scan metadata as attributes.
        Unlike the USID files the old ``convert_to_h5`` produced, this needs
        nothing but :mod:`h5py` to read back.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file.
        overwrite : bool, optional
            Overwrite ``path`` if it already exists.

        Returns
        -------
        str
            The path written to.
        """
        path = os.fspath(path)
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")

        with h5py.File(path, "w") as handle:
            handle.attrs["format"] = "epytaxy-spm"
            handle.attrs["class"] = type(self).__name__
            handle.attrs["filename"] = self.filename
            handle.attrs["x_size"] = self.x_size
            handle.attrs["y_size"] = self.y_size
            # h5py iterates group members alphabetically, so record the order
            # explicitly to round-trip it.
            handle.attrs["channel_order"] = list(self.channels)

            metadata = handle.create_group("metadata")
            for key, value in self.metadata.items():
                try:
                    metadata.attrs[key] = value
                except TypeError:
                    metadata.attrs[key] = str(value)

            channels = handle.create_group("channels")
            for key, channel in self.channels.items():
                dataset = channels.create_dataset(key, data=channel.data)
                dataset.attrs.update(
                    {
                        "name": channel.name,
                        "quantity": channel.quantity,
                        "units": channel.units,
                        "display_units": channel.display_units,
                        "scale": channel.scale,
                    }
                )
                if channel.harmonic is not None:
                    dataset.attrs["harmonic"] = channel.harmonic
                if channel.direction is not None:
                    dataset.attrs["direction"] = channel.direction

        return path

    @classmethod
    def from_hdf5(cls, path) -> "SPMScan":
        """Read a scan written by :meth:`to_hdf5`."""
        with h5py.File(os.fspath(path), "r") as handle:
            if handle.attrs.get("format") != "epytaxy-spm":
                raise ValueError(
                    f"{os.fspath(path)!r} was not written by SPMScan.to_hdf5; "
                    "use read_usid_h5 for files from the old convert_to_h5"
                )
            group = handle["channels"]
            order = [
                key.decode() if isinstance(key, bytes) else str(key)
                for key in handle.attrs.get("channel_order", [])
            ]
            keys = [key for key in order if key in group]
            keys += [key for key in group if key not in keys]

            channels = {}
            for key in keys:
                dataset = group[key]
                attrs = dict(dataset.attrs)
                channels[key] = channel = Channel(
                    name=attrs.get("name", key),
                    data=dataset[()],
                    quantity=attrs.get("quantity", ""),
                    units=attrs.get("units", ""),
                    display_units=attrs.get("display_units", ""),
                    scale=float(attrs.get("scale", 1.0)),
                    harmonic=(
                        int(attrs["harmonic"]) if "harmonic" in attrs else None
                    ),
                    direction=attrs.get("direction"),
                )
                channel.key = key
            return SPMScan(
                channels=channels,
                metadata=dict(handle["metadata"].attrs),
                x_size=float(handle.attrs["x_size"]),
                y_size=float(handle.attrs["y_size"]),
                filename=str(handle.attrs.get("filename", "")),
            )


def _unit_scale(units: str, channel: Channel) -> float:
    """SI -> ``units`` divisor for a channel."""
    scales = {
        "m": 1.0,
        "mm": 1e-3,
        "um": 1e-6,
        "µm": 1e-6,
        "nm": 1e-9,
        "pm": 1e-12,
        "deg": 1.0,
        "degrees": 1.0,
        "rad": 1.0,
        "Hz": 1.0,
        "kHz": 1e3,
        "MHz": 1e6,
        "V": 1.0,
        "mV": 1e-3,
        "A": 1.0,
        "nA": 1e-9,
        "pA": 1e-12,
        "a.u.": 1.0,
        "": 1.0,
    }
    try:
        return scales[units]
    except KeyError:
        raise ValueError(
            f"unknown units {units!r} for channel {channel.key!r}; "
            f"expected one of {sorted(scales)}"
        ) from None


# ----------------------------------------------------------------------
# Asylum Research
# ----------------------------------------------------------------------
def read_usid_h5(path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Read an HDF5 file written by the old pyUSID-based ``IgorIBWTranslator``.

    Implemented with plain :mod:`h5py` so previously converted data stays
    readable without ``pyUSID`` installed.  Channels are identified by the
    ``quantity`` attribute recorded on each dataset rather than by their
    position in the file, which is what the original reader got wrong.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the ``.h5`` file.

    Returns
    -------
    channels : dict of str to numpy.ndarray
        Channel label -> 2D data, indexed ``[row, col] = [Y, X]``.
    metadata : dict
        Scan parameters from the ``Measurement_000`` group attributes.
    """
    path = os.fspath(path)
    with h5py.File(path, "r") as handle:
        groups = sorted(k for k in handle if k.startswith("Measurement_"))
        if not groups:
            raise ValueError(
                f"{path!r} has no Measurement_* group; it does not look like a "
                "file written by convert_to_h5"
            )
        measurement = handle[groups[0]]
        metadata = {
            key: value.decode() if isinstance(value, bytes) else value
            for key, value in measurement.attrs.items()
        }

        n_rows = int(metadata.get("ScanLines", 0))
        n_cols = int(metadata.get("ScanPoints", 0))

        channels: dict[str, np.ndarray] = {}
        for name in sorted(k for k in measurement if k.startswith("Channel_")):
            dataset = measurement[name]["Raw_Data"]
            label = dataset.attrs.get("quantity", name)
            if isinstance(label, bytes):
                label = label.decode()
            flat = np.asarray(dataset[()], dtype=float).ravel()

            rows, cols = n_rows, n_cols
            if rows * cols != flat.size:
                side = int(round(np.sqrt(flat.size)))
                if side * side != flat.size:
                    raise ValueError(
                        f"cannot reshape channel {label!r} of {flat.size} points "
                        f"into a {n_rows}x{n_cols} image"
                    )
                rows = cols = side
            channels[str(label)] = flat.reshape(rows, cols)

        if not channels:
            raise ValueError(f"{path!r} contains no Channel_* datasets")

    return channels, metadata


class AsylumScan(SPMScan):
    """
    An Asylum Research (Cypher/MFP-3D) scan.

    Reads ``.ibw`` files directly, and ``.h5`` files written either by
    :meth:`SPMScan.to_hdf5` or by the old pyUSID-based ``convert_to_h5``.

    Parameters
    ----------
    fname : str or os.PathLike
        File to read.
    data_dir : str or os.PathLike, optional
        Directory ``fname`` lives in.

    See Also
    --------
    AsylumDART : dual-frequency (DART / dual-AC) PFM scans.
    AsylumSF : single-frequency PFM scans.
    load_spm : pick the right class automatically.
    """

    #: Channels a subclass requires; checked after reading.
    _required: tuple[str, ...] = ()

    def __init__(self, fname, data_dir=".") -> None:
        path = os.path.join(os.fspath(data_dir), os.fspath(fname))
        extension = os.path.splitext(path)[1].lower()

        if extension == ".ibw":
            raw, metadata, kind = read_ibw(path)
            if kind != "image":
                raise ValueError(
                    f"{path!r} holds a force curve, not an image; force curves "
                    "are not handled by AsylumScan"
                )
        elif extension == ".h5":
            raw, metadata = read_usid_h5(path)
        else:
            raise ValueError(
                f"unsupported Asylum file type {extension!r}; expected '.ibw' or '.h5'"
            )

        channels = _build_channels(raw)
        x_size, y_size = _asylum_scan_size(metadata)

        super().__init__(
            channels=channels,
            metadata=metadata,
            x_size=x_size,
            y_size=y_size,
            filename=os.path.basename(path),
        )
        self.filepath = path

        missing = [key for key in self._required if key not in self.channels]
        if missing:
            raise ValueError(
                f"{os.path.basename(path)} is missing the channels {missing} "
                f"expected by {type(self).__name__}; it has {self.keys()}. "
                "Use load_spm() to pick the right class automatically."
            )

    @property
    def is_dart(self) -> bool:
        """Whether the scan was taken in a dual-frequency (DART) mode."""
        if int(self.metadata.get("DualACMode", 0)) == 1:
            return True
        return any(channel.harmonic == 2 for channel in self.channels.values())

    @property
    def drive_frequencies(self) -> tuple[float, ...]:
        """Drive frequencies used, in hertz."""
        keys = ("DriveFrequency", "DriveFrequency1", "DriveFrequency2")
        return tuple(
            float(self.metadata[key]) for key in keys if key in self.metadata
        )


def _build_channels(raw: dict[str, np.ndarray]) -> dict[str, Channel]:
    """
    Turn ``label -> array`` into ``canonical key -> Channel``.

    When a file holds both the trace and retrace of the same quantity the first
    one keeps the plain key and the second is suffixed with its scan direction,
    so nothing is silently dropped.
    """
    channels: dict[str, Channel] = {}
    for label, array in raw.items():
        channel = Channel(name=label, data=array)
        key = channel.key
        if key in channels:
            key = f"{key}_{channel.direction or len(channels)}"
        channel.key = key
        channels[key] = channel
    return channels


def _asylum_scan_size(metadata: dict[str, Any]) -> tuple[float | None, float | None]:
    """Fast (X) and slow (Y) scan size in metres, from the scan parameters."""

    def get(*keys):
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, (int, float)) and np.isfinite(value) and value > 0:
                return float(value)
        return None

    x_size = get("FastScanSize", "ScanSize", "Initial FastScanSize")
    y_size = get("SlowScanSize", "ScanSize", "Initial SlowScanSize")
    return x_size, y_size


class AsylumDART(AsylumScan):
    """
    A dual-frequency (DART / dual-AC) Asylum PFM scan.

    Two drive frequencies straddle the cantilever contact resonance, giving an
    amplitude and phase channel per frequency plus a resonance frequency
    channel.

    Parameters
    ----------
    fname : str or os.PathLike
        ``.ibw`` file from the microscope, or ``.h5`` file from the old
        ``convert_to_h5``.
    data_dir : str or os.PathLike, optional
        Directory ``fname`` lives in.

    Attributes
    ----------
    topography : numpy.ndarray
        Height channel, in metres.
    amplitude_1, amplitude_2 : numpy.ndarray
        Amplitude at each drive frequency, in metres.
    phase_1, phase_2 : numpy.ndarray
        Phase at each drive frequency, in degrees.
    frequency : numpy.ndarray
        Resonance frequency channel, in hertz.

    Examples
    --------
    >>> scan = AsylumDART("P0125OP0000.h5", data_dir="PFM analysis/data")
    >>> scan.keys()
    ['Topography', 'Amplitude_1', 'Amplitude_2', 'Phase_1', 'Phase_2', 'Frequency']
    >>> fig, axes = scan.plot_channels(["Topography", "Amplitude_1", "Phase_1"])
    """

    _required = ("Topography", "Amplitude_1", "Phase_1", "Amplitude_2", "Phase_2")


class AsylumSF(AsylumScan):
    """
    A single-frequency Asylum PFM scan.

    One amplitude and one phase channel, driven at a fixed frequency.

    Parameters
    ----------
    fname : str or os.PathLike
        ``.ibw`` file from the microscope, or ``.h5`` file from the old
        ``convert_to_h5``.
    data_dir : str or os.PathLike, optional
        Directory ``fname`` lives in.

    Attributes
    ----------
    topography : numpy.ndarray
        Height channel, in metres.
    amplitude : numpy.ndarray
        Response amplitude, in metres.
    phase : numpy.ndarray
        Response phase, in degrees.
    """

    _required = ("Topography",)

    def __init__(self, fname, data_dir=".") -> None:
        super().__init__(fname, data_dir=data_dir)
        if self.is_dart:
            raise ValueError(
                f"{self.filename} is a dual-frequency (DART) scan; use AsylumDART "
                "or load_spm() instead"
            )
        if not any(channel.quantity == "Amplitude" for channel in self.channels.values()):
            raise ValueError(
                f"{self.filename} has no amplitude channel; it has {self.keys()}"
            )


# ----------------------------------------------------------------------
# Bruker
# ----------------------------------------------------------------------
class Innova(SPMScan):
    """
    A Bruker Innova ``.flt`` AFM image.

    Parameters
    ----------
    filename : str or os.PathLike
        The ``.flt`` file.
    data_dir : str or os.PathLike, optional
        Directory ``filename`` lives in.

    Attributes
    ----------
    data : numpy.ndarray
        The height channel, in metres.

    Notes
    -----
    The header is parsed as ``key=value`` lines rather than by position, so
    files re-saved by Nanoscope (which reorders the header) still read -- the
    original implementation depended on the exact line order.  It is also a
    :class:`SPMScan` now rather than a wrapper around one, so the whole scan API
    (indexing, iteration, ``len``) works on it.
    """

    def __init__(self, filename, data_dir=".") -> None:
        self.filepath = os.path.join(os.fspath(data_dir), os.fspath(filename))

        metadata = self.get_metadata()
        data, x_size, y_size = self._read_data(metadata)

        super().__init__(
            channels={
                "Topography": Channel(
                    name=str(metadata.get("dataname", "Height")),
                    data=data,
                    quantity="Topography",
                    units="m",
                )
            },
            metadata=metadata,
            x_size=x_size,
            y_size=y_size,
            filename=os.path.basename(os.fspath(filename)),
        )

    @property
    def data(self) -> np.ndarray:
        """The height channel, in metres."""
        return self.channels["Topography"].data

    def get_metadata(self) -> dict[str, Any]:
        """
        Parse the text header of the ``.flt`` file.

        Returns
        -------
        dict
            Every ``key=value`` pair in the header.  Numeric values are
            converted; values carrying a unit (``"10 um"``) are kept as strings
            and also stored as ``"<key>_value"``/``"<key>_unit"``.
        """
        with open(self.filepath, "rb") as handle:
            preamble = handle.read(65536)

        metadata: dict[str, Any] = {}
        offset = None
        for line in preamble.split(b"\n"):
            key, sep, value = line.decode("ISO-8859-1").partition("=")
            if not sep:
                continue
            key, value = key.strip(), value.strip().rstrip("\r")
            if not key:
                continue
            metadata[key] = value
            if key == "data_offset":
                offset = int(float(value))

        if offset is None:
            raise ValueError(
                f"no 'data_offset' entry in the header of {self.filepath!r}; "
                "this does not look like an Innova .flt file"
            )
        metadata["data_offset"] = offset

        # Split "10 um" style values and coerce plain numbers.
        for key, value in list(metadata.items()):
            if not isinstance(value, str):
                continue
            parts = value.split()
            if len(parts) == 2:
                try:
                    metadata[f"{key}_value"] = float(parts[0])
                    metadata[f"{key}_unit"] = parts[1]
                    continue
                except ValueError:
                    pass
            try:
                metadata[key] = float(value) if "." in value or "e" in value.lower() else int(value)
            except ValueError:
                pass

        return metadata

    def _read_data(
        self, metadata: dict[str, Any]
    ) -> tuple[np.ndarray, float | None, float | None]:
        """Read the binary image and work out the scan extents, in metres."""
        n_cols = int(metadata["resolutionX"])
        n_rows = int(metadata["resolutionY"])

        with open(self.filepath, "rb") as handle:
            flat = np.fromfile(
                handle, dtype="float32", offset=int(metadata["data_offset"])
            )
        if flat.size < n_rows * n_cols:
            raise ValueError(
                f"{self.filepath!r} holds {flat.size} points, expected "
                f"{n_rows * n_cols} for a {n_cols}x{n_rows} image"
            )

        # z_transfer_coeff is given per volt with a trailing unit string.
        z_coefficient = metadata.get("z_transfer_coeff_value")
        if z_coefficient is None:
            z_coefficient = float(str(metadata["z_transfer_coeff"]).split()[0])
        z_unit = str(metadata.get("z_transfer_coeff_unit", "um"))

        data = flat[: n_rows * n_cols].reshape(n_rows, n_cols) * z_coefficient
        data = data * _LENGTH_UNITS.get(z_unit.split("/")[0], 1e-6)

        x_size = _length_in_metres(metadata, "scan_range_X")
        y_size = _length_in_metres(metadata, "scan_range_Y")
        return data, x_size, y_size

    def to_pyspm(self):
        """
        Wrap the image in a :class:`pySPM.SPM.SPM_image`.

        Gives access to the pySPM levelling helpers (``correct_lines``,
        ``corr_fit2d``, ``correct_slope``).  :meth:`SPMScan.level` covers the
        common cases without the dependency.
        """
        from pySPM.SPM import SPM_image

        return SPM_image(
            self.data / 1e-9,
            channel=str(self.metadata.get("dataname", "Height")),
            real={
                "x": self.x_size / 1e-6,
                "y": self.y_size / 1e-6,
                "unit": "um",
            },
            zscale="nm",
        )


#: Length unit -> metres.
_LENGTH_UNITS = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9, "pm": 1e-12}


def _length_in_metres(metadata: dict[str, Any], key: str) -> float | None:
    """Read a ``"10 um"`` style metadata entry as metres."""
    value = metadata.get(f"{key}_value")
    if value is None:
        return None
    unit = str(metadata.get(f"{key}_unit", "um"))
    return float(value) * _LENGTH_UNITS.get(unit, 1e-6)


class BrukerScan(SPMScan):
    """
    A Bruker Nanoscope ``.spm`` scan, read with :mod:`pySPM`.

    Parameters
    ----------
    fname : str or os.PathLike
        The ``.spm`` file.
    data_dir : str or os.PathLike, optional
        Directory ``fname`` lives in.
    channels : sequence of str, optional
        Channel names to read.  Every channel in the file is read by default.

    Notes
    -----
    The original code read Bruker files inside ``AsylumDART``, which meant an
    Asylum object could end up holding Bruker channels.  They are separate
    classes now.
    """

    def __init__(self, fname, data_dir=".", channels=None) -> None:
        import pySPM

        path = os.path.join(os.fspath(data_dir), os.fspath(fname))
        handle = pySPM.Bruker(path)

        names = list(channels) if channels is not None else _bruker_channel_names(handle)

        built: dict[str, Channel] = {}
        x_size = y_size = None
        for name in names:
            try:
                layer = handle.get_channel(name)
            except Exception as exc:  # pragma: no cover - depends on the file
                warnings.warn(
                    f"could not read Bruker channel {name!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            quantity, _, _ = parse_channel_name(name.replace(" ", ""))
            scale = 1e-9 if quantity in ("Topography", "ZSensor") else None
            data = np.asarray(layer.pixels, dtype=float)
            if scale is not None:
                data = data * scale  # pySPM returns nm for height channels

            channel = Channel(
                name=name.replace(" ", ""),
                data=data,
                quantity=quantity,
                metadata={"bruker_channel": name},
            )
            built[canonical_key(channel.quantity, channel.harmonic)] = channel

            extent = layer.get_extent()
            x_size = abs(extent[1] - extent[0]) * 1e-6
            y_size = abs(extent[3] - extent[2]) * 1e-6

        if not built:
            raise ValueError(f"no readable channels found in {path!r}")

        super().__init__(
            channels=built,
            metadata={"scan_type": "Bruker", "channels": names},
            x_size=x_size,
            y_size=y_size,
            filename=os.path.basename(path),
        )
        self.filepath = path


def _bruker_channel_names(handle) -> list[str]:
    """Names of the image channels in an open :class:`pySPM.Bruker` file."""
    names = []
    for layer in getattr(handle, "layers", []):
        raw = layer.get(b"@2:Image Data", layer.get("@2:Image Data"))
        if not raw:
            continue
        text = raw[0].decode() if isinstance(raw[0], bytes) else str(raw[0])
        if '"' in text:
            names.append(text.split('"')[1])
    return names or ["Height", "Deflection Error"]


# ----------------------------------------------------------------------
# loading helpers
# ----------------------------------------------------------------------
def load_spm(fname, data_dir="."):
    """
    Read a scanning probe file, choosing the right class from its contents.

    Parameters
    ----------
    fname : str or os.PathLike
        File to read.  ``.ibw`` and ``.h5`` are read as Asylum scans, ``.spm``
        as a Bruker scan and ``.flt`` as an Innova image.
    data_dir : str or os.PathLike, optional
        Directory ``fname`` lives in.

    Returns
    -------
    AsylumDART or AsylumSF or BrukerScan or Innova

    Examples
    --------
    >>> scan = load_spm("P0125OP0000.h5", data_dir="PFM analysis/data")
    >>> type(scan).__name__
    'AsylumDART'
    """
    path = os.path.join(os.fspath(data_dir), os.fspath(fname))
    extension = os.path.splitext(path)[1].lower()

    if extension in (".ibw", ".h5"):
        if extension == ".h5" and _is_epytaxy_h5(path):
            return SPMScan.from_hdf5(path)
        scan = AsylumScan(path)
        target = AsylumDART if scan.is_dart else AsylumSF
        # Re-tag rather than re-read: the file has already been parsed.
        scan.__class__ = target
        missing = [key for key in target._required if key not in scan.channels]
        if missing:
            scan.__class__ = AsylumScan
        return scan
    if extension == ".spm":
        return BrukerScan(path)
    if extension == ".flt":
        return Innova(path)
    raise ValueError(
        f"unsupported file type {extension!r}; expected '.ibw', '.h5', '.spm' or '.flt'"
    )


def _is_epytaxy_h5(path) -> bool:
    """Whether ``path`` was written by :meth:`SPMScan.to_hdf5`."""
    try:
        with h5py.File(os.fspath(path), "r") as handle:
            return handle.attrs.get("format") == "epytaxy-spm"
    except OSError:
        return False


def load_files(directory, pattern=(".ibw", ".h5")) -> tuple[list, list[str]]:
    """
    Load every scanning probe file in a directory.

    Parameters
    ----------
    directory : str or os.PathLike
        Directory to read.
    pattern : tuple of str, optional
        File extensions to include.

    Returns
    -------
    scans : list
        The loaded scans.  Files that fail to read are skipped with a warning.
    filenames : list of str
        Names of the files that were read, in the same order as ``scans``.

    Notes
    -----
    The old version of this function returned open :class:`h5py.File` handles
    that were never closed; it returns scan objects now.
    """
    scans, filenames = [], []
    for name in sorted(os.listdir(os.fspath(directory))):
        if not name.lower().endswith(tuple(pattern)):
            continue
        try:
            scans.append(load_spm(name, data_dir=directory))
        except Exception as exc:
            warnings.warn(f"skipping {name}: {exc}", RuntimeWarning, stacklevel=2)
            continue
        filenames.append(name)
    return scans, filenames


def convert_to_h5(directory, overwrite: bool = False) -> list[str]:
    """
    Convert every ``.ibw`` file in a directory to HDF5.

    .. deprecated::
        ``.ibw`` files are read directly now, so there is no need to convert
        anything -- pass the ``.ibw`` file to :func:`load_spm` or
        :class:`AsylumDART` instead.  This function is kept because the
        resulting files are convenient to share; it writes the plain format of
        :meth:`SPMScan.to_hdf5`, not the old USID layout.

    Parameters
    ----------
    directory : str or os.PathLike
        Directory holding the ``.ibw`` files.
    overwrite : bool, optional
        Replace existing ``.h5`` files.

    Returns
    -------
    list of str
        Paths of the files written.
    """
    warnings.warn(
        "convert_to_h5 is no longer needed: .ibw files are read directly by "
        "load_spm/AsylumDART/AsylumSF.",
        DeprecationWarning,
        stacklevel=2,
    )
    written = []
    for name in sorted(os.listdir(os.fspath(directory))):
        if not name.lower().endswith(".ibw"):
            continue
        destination = os.path.join(directory, os.path.splitext(name)[0] + ".h5")
        if os.path.exists(destination) and not overwrite:
            continue
        scan = AsylumScan(name, data_dir=directory)
        written.append(scan.to_hdf5(destination, overwrite=overwrite))
    return written
