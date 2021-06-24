import os
import h5py
import pyUSID as usid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pycroscopy as px
from scipy.optimize import curve_fit
from ipywidgets import interact 

import ipywidgets as widgets
import skimage.feature
from skimage.measure import profile_line

from matplotlib.patches import ConnectionPatch

from vlabs.spm.igor_ibw import IgorIBWTranslator

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


class AsylumDART(object):
    """
    Class that handles DART Asylum PFM files. 

    Parameters
    ----------
    fname       :   str
        The filename of the scan you want to open
    data_dir    :   str, optional
        The data folder where the file is located

    Attributes
    ----------
    filename        :   str
        Name of Asylum DART file that has been read
    axis_length     :   int
        Length (in pixels) of each axis of the scan. TODO: Assumes a 
        square scan and should be generalised throughout this class to 
        treat x and y independently.
    pos_max         :   numpy.float64
        The size of the scan. Assumes a square scansize.
    x_vec           :   numpy.ndarray
        A 1D array with length ``axis_length`` to define the x and y axes
        of the ``matplotlib.pypot.imshow`` maps
    
    Channels
    --------
        - topography    :   numpy.ndarray
            Topography channel in Asylum DART .h5 file
        - amplitude_1   :   numpy.ndarray
            Amplitude channel for the first drive frequency in the DART file.
        - phase_1       :   numpy.ndarray
            Phase channel for the first drive frequency in the DART file.
        - amplitude_2   :   numpy.ndarray
            Amplitude channel for the second drive frequency in the DART file.
        - phase_2       :   numpy.ndarray
            Phase channel for the second drive frequency in the DART file.
        - frequency     :   numpy.ndarray
            Frequency channel for the DART file.
    
    - channels      :   dict
        A dictionary with all the channels in the file (for ease-of-use).
            "Topography"    :   AsylumDART.topography
            "Amplitude_1"   :   AsylumDART.amplitude_1
            "Phase_1"       :   AsylumDART.phase_1
            "Amplitude_2"   :   AsylumDART.amplitude_2
            "Phase_2"       :   AsylumDART.phase_2
            "Frequency"     :   AsylumDART.frequency
    
    - line_slice    :   dict
        A dictionary with possible line profiles for each channel
            "Topography"    :   numpy.ndarray or None
            "Amplitude_1"   :   numpy.ndarray or None
            "Phase_1"       :   numpy.ndarray or None
            "Amplitude_2"   :   numpy.ndarray or None
            "Phase_2"       :   numpy.ndarray or None
            "Frequency"     :   numpy.ndarray or None
    
    - channel_edges :   dict
        A dictionary of 2D arrays for each channel that are populated after
        using ``AsylumDART.edge_detection``
    
    
    """

    def __init__(self, fname, data_dir = "."):

        if fname.endswith(".ibw"):
            raise ValueError("Need to convert .ibw file to .h5 beforehand. use convert_to_h5(directory)")
        elif fname.endswith(".h5"):
            self.file = h5py.File( os.path.join(data_dir, fname), mode='r' )
        else:
            raise ValueError("filetype needs to be .h5")
        
        self.filename = os.path.basename(fname)
        d = self._get_channels(fmt="arr")
        self.channel_edges = {}
        self.topography = d[0].T
        self.amplitude_1 = d[1].T
        self.amplitude_2 = d[2].T
        self.phase_1 = d[3].T
        self.phase_2 = d[4].T
        self.frequency = d[5].T
        self.channels = dict(
            {
                "Topography" : self.topography,
                "Amplitude_1" : self.amplitude_1,
                "Amplitude_2" : self.amplitude_2,
                "Phase_1" : self.phase_1,
                "Phase_2" : self.phase_2,
                "Frequency" : self.frequency
            }
        )
        self.line_slice =  {
            "Topography" : None,
            "Amplitude_1" : None,
            "Amplitude_2" : None,
            "Phase_1" : None,
            "Phase_2" : None,
            "Frequency" : None
        }
        self.px = {            
            "Topography" : [],
            "Amplitude_1" : [],
            "Amplitude_2" : [],
            "Phase_1" : [],
            "Phase_2" : [],
            "Frequency" : [],
        }
        self.py = {
            "Topography" : [],
            "Amplitude_1" : [],
            "Amplitude_2" : [],
            "Phase_1" : [],
            "Phase_2" : [],
            "Frequency" : [],
        }
        self.line_profile_x = {
            "Topography" : None,
            "Amplitude_1" : None,
            "Amplitude_2" : None,
            "Phase_1" : None,
            "Phase_2" : None,
            "Frequency" : None
        }

        self.axis_length = len(self.topography)
        self.pos_max = usid.hdf_utils.get_attributes(self.file['Measurement_000'])['ScanSize']/10**(-6)
        self.x_vec = np.linspace(0, self.pos_max, self.axis_length)

        self.mask = None

    def get_map_range(self, channel, func=gaussian, plot=False, nsig=4):
        """
        Fits the histogram of a given data channel with a given function and returns
        an appropriate minimum and maximum range based on the width of the histogram peak.

        Parameters
        ----------
        channel :   str or nd.array
            Data channel to get the range from. If channel is a str, 
            then it must be the name of a data channel in the Asylum file
        func    :   obj, optional
            Function object to describe the histogram of the data channel
        plot    :   bool, optional
            Plots histogram and fitted ``func`` to the data
        nsig    :   float
            Number of standard deviations to allow on either side of the
            centre from the fitted data histogram. 

        Returns
        ---------
        zmin, zmax  :   tuple
            A tuple that describes the minimum and maximum z-values for
            the data channel
        """
        # Check if channel is a string specifying the PFM channel to plot
        if type(channel) is str:
            if channel in self.channels.keys():
                data = self.channels[channel]
            else:
                raise ValueError("Channel to plot needs to be one of {}".format(self.channels.keys()))

        # Check if it is the data array of the PFM channel to plot 
        elif isinstance(channel, np.ndarray):
            if channel in self.channels.values():
                data = channel
            else:
                raise ValueError("Channel data does not match loaded channels in class!")

        # Take histogram of data and determine height and centre of histogram. 

        counts, bin_edges = np.histogram(data.flatten(), bins=200)
        bin_centres = np.array([0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(bin_edges)-1)])
        centre = bin_centres[np.argmax(counts)]
        amp = max(counts)
        p0 = [amp, centre, centre, 1e-8]
        # If channel is phase, return standard phase range
        if "Phase" in channel:
            return (0,360)
        # If channel is amplitude, change fitting function to skewed gauss
        if "Amplitude" in channel:
            func=skewed_gauss
            p0 = [amp, centre, 1e-10, 0]
            popt, _ = curve_fit(func, bin_centres, counts, p0=p0)
            centre = popt[1]
            width = popt[2]
            return (0, centre + 2*width)

        popt, _ = curve_fit(func, bin_centres, counts, p0=p0)

        centre = popt[1]
        width = popt[2]
        if plot:
            fig, ax = plt.subplots()
            _ = ax.hist(data.flatten(), bins=200)
            ax.plot(bin_centres, func(bin_centres, *popt))
        return (-nsig*width, nsig*width)

    def apply_mask(self, channel, threshold):
        """
        Applies a mask to a channel based of a specific threshold
        """
        
    def _get_channels(self, fmt="arr"):
        """
        Function that gets the different channels from an Asylum file and returns them 

        TODO: Need to make the assignment of channels more generic, and check for extra channels
        in a better way (in case the user has flattened the topography data or done similar amendments
        to the data)
        """

        topo1 = usid.USIDataset(self.file['Measurement_000/Channel_000/Raw_Data']) 
        ampl1 = usid.USIDataset(self.file['Measurement_000/Channel_001/Raw_Data']) 
        phase1 = usid.USIDataset(self.file['Measurement_000/Channel_002/Raw_Data']) 
        ampl2 = usid.USIDataset(self.file['Measurement_000/Channel_003/Raw_Data']) 
        phase2 = usid.USIDataset(self.file['Measurement_000/Channel_004/Raw_Data']) 
        frequency = usid.USIDataset(self.file['Measurement_000/Channel_005/Raw_Data'])
        
        
        if fmt == 'arr':
            topo1_nd = np.transpose(topo1.get_n_dim_form().squeeze())
            ampl1_nd = np.transpose(ampl1.get_n_dim_form().squeeze())
            ampl2_nd = np.transpose(ampl2.get_n_dim_form().squeeze())
            phase1_nd = np.transpose(phase1.get_n_dim_form().squeeze())
            phase2_nd = np.transpose(phase2.get_n_dim_form().squeeze())
            freq_nd = np.transpose(frequency.get_n_dim_form().squeeze())
        
            if len(self.file['Measurement_000']) > 10:
                topo2 = usid.USIDataset(self.file['Measurement_000/Channel_006/Raw_Data']) 
                topo2_nd = np.transpose(topo2.get_n_dim_form().squeeze())
            
                return [topo1_nd, ampl1_nd, phase1_nd, ampl2_nd, phase2_nd, topo2_nd]
            return [topo1_nd, ampl1_nd, phase1_nd, ampl2_nd, phase2_nd, freq_nd]
            
        
        if len(self.file['Measurement_000']) > 10:
            topo2 = usid.USIDataset(self.file['Measurement_000/Channel_006/Raw_Data']) 
            topo2_nd = np.transpose(topo2.get_n_dim_form().squeeze())
            
            return [topo1, ampl1, phase1, ampl2, phase2, frequency, topo2]
        
        return [topo1, ampl1, phase1, ampl2, phase2, frequency] 
    
    def single_image_plot(
        self, 
        channel="Topography", 
        cmap=None, 
        units="nm", 
        zrange=None, 
        ax=None, 
        fig=None, 
        **kwargs
    ):
        """
        Plots a single data channel from an Asylum datafile. 

        Parameters
        ----------
        channel : str or np.array
                    The key or value of the dictionary of channels in AsylumData.
                    e.g. "Topography", "Amplitude_1", "Phase_2"
        cmap    : str
                    Name of a valid colormap
                    e.g. "afmhot", "magma"
        units   : str
                    Units of channel with which to format the plot with. For topography, 
                    the units are typically nanometres, so the map will be divided by 1e-9
                    and units of "nm" will be put on the colourbar. If degrees, no division
                    takes place but units are displayed on the scalebar. 
        zrange  : tuple

        """
        channel_dict = {
            "Topography" : gaussian,
            "Amplitude_1" : skewed_gauss,
            "Phase_1"     : None,
        }
        # Check if channel is a string specifying the PFM channel to plot
        if type(channel) is str:
            if channel in self.channels.keys():
                data = self.channels[channel]
                data_title = channel
            else:
                raise ValueError("Channel to plot needs to be one of {}".format(self.channels.keys()))
        # Check if it is the data array of the PFM channel to plot 
        elif isinstance(channel, np.ndarray):
            data = channel
        
        factor = 1
        # Instantiate scale factor if needed
        if units == "nm":
            factor = 1e-9
        elif units == "pm":
            factor = 1e-12
        elif units == "degrees":
            factor = 1
            

        if ax is None:
            fig, ax = plt.subplots(figsize=(3,3))   

        if zrange is None:
            (zmin, zmax) = self.get_map_range(channel, channel_dict[channel], **kwargs)
            usid.plot_utils.plot_map(ax, data/factor, cmap=cmap, cbar_label=units, vmin=zmin/factor, vmax=zmax/factor, x_vec = self.x_vec, y_vec= self.x_vec)
            #axis.set_title(title)    
            ax.set_xlabel('X ($\\mathrm{\\mu}$m)')
            ax.set_ylabel('Y ($\\mathrm{\\mu}$m)')
        else:
            
            usid.plot_utils.plot_map(ax, data/factor, cmap=cmap, cbar_label=units, vmin=zrange[0], vmax=zrange[1], x_vec = self.x_vec, y_vec=self.x_vec, **kwargs)
            #axis.set_title(title)        
            ax.set_xlabel('X ($\\mathrm{\\mu}$m)')
            ax.set_ylabel('Y ($\\mathrm{\\mu}$m)')

        return fig, ax

    def line_profile(self, channel, width=50.0, **kwargs):
        """
        Plots a data channel in an interactive jupyter notebook figure.

        Interactive clicking on the data channel plot can create a line profile, 
        which is then plotted in the adjacent window. Clicking on channel 2D plot 
        creates a a startpoint, and a second click creates the end-point for 
        the line profile.

        Parameters
        ----------
        channel     :   str or np.ndarray
            Channel to plot and obtain line profile
        width       :   float
            Width in nanometres to integrate the linescan over. If you think this
            unit of width is silly then come fight me 
        
        """

        if "cmap" not in kwargs.keys():
            kwargs.update(cmap = "afmhot")

        factor = 1
        # Instantiate scale factor and units if needed
        if "Topography" in channel:
            zlabel = "Integrated Height"
            units = "nm"
            factor = 1e-9
        elif "Amplitude" in channel:
            zlabel = "Integrated Amplitude"
            units = "pm"
            factor = 1e-12
        elif "Phase" in channel:
            zlabel = "Integrated Phase"
            units = "degrees"
            factor = 1

        fig, ax = plt.subplots(1,2)
        
        fig, ax[0] = self.single_image_plot(channel, fig=fig, ax=ax[0], **kwargs)

        pos = []
        line = []
        perp1, perp2 = [], []
        self.xyA, self.xyB = (), ()
        # Convert width between nanometres and pixels
        width = int(np.round(width / 1000 / self.pos_max * self.axis_length))
        

        def onclick(event):
            if len(pos) == 0:
                # plot first scatter
                scatter = ax[0].scatter(event.xdata, event.ydata, s=5, color="black")
                pos.append(scatter)
                self.px[channel].append(event.xdata)
                self.py[channel].append(event.ydata)

            elif len(pos) == 1:
                # plot second scatter and line
                scatter = ax[0].scatter(event.xdata, event.ydata, s=5, color="black")
                pos.append(scatter)
                self.px[channel].append(event.xdata)
                self.py[channel].append(event.ydata)
                x_values = [self.px[channel][0], self.px[channel][1]]
                y_values = [self.py[channel][0], self.py[channel][1]]
                ln = ax[0].plot(x_values, y_values, "-", color="black")
                line.append(ln.pop(0))
                # Plot line profile of data
                
                # Get distance of line profile in pixels
                pix_dist = int(np.hypot(x_values[1]-x_values[0], y_values[1]-y_values[0])) 

                sample_distance = pix_dist / self.axis_length * self.pos_max
                # Use skimage.measure.profile_line to slice the data  between selected
                # pixels, with given width.
                self.line_slice[channel] = profile_line(
                    self.channels[channel].T,
                    (x_values[0], y_values[0]),
                    (x_values[1], y_values[1]),
                    linewidth = width,
                ) 

                self.line_profile_x[channel] = np.linspace(0, sample_distance, len(self.line_slice[channel]))

                # Plot line profile in adjacent subplot
                ax[1].plot(
                    self.line_profile_x[channel],
                    self.line_slice[channel] / factor,
                    color="black",
                    label=f"{channel} line profile",
                )
                ax[1].set(xlabel="Distance ($\\mathrm{\\mu}}$m)", ylabel=f"{zlabel} ({units})")
                line_vec = np.array([x_values[1] - x_values[0], y_values[1] - y_values[0]])
                #perp_vec = (-y_values[1] + y_values[0], x_values[1] - x_values[0]) / self.axis_length * self.pos_max

                # Find angle between the specified line and the x-axis
                angle_x = np.angle(line_vec[0] + line_vec[1]*1j, deg=False)

                xy_i = (x_values[0], y_values[0])
                xy_f = (x_values[1], y_values[1])

                # Calculate the offset in X and Y for the linewidth start
                # and end points at the start point for the line profile
                self.xyA_i = (
                    (xy_i[0] - width/2 * np.sin(angle_x)),
                    (xy_i[1] + width/2 * np.cos(angle_x)),
                )
                self.xyB_i = (
                    (xy_i[0] + width/2 * np.sin(angle_x)),
                    (xy_i[1] - width/2 * np.cos(angle_x)),
                )

                p1 = ConnectionPatch(
                    xyA=self.xyA_i,
                    xyB=self.xyB_i,
                    coordsA="data",
                    coordsB="data",
                    axesA=ax[0],
                    axesB=ax[0],
                )
                perp1.append(p1)
                ax[0].add_artist(p1)

                # Calculate the offset in X and Y for the linewidth start
                # and end points at the end point for the line profile
                self.xyA_f = (
                    (xy_f[0] - width/2 * np.sin(angle_x)),
                    (xy_f[1] + width/2 * np.cos(angle_x)),
                )
                self.xyB_f = (
                    (xy_f[0] + width/2 * np.sin(angle_x)),
                    (xy_f[1] - width/2 * np.cos(angle_x)),
                )
                
                p2 = ConnectionPatch(
                    xyA=self.xyA_f,
                    xyB=self.xyB_f,
                    coordsA="data",
                    coordsB="data",
                    axesA=ax[0],
                    axesB=ax[0],
                )
                perp2.append(p2)
                ax[0].add_artist(p2)
            else:
            # clear variables 
                for scatter in pos:
                    scatter.remove()
                
                self.px[channel].clear()
                self.py[channel].clear()
                pos.clear()
                ax[1].clear()
                line[0].remove()
                line.clear()
                perp1[0].remove()
                perp2[0].remove()
                perp1.clear()
                perp2.clear()

            fig.canvas.draw()
    
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

    def multi_image_plot(self, channels, titles=None, cmap=None, 
                     zrange=None, axis=None, fig=None,  gs=None, 
                     posn=None, ntick=4, **kwargs):
        """
        Handy function that plots three image channels side by side with colorbars

        Parameters
        ----------
        channels     : list of str or list of array
                    List of three keys or values for the specified channelsimages defined as 2D numpy arrays
        experiment : list or array
                    The list of datafiles from the PFM measurements
        titles     : (Optional) list or array-like (optional)
                    List of the titles for each image
        cmap       : (Optional) matplotlib.pyplot colormap object or string 
                    Colormap to use for displaying the images
        zrange     : (Optional) list of array-like 
                    List of z_ranges for height, amplitude, and phase images respectively 
        axis       : (Optional) matplotlib axis 
        
        fig        : (Optional) matplotlib figure 

        Returns
        -------
        fig : Figure
            Figure containing the plots
        axes : 1D array_like of axes objects
            Axes of the individual plots within `fig`
        """
        channel_dict = {
            "Topography" : gaussian,
            "Amplitude_1" : skewed_gauss,
            "Phase_1"     : None,
        }
        for channel in channels:
            if type(channel) is str:
                if channel in self.channels.keys():
                    data = self.channels[channel]
                    data_title = channel
                else:
                    raise ValueError("Channel to plot needs to be one of {}".format(self.channels.keys()))

            # Check if it is the data array of the PFM channel to plot 
            elif isinstance(channel, np.ndarray):
                if channel in self.channels.values():
                    data = channel
                    data_title = list(self.channels.keys())[list(self.channels.values()).index(data)]
                else:
                    raise ValueError("Channel data does not match loaded channels in class!")
        
        factors = [1, 1, 1]
        for idx, channel in enumerate(channels):
            if channel == "Topography":
                factors[idx] = 1e-9
            elif channel in ["Amplitude_1", "Amplitude_2"]:
                factors[idx] = 1e-12

        if zrange is None:
            (zmin, zmax) = self.get_map_range("Topography", channel_dict["Topography"])
            (amin, amax) = self.get_map_range("Amplitude_1", channel_dict["Amplitude_1"])
            (pmin, pmax) = self.get_map_range("Phase_1")
            ph1 = -90
            ph2 = ph1 + 360
            zrange = [ (zmin, zmax), (0, amax), (pmin,pmax) ] # Preset height, amplitude, and phase channel max/min

        #[t1, a1, p1, a2, p2] = images
        #channel2 = [ t1, a2, p2 ]
        if titles is None:
            titles = ['topo', 'ampl', 'phase']

        if axis is None:
            fig = plt.figure(figsize=(12,4))   
            gs = GridSpec(1, 3)

        
        axes = []
        ax1 = fig.add_subplot(gs[0])
        usid.plot_utils.plot_map(
            ax1, 
            self.channels[channels[0]]/factors[0], 
            num_ticks=4, 
            vmin=zrange[0][0],
            vmax=zrange[0][1],
            cmap=cmap, 
            x_vec = self.x_vec, 
            y_vec = self.x_vec,
            **kwargs,
        )
        ax1.set_xlabel('X ($\mathrm{\mu}$m)')
        ax1.set_ylabel('Y ($\mathrm{\mu}$m)')

        ax2 = fig.add_subplot(gs[1])
        usid.plot_utils.plot_map(
            ax2, 
            self.channels[channels[1]]/factors[1], 
            num_ticks=4, 
            vmin=zrange[1][0], 
            vmax=zrange[1][1], 
            cmap=cmap, 
            x_vec = self.x_vec, 
            y_vec=self.x_vec,
            **kwargs,
        )
        ax2.set_xlabel('X ($\mathrm{\mu}$m)')

        ax3 = fig.add_subplot(gs[2])
        usid.plot_utils.plot_map(
            ax3, 
            self.channels[channels[2]]/factors[2], 
            num_ticks=4, 
            vmin=zrange[2][0], 
            vmax=zrange[2][1], 
            cmap=cmap, 
            x_vec = self.x_vec,
            y_vec=self.x_vec,
            **kwargs,
        )
        ax3.set_xlabel('X ($\mathrm{\mu}$m)')

        axes = [ax1, ax2, ax3]
        return fig, axes

    def edge_detection(self, channel, sigma, low_threshold, high_threshold):
        # Check if channel is a string specifying the PFM channel to plot
        if type(channel) is str:
            if channel in self.channels.keys():
                data = self.channels[channel]
                data_title = channel
                print(data_title)
            else:
                raise ValueError("Channel to plot needs to be one of {}".format(self.channels.keys()))

        # Check if it is the data array of the PFM channel to plot 
        elif isinstance(channel, np.ndarray):
            for ch in self.channels:
                if channel == self.channels[ch]:
                    data_title = ch
                    print(data_title)
                    data = channel


        edges = skimage.feature.canny(
            image=data,
            sigma=sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )

        self.channel_edges[data_title] = np.ma.masked_array(np.zeros((self.axis_length,self.axis_length)), np.invert(edges))


class AsylumSF(object):
    """
    Class that handles Single-Frequency (SF) Asylum PFM files. 

    Parameters
    ----------
    

    
    """

    def __init__(self, fname, data_dir = "."):

        if fname.endswith(".ibw"):
            raise ValueError("Need to convert .ibw file to .h5 beforehand. use convert_to_h5(directory)")
        elif fname.endswith(".h5"):
            self.file = h5py.File( os.path.join(data_dir, fname), mode='r' )
        else:
            raise ValueError("filetype needs to be .h5")
        
        self.filename = os.path.basename(fname)
        d = self._get_channels(fmt="arr")
        self.topography = d[0].T
        self.amplitude = d[1].T
        self.phase = d[2].T
        self.frequency = d[3].T
        self.channels = dict(
            {
                "Topography" : self.topography,
                "Amplitude" : self.amplitude,
                "Phase" : self.phase,
                "Frequency" : self.frequency
            }
        )
        self.pos_max = usid.hdf_utils.get_attributes(self.file['Measurement_000'])['ScanSize']/10**(-6)
        self.x_vec = np.linspace(0, self.pos_max, len(self.topography))
        self.mask = None

    def get_map_range(self, channel, func=gaussian):
        # Check if channel is a string specifying the PFM channel to plot
        if type(channel) is str:
            if channel in self.channels.keys():
                data = self.channels[channel]
            else:
                raise ValueError("Channel to plot needs to be one of {}".format(self.channels.keys()))

        # Check if it is the data array of the PFM channel to plot 
        elif isinstance(channel, np.ndarray):
            if channel in self.channels.values():
                data = channel
            else:
                raise ValueError("Channel data does not match loaded channels in class!")

        data_hist = np.hist(data)

        popt, pcov = curve_fit()

    def apply_mask(self, channel, threshold):
        """
        Applies a mask to a channel based of a specific threshold
        """
        
    def _get_channels(self, fmt="arr"):
        """
        Function that gets the different channels from an Asylum file and returns them 

        TODO: Need to make the assignment of channels more generic, and check for extra channels
        in a better way (in case the user has flattened the topography data or done similar amendments
        to the data)
        """

        topo1 = usid.USIDataset(self.file['Measurement_000/Channel_000/Raw_Data']) 
        ampl1 = usid.USIDataset(self.file['Measurement_000/Channel_001/Raw_Data']) 
        phase1 = usid.USIDataset(self.file['Measurement_000/Channel_002/Raw_Data']) 
        frequency = usid.USIDataset(self.file['Measurement_000/Channel_003/Raw_Data'])
        
        
        if fmt == 'arr':
            topo1_nd = np.transpose(topo1.get_n_dim_form().squeeze())
            ampl1_nd = np.transpose(ampl1.get_n_dim_form().squeeze())
            phase1_nd = np.transpose(phase1.get_n_dim_form().squeeze())
            freq_nd = np.transpose(frequency.get_n_dim_form().squeeze())
        
            if len(self.file['Measurement_000']) > 10:
                topo2 = usid.USIDataset(self.file['Measurement_000/Channel_006/Raw_Data']) 
                topo2_nd = np.transpose(topo2.get_n_dim_form().squeeze())
            
                return [topo1_nd, ampl1_nd, phase1_nd, ampl2_nd, phase2_nd, topo2_nd]
            return [topo1_nd, ampl1_nd, phase1_nd, freq_nd]
            
        
        if len(self.file['Measurement_000']) > 10:
            topo2 = usid.USIDataset(self.file['Measurement_000/Channel_006/Raw_Data']) 
            topo2_nd = np.transpose(topo2.get_n_dim_form().squeeze())
            
            return [topo1, ampl1, phase1, frequency, topo2]
        
        return [topo1, ampl1, phase1, frequency] 
    
    #@interact(channel=["Topography", "Amplitude_1", "Phase_1", "Amplitude_2", "Phase_2"])
    def single_image_plot(
        self, 
        channel="Topography", 
        cmap=None, 
        units="nm", 
        zrange=None, 
        ax=None, 
        fig=None, 
        **kwargs
    ):
        """
        Plots a single data channel from an Asylum datafile. 

        Parameters
        ----------
        channel : str or np.array
                    The key or value of the dictionary of channels in AsylumData.
                    e.g. "Topography", "Amplitude_1", "Phase_2"
        cmap    : str
                    Name of a valid colormap
                    e.g. "afmhot", "magma"
        units   : str
                    Units of channel with which to format the plot with. For topography, 
                    the units are typically nanometres, so the map will be divided by 1e-9
                    and units of "nm" will be put on the colourbar. If degrees, no division
                    takes place but units are displayed on the scalebar. 
        zrange  : tuple

        """
        # Check if channel is a string specifying the PFM channel to plot
        if type(channel) is str:
            if channel in self.channels.keys():
                data = self.channels[channel]
                data_title = channel
            else:
                raise ValueError("Channel to plot needs to be one of {}".format(self.channels.keys()))

        # Check if it is the data array of the PFM channel to plot 
        elif isinstance(channel, np.ndarray):
            data = channel
        
        factor = 1
        # Instantiate scale factor if needed
        if units == "nm":
            factor = 1e-9
        elif units == "pm":
            factor = 1e-12
            

        if ax is None:
            fig, ax = plt.subplots(figsize=(3,3))   

        if zrange is None:
            #data_hist = np.hist(data)
            #popt, pcov = curve_fit(gaussian, )
            usid.plot_utils.plot_map(ax, data/factor, cmap=cmap, cbar_label=units, x_vec = self.x_vec, y_vec= self.x_vec)
            #axis.set_title(title)    
            ax.set_xlabel('X ($\\mathrm{\\mu}$m)')
            ax.set_ylabel('Y ($\\mathrm{\\mu}$m)')
        else:
            
            usid.plot_utils.plot_map(ax, data/factor, cmap=cmap, cbar_label=units, vmin=zrange[0], vmax=zrange[1], x_vec = self.x_vec, y_vec=self.x_vec, **kwargs)
            #axis.set_title(title)        
            ax.set_xlabel('X ($\\mathrm{\\mu}$m)')
            ax.set_ylabel('Y ($\\mathrm{\\mu}$m)')

        return fig, ax

    def multi_image_plot(self, channels, titles=None, cmap=None, 
                     zrange=None, axis=None, fig=None,  gs=None, 
                     posn=None, ntick=4, **kwargs):
        """
        Handy function that plots three image channels side by side with colorbars

        Parameters
        ----------
        channels     : list of str or list of array
                    List of three keys or values for the specified channelsimages defined as 2D numpy arrays
        experiment : list or array
                    The list of datafiles from the PFM measurements
        titles     : (Optional) list or array-like (optional)
                    List of the titles for each image
        cmap       : (Optional) matplotlib.pyplot colormap object or string 
                    Colormap to use for displaying the images
        zrange     : (Optional) list of array-like 
                    List of z_ranges for height, amplitude, and phase images respectively 
        axis       : (Optional) matplotlib axis 
        
        fig        : (Optional) matplotlib figure 

        Returns
        -------
        fig : Figure
            Figure containing the plots
        axes : 1D array_like of axes objects
            Axes of the individual plots within `fig`
        """

        for channel in channels:
            if type(channel) is str:
                if channel in self.channels.keys():
                    data = self.channels[channel]
                    data_title = channel
                else:
                    raise ValueError("Channel to plot needs to be one of {}".format(self.channels.keys()))

            # Check if it is the data array of the PFM channel to plot 
            elif isinstance(channel, np.ndarray):
                if channel in self.channels.values():
                    data = channel
                    data_title = list(self.channels.keys())[list(self.channels.values()).index(data)]
                else:
                    raise ValueError("Channel data does not match loaded channels in class!")
        
        factors = [1, 1, 1]
        for idx, channel in enumerate(channels):
            if channel == "Topography":
                factors[idx] = 1e-9
            elif channel == "Amplitude":
                factors[idx] = 1e-12
            elif channel == "Phase":
                factors[idx] = 1e-9

        if zrange is None:
            ph1 = -90
            ph2 = ph1 + 360
            zrange = [ (-15, 15), (0, max(self.channels[channels[1]].flatten())/factors[1]), (ph1,ph2) ] # Preset height, amplitude, and phase channel max/min

        #[t1, a1, p1, a2, p2] = images
        #channel2 = [ t1, a2, p2 ]
        if titles is None:
            titles = ['topo', 'ampl', 'phase']

        if axis is None:
            fig = plt.figure(figsize=(12,4))   
            gs = GridSpec(1, 3)

        
        axes = []
        ax1 = fig.add_subplot(gs[0])
        usid.plot_utils.plot_map(
            ax1, 
            self.channels[channels[0]]/factors[0], 
            num_ticks=4, 
            vmin=zrange[0][0],
            vmax=zrange[0][1],
            cmap=cmap, 
            x_vec = self.x_vec, 
            y_vec = self.x_vec,
            **kwargs,
        )
        ax1.set_xlabel('X ($\mathrm{\mu}$m)')
        ax1.set_ylabel('Y ($\mathrm{\mu}$m)')

        ax2 = fig.add_subplot(gs[1])
        usid.plot_utils.plot_map(
            ax2, 
            self.channels[channels[1]]/factors[1], 
            num_ticks=4, 
            vmin=zrange[1][0], 
            vmax=zrange[1][1], 
            cmap=cmap, 
            x_vec = self.x_vec, 
            y_vec=self.x_vec,
            **kwargs,
        )
        ax2.set_xlabel('X ($\mathrm{\mu}$m)')

        ax3 = fig.add_subplot(gs[2])
        usid.plot_utils.plot_map(
            ax3, 
            self.channels[channels[2]]/factors[2], 
            num_ticks=4, 
            vmin=zrange[2][0], 
            vmax=zrange[2][1], 
            cmap=cmap, 
            x_vec = self.x_vec,
            y_vec=self.x_vec,
            **kwargs,
        )
        ax3.set_xlabel('X ($\mathrm{\mu}$m)')

        axes = [ax1, ax2, ax3]
        return fig, axes


def convert_to_h5( directory ):
    trans = IgorIBWTranslator()
    c = 1
    for file in os.listdir( directory ):
        if file.endswith(".ibw"):
            tmp = trans.translate( os.path.join(directory, file))
            h5_file = h5py.File( tmp, mode='r' ) 
            print(os.path.join( directory, file ) + " - " + str(c))
            h5_file.close()
            c = c + 1
    print('Completed')
    return
    
def load_files( directory ):
    
    experiment = [ h5py.File( os.path.join(directory,file), mode='r' ) for file in os.listdir(directory) if file.endswith(".h5")]
    filenames = [ file for file in os.listdir(directory) if file.endswith(".h5")]
    
    return experiment, filenames

def file_preview(directory, **kwargs):
    """
    handy function to plot Asylum data with dropdown menus

    Parameters
    ----------
    directory   :   string or os.path
                Directory to preview files in
    
    Returns
    --------- 
    fig : Figure
            Figure of the plots that can be toggled using dropdown menus
    ax : 1D array_like of axes objects
            Axes of the individual plots within `fig`
    

    """
    filenames = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".h5")]

    def f(x, channel, **kwargs):
        #fig, ax = plt.subplots(1,3, figsize=(20,8))

        try:
            scan = AsylumDART(x)
            print("DART detected")
            if channel == "Channel 1":
                fig, ax = scan.multi_image_plot(
                    ["Topography", "Amplitude_1", "Phase_1"],
                    **kwargs
                )
            elif channel == "Channel 2":
                fig, ax = scan.multi_image_plot(
                    ["Topography", "Amplitude_2", "Phase_2"],
                    **kwargs
                )
        except:
            scan = AsylumSF(x)
            print("SF detected")
            fig, ax = scan.multi_image_plot(
                    ["Topography", "Amplitude", "Phase"],
                    **kwargs
            )
        
        
        return fig, ax

    widgets.interact(f, x=filenames, channel=["Channel 1", "Channel 2"], **kwargs)
