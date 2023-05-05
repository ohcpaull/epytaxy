import os
import h5py
import pyUSID as usid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pycroscopy as px
from scipy.optimize import curve_fit
import scipy
from ipywidgets import interact 
from tqdm import tqdm
import pySPM
from pySPM.SPM import SPM_image

import ipywidgets as widgets
import skimage.feature

import matplotlib as mpl
from matplotlib.patches import ConnectionPatch

from epytaxy.spm.igor_ibw import IgorIBWTranslator

from epytaxy.spm.utils import(
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
    LineProfile,
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
            d = self._get_channels(fmt="arr")

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
                    "Frequency" : self.frequency,
                }
            )
            self.pos_max = usid.hdf_utils.get_attributes(self.file['Measurement_000'])['ScanSize']/10**(-6)
        elif fname.endswith(".spm"):
            self.file = pySPM.Bruker(os.path.join(data_dir, fname))
            t = self.file.get_channel("Height")
            d =  self.file.get_channel("Deflection Error")
            self.topography = t.pixels * 10**(-9)
            self.deflection = d.pixels

            #TODO: get metadata from SPM files
            self.pos_max = t.get_extent()[1]
            self.channels = dict(
                {
                    "Topography" : self.topography,
                    "Deflection_Error" : self.deflection,
                }
            )
        else:
            raise ValueError("filetype needs to be .h5")
        
        self.filename = os.path.basename(fname)
        self.channel_edges = {}        
        self.factors = dict(
            {
                "Topography" : 1e-9,
                "Amplitude_1" : 1e-12,
                "Amplitude_2" : 1e-12,
                "Phase_1" : 1,
                "Phase_2" : 1,
                "Frequency" : 1,
                "Deflection Error" : 1
            }
        )

        self.line_slice =  {
            "Topography" : None,
            "Amplitude_1" : None,
            "Amplitude_2" : None,
            "Phase_1" : None,
            "Phase_2" : None,
            "Frequency" : None,
            "FFT_Topography" : None,
            "FFT_Amplitude_1" : None,
            "FFT_Amplitude_2" : None,
            "FFT_Phase_1" : None,
            "FFT_Phase_2" : None,
            "Deflection Error" : None
        }
        self.px = {            
            "Topography" : [],
            "Amplitude_1" : [],
            "Amplitude_2" : [],
            "Phase_1" : [],
            "Phase_2" : [],
            "Frequency" : [],
            "FFT_Topography" : [],
            "FFT_Amplitude_1" : [],
            "FFT_Amplitude_2" : [],
            "FFT_Phase_1" : [],
            "FFT_Phase_2" : [],
            "Deflection Error" : [],
        }
        self.py = {
            "Topography" : [],
            "Amplitude_1" : [],
            "Amplitude_2" : [],
            "Phase_1" : [],
            "Phase_2" : [],
            "Frequency" : [],
            "FFT_Topography" : [],
            "FFT_Amplitude_1" : [],
            "FFT_Amplitude_2" : [],
            "FFT_Phase_1" : [],
            "FFT_Phase_2" : [],
            "Deflection Error" : [],
        }
        self.line_profile_x = {
            "Topography" : None,
            "Amplitude_1" : None,
            "Amplitude_2" : None,
            "Phase_1" : None,
            "Phase_2" : None,
            "Frequency" : None,
            "FFT_Topography" : None,
            "FFT_Amplitude_1" : None,
            "FFT_Amplitude_2" : None,
            "FFT_Phase_1" : None,
            "FFT_Phase_2" : None,
            "Deflection Error" : None
        }

        self.axis_length = len(self.topography)
        
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
        d = scipy.stats.describe(data.flatten())


        counts, bin_edges = np.histogram(data.flatten(), bins=200)
        bin_centres = np.array([0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(bin_edges)-1)])
        centre = bin_centres[np.argmax(counts)]
        amp = max(counts)
        
        # If channel is phase, return standard phase range
        if "Phase" in channel:
            func = gaussian
            #phase_hist = np.histogram(self.channels[channel])

            zmin = 0
            zmax = 360

        # If channel is amplitude, change fitting function to skewed gauss
        if "Amplitude" in channel:
            zmin = 0
            zmax = nsig * d.mean
        elif "Topography" in channel:
            p0 = [amp, centre, centre, 1e-8]
            popt, _ = curve_fit(func, bin_centres, counts, p0=p0)
            width = popt[2]
            zmin = -nsig * width
            zmax = nsig * width
        
        if plot:
            fig, ax = plt.subplots()
            _ = ax.hist(data.flatten(), bins=200)
            ax.plot(bin_centres, func(bin_centres, *popt))
        return (zmin, zmax)

    def apply_mask(self, channel, threshold):
        """
        Applies a mask to a channel based of a specific threshold
        """

    def get_2D_fft(self, channel):
        """
        Calculates the 2-dimensional FFT of a given channel and
        adds it to the list of channels in the object

        """
        fft =  np.fft.fftshift(np.fft.fft2(self.channels[channel]))

        self.channels[f"FFT_{channel}"] = np.abs(fft)
        
        return np.abs(fft)
        
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
        rotate=False,
        log=False,
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
            "Amplitude_2" : skewed_gauss,
            "Phase_2"     : None,
            "FFT_Topography" : gaussian,
            "FFT_Ampltide_1" : gaussian,
            "FFT_Phase_1" : gaussian,
            "FFT_Amplitude_2" : gaussian,
            "FFT_Phase_2" : gaussian,
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

        if rotate:
            data = np.rot90(data)

        if log:
            data = np.log10(data)

        if zrange is None:
            (zmin, zmax) = self.get_map_range(channel, channel_dict[channel])
            imhandle, cbar = plot_map(ax, data/factor, cmap=cmap, cbar_label=units, vmin=zmin/factor, vmax=zmax/factor, x_vec = self.x_vec, y_vec= self.x_vec, **kwargs)
            #axis.set_title(title)    
            ax.set_xlabel('X ($\\mathrm{\\mu}$m)')
            ax.set_ylabel('Y ($\\mathrm{\\mu}$m)')
        else:
            
            imhandle, cbar = plot_map(ax, data/factor, cmap=cmap, cbar_label=units, vmin=zrange[0], vmax=zrange[1], x_vec = self.x_vec, y_vec=self.x_vec, **kwargs)
            #axis.set_title(title)        
            ax.set_xlabel('X ($\\mathrm{\\mu}$m)')
            ax.set_ylabel('Y ($\\mathrm{\\mu}$m)')
        
        return fig, ax, imhandle

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
        
        if "color" in kwargs.keys():
            color = kwargs.pop("color")
        else:
            color = "black"

        factor = 1
        # Instantiate scale factor and units if needed
        if "FFT" in channel:
            zlabel = "Integrated Height"
            units = None
            factor = 1
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
        elif "FFT" in channel:
            zlabel= "Integrated signal"
            units = "a.u."
            factor=1

        fig, ax = plt.subplots(1,2)
        
        fig, ax[0], im = self.single_image_plot(channel, fig=fig, ax=ax[0], **kwargs)

        pos = []
        line = []
        self.xyA, self.xyB = (), ()
        # Convert width between nanometres and pixels
        width = int(np.round(width / 1000 / self.pos_max * self.axis_length))


        def onclick(event):
            if len(pos) == 0:
                # plot first scatter
                scatter = ax[0].scatter(event.xdata, event.ydata, s=5, color=color)
                pos.append(scatter)
                self.px[channel].append(event.xdata)
                self.py[channel].append(event.ydata)

            elif len(pos) == 1:
                # plot second scatter and line
                scatter = ax[0].scatter(event.xdata, event.ydata, s=5, color=color)
                pos.append(scatter)
                self.px[channel].append(event.xdata)
                self.py[channel].append(event.ydata)
                x_values = [self.px[channel][0], self.px[channel][1]]
                y_values = [self.py[channel][0], self.py[channel][1]]

                # Plot line profile of data
                lp = LineProfile(
                    (x_values[0], y_values[0]),
                    (x_values[1], y_values[1]),
                    width=width,
                    color=color
                )


                lp.cut_channel(self.channels[channel])
                line.append(lp.cpatch_line)
                line.append(lp.cpatch_i)
                line.append(lp.cpatch_f)


                diff_x = (lp.px_f[0] - lp.px_i[0]) / self.axis_length * self.pos_max
                diff_y = (lp.px_f[1] - lp.px_i[1]) / self.axis_length * self.pos_max
                sample_distance = np.hypot(diff_x, diff_y)

                lp.s_dist = np.linspace(0, sample_distance, len(lp.line_profile))

                # Plot line profile in adjacent subplot
                ax[1].plot(
                    lp.s_dist,
                    lp.line_profile / factor,
                    label=f"{channel} line profile",
                )
                ax[1].set(xlabel="Distance ($\\mathrm{\\mu}}$m)", ylabel=f"{zlabel} ({units})")
                
                # Plot line and width 
                lp._plot_over_channel(ax[0])
                self.line_slice[channel] = lp

                
            else:
            # clear variables 
                for scatter in pos:
                    scatter.remove()
                
                self.px[channel].clear()
                self.py[channel].clear()
                pos.clear()
                ax[1].clear()
                line[0].remove()
                line[1].remove()
                line[2].remove()
                line.clear()



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
            channels[0] : gaussian,
            channels[1] : skewed_gauss,
            channels[2]     : None,
        }
        ch_factors = {
            "Topography" : 1e-9,
            "Amplitude_1": 1e-12,
            "Phase_1"    : 1,
            "Amplitude_2": 1e-12,
            "Phase_2"    : 1,
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
        

        if zrange is None:
            zrange = {
                channels[0]       :   self.get_map_range(channels[0], channel_dict[channels[0]]),
                channels[1]       :   self.get_map_range(channels[1], channel_dict[channels[1]]),
                channels[2]       :   self.get_map_range(channels[2]),
            }
            #zrange = [ (zmin, zmax), (0, amax), (pmin,pmax) ] # Preset height, amplitude, and phase channel max/min

        #[t1, a1, p1, a2, p2] = images
        #channel2 = [ t1, a2, p2 ]
        if titles is None:
            titles = ['topo', 'ampl', 'phase']

        if axis is None:
            fig = plt.figure(figsize=(12,4))   
            gs = GridSpec(1, 3)

        
        axes = []
        for idx, ch in enumerate(channels):
            print(type(zrange[ch][0]))
            axes.append(fig.add_subplot(gs[idx]))
            usid.plot_utils.plot_map(
                axes[idx],
                self.channels[ch]/ch_factors[ch],
                num_ticks=3,
                vmin=zrange[ch][0]/ch_factors[ch],
                vmax=zrange[ch][1]/ch_factors[ch],
                cmap=cmap,
                x_vec=self.x_vec,
                y_vec=self.x_vec,
                **kwargs
            )
            axes[idx].set(xlabel="X ($\mathrm{\mu}$m)", ylabel="Y ($\mathrm{\mu}$m)")

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
        channel_dict = {
            "Topography" : gaussian,
            "Amplitude" : skewed_gauss,
            "Phase"     : None,
        }
        ch_factors = {
            "Topography" : 1e-9,
            "Amplitude": 1e-12,
            "Phase"    : 1,
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
        

        if zrange is None:
            zrange = {
                "Topography"    :   self.get_map_range("Topography", channel_dict["Topography"]),
                "Amplitude"   :   self.get_map_range("Amplitude", channel_dict["Amplitude"]),
                "Phase"       :   self.get_map_range("Phase"),
            }
            #zrange = [ (zmin, zmax), (0, amax), (pmin,pmax) ] # Preset height, amplitude, and phase channel max/min

        #[t1, a1, p1, a2, p2] = images
        #channel2 = [ t1, a2, p2 ]
        if titles is None:
            titles = ['topo', 'ampl', 'phase']

        if axis is None:
            fig = plt.figure(figsize=(12,4))   
            gs = GridSpec(1, 3)

        
        axes = []
        for idx, ch in enumerate(channels):

            axes.append(fig.add_subplot(gs[idx]))
            usid.plot_utils.plot_map(
                axes[idx],
                self.channels[ch]/ch_factors[ch],
                num_ticks=3,
                vmin=zrange[ch][0],
                vmax=zrange[ch][1],
                cmap=cmap,
                x_vec=self.x_vec,
                y_vec=self.y_vec,
                **kwargs
            )
            axes[idx].set(xlabel="X ($\mathrm{\mu}$m)", ylabel="Y ($\mathrm{\mu}$m)")

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


class Innova:
    """
    Class to read and plot Innova .FLT atomic force microscopy files
    
    Uses pySPM to put the data into a SPM_image type so `SPM_image.correct_lines`, 
    `SPM_image.corr_fit2d`, and `SPM_image.correct_slope` can be used as needed.
    
    """
    def __init__(self, filename, data_dir="."):
        self.filename = filename
        self.filepath = os.path.join(data_dir, filename)
        
        self.metadata = self.get_metadata()
        
        with open(self.filepath, 'rb') as f:
            d = np.fromfile(f, dtype='float32', offset=int(self.metadata["data_offset"]))
        
        # read the dataset
        self.data = d.reshape(int(self.metadata["resolutionX"]), int(self.metadata["resolutionY"])) * float(self.metadata["z_transfer_coeff"][:-5]) * 1000
        scan_range_x, units = self.metadata["scan_range_X"].split(" ")
        scan_range_y, units = self.metadata["scan_range_Y"].split(" ")

        self.image = SPM_image(
            self.data,
            channel=self.metadata["dataname"],
            real={
                "x" : float(scan_range_x),
                "y" : float(scan_range_y),
                "unit" : units
            },
            zscale="nm"
        )
         
    def get_metadata(self):
        """
        Get metadata from the file. This function assumes a very specific order of the metadata, 
        and will fail if the metadata is in a different order (which seems to be the case if you
        edit a file in Nanoscope and save it). Will eventually make this more robust to deal with 
        this.
        """
        metadata = {}
        with open(self.filepath, 'rb') as f:
    
            _ = f.readline()
            metadata["program"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["version"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            _ = f.readline()
            _ = f.readline()
            metadata["creationtime"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["dataname"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["dataID"] = f.readline().decode("ISO-8859-1").split('=')[1].rstrip()
            metadata["data_offset"] = f.readline().decode("ISO-8859-1").split('=')[1].rstrip()
            metadata["scan_range_X"] = f.readline().decode("ISO-8859-1").split('=')[1].rstrip()
            metadata["scan_range_Y"] = f.readline().decode("ISO-8859-1").split('=')[1].rstrip()
            metadata["offsetX"] = f.readline().decode("ISO-8859-1").split('=')[1].rstrip()
            metadata["offsetY"] = f.readline().decode("ISO-8859-1").split('=')[1].rstrip()
            metadata["rotation"] = f.readline().decode("ISO-8859-1").split('=')[1].rstrip()
            metadata["scan_rate"] = f.readline().decode("ISO-8859-1").split('=')[1].rstrip()
            metadata["resolutionX"] = f.readline().decode("ISO-8859-1").split('=')[1].rstrip()
            metadata["resolutionY"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["scan_direction"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["z_transfer_coeff"] = f.readline().decode("ISO-8859-1").split('=')[1].rstrip()
            metadata["leveling"] = f.readline().decode("utf-8").split('=')[1].rstrip()

            for i in range(5):
                _ = f.readline()

            metadata["X_lin_on"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["Y_lin_on"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["mode"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["setpoint"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["gain_P"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["gain_I"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["gain_D"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["X_lin_gain_P"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["X_lin_gain_I"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["X_lin_gain_D"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["Y_lin_gain_P"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["Y_lin_gain_I"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["Y_lin_gain_D"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["drive_frequency"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["drive_amplitude"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            metadata["drive_phase"] = f.readline().decode("ISO-8859-1").split('=')[1].rstrip()
            metadata["input_gain_selector"] = f.readline().decode("utf-8").split('=')[1].rstrip()
            
        return metadata
    
    def get_map_range(self, func=gaussian, plot=False, nsig=3):
        """
        Fits the histogram of a given data channel with a given function and returns
        an appropriate minimum and maximum range based on the width of the histogram peak.

        Parameters
        ----------
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
        data = self.data

        # Take histogram of data and determine height and centre of histogram. 
        d = scipy.stats.describe(data.flatten())


        counts, bin_edges = np.histogram(data.flatten(), bins=200)
        bin_centres = np.array([0.5 * (bin_edges[i] + bin_edges[i+1]) for i in range(len(bin_edges)-1)])
        centre = bin_centres[np.argmax(counts)]
        amp = max(counts)
        

        p0 = [amp, centre, centre, 1e-8]
        popt, _ = curve_fit(func, bin_centres, counts, p0=p0)
        width = np.abs(popt[2])
        zmin = -nsig * width
        zmax = nsig * width
        
        if plot:
            fig, ax = plt.subplots()
            _ = ax.hist(data.flatten(), bins=200)
            ax.plot(bin_centres, func(bin_centres, *popt))

        
        return (zmin, zmax)

    def plot(self, zrange=None, cmap="afmhot", fig=None, ax=None, **kwargs):
        """
        Plots a AFM image from Bruker Innova
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(3,3))
        
        if zrange is None:
            (zmin, zmax) = self.get_map_range()
        else:
            (zmin, zmax) = zrange

        imhandle, cbar = plot_map(
            ax, 
            self.data, 
            cmap=cmap, 
            cbar_label=self.image.zscale, 
            vmin=zmin, 
            vmax=zmax, 
            x_vec = self.image.size["real"]["x"], 
            y_vec= self.image.size["real"]["y"], 
            **kwargs
        )
        ax.set_xlabel('X ($\\mathrm{\\mu}$m)')
        ax.set_ylabel('Y ($\\mathrm{\\mu}$m)')

        return fig, ax, imhandle


def convert_to_h5(directory):
    trans = IgorIBWTranslator()
    for idx, file in tqdm(enumerate(os.listdir(directory))):
        if file.endswith(".ibw"):
            fname, ext = file.split(".")
            tmp = fname + ".h5"
            print(tmp)
            if tmp in os.listdir(directory):
                continue
            tmp = trans.translate( os.path.join(directory, file))
            h5_file = h5py.File( tmp, mode='r' ) 
            print(os.path.join( directory, file ) + " - " + str(idx+1))
            h5_file.close()
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
    bnames = [os.path.basename(f) for f in filenames]
    print(bnames)
    def f(x, channel, fdir, **kwargs):
        #fig, ax = plt.subplots(1,3, figsize=(20,8))

        try:
            scan = AsylumDART(os.path.join(fdir, x))
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

    widgets.interact(f, x=bnames, channel=["Channel 1", "Channel 2"], fdir=directory, **kwargs)

def edge_detection_widget(
    data, 
    channel, 
    zrange=None, 
    sigmainit = 0.2,
    lowthresinit = -0.1,
    highthresinit = 0.1,
    **kwargs
    ):
    """
    Interactive plotting of a PFM channel where the ranges
    and smoothing of the edge detection can be quickly optimised.

    Parameters
    ----------
    data       :   AsylumDART or AsylumSF object
        PFM file object to use in interactive plotting
    channel :   str
        Channel in AsylumDART or AsylumSF object to plot with
        edge detection.
    zrange  :   tuple, optional
        zmin and zmax for plotting the data channel. This also specifies the
        range of the lower and upper limits of the `skimage.feature.canny`
        edge detection
    """
    global d
    d = data
    global ch
    ch = channel

    if "cmap" not in kwargs.keys():
        kwargs.update({"cmap" : "afmhot"})
    
    if zrange is None:
        zrange = d.get_map_range(ch, nsig=5)
    (zmin, zmax) = zrange

    fig, ax = plt.subplots(figsize=(5,5))
    fig, ax, im = d.single_image_plot(ch, fig=fig, ax=ax, zrange=zrange, **kwargs)

    edges = skimage.feature.canny(
        image=d.channels[ch]/d.factors[ch],
        sigma= sigmainit,
        low_threshold=lowthresinit,
        high_threshold=highthresinit,
        )
    edges = np.invert(edges)
    masked_d = np.ma.masked_array(np.zeros((d.axis_length,d.axis_length)), edges)
    #d = np.ma.masked_where(edges == False,  np.zeros((512,512)))

    ax.imshow( masked_d, cmap="binary_r", interpolation = 'none')

    #ax_sigma = plt.axes([0.15, 0.05, 0.65, 0.03])
    sigma_slider = widgets.FloatSlider(description = 'sigma',
                    min = 0, max = 10, value = sigmainit)
    low_threshold_slider = widgets.FloatSlider(description = 'low threshold',
                    min = zmin, max = zmax, step=0.01, value = lowthresinit)
    high_threshold_slider = widgets.FloatSlider(description = 'high threshold',
                    min = zmin, max = zmax , step=0.01, value = highthresinit)


    def update(sigma, low_threshold, high_threshold):
        ax.clear()
        _ = d.single_image_plot(ch, fig=fig, ax=ax, cmap=cmap, show_cbar=False)
        edges = skimage.feature.canny(
            image=d.channels[ch]/d.factors[ch],
            sigma= sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        
        d = np.ma.masked_where(edges == False,  np.zeros((512,512)))
        ax.imshow(d, cmap="binary_r", interpolation = 'none')


    widgets.interact(update, sigma = sigma_slider, 
                        low_threshold = low_threshold_slider,
                        high_threshold = high_threshold_slider)

def Innova_edge_detection_widget(
    scan, 
    zrange=None, 
    sigmainit = 0.2,
    lowthresinit = -0.1,
    highthresinit = 0.1,
    **kwargs
    ):
    """
    Interactive plotting of a PFM channel where the ranges
    and smoothing of the edge detection can be quickly optimised.

    Parameters
    ----------
    data       :   AsylumDART or AsylumSF object
        PFM file object to use in interactive plotting
    channel :   str
        Channel in AsylumDART or AsylumSF object to plot with
        edge detection.
    zrange  :   tuple, optional
        zmin and zmax for plotting the data channel. This also specifies the
        range of the lower and upper limits of the `skimage.feature.canny`
        edge detection
    """
    global d
    d = scan


    if "cmap" not in kwargs.keys():
        kwargs.update({"cmap" : "afmhot"})
    
    if zrange is None:
        (zmin, zmax) = d.get_map_range(nsig=3)
    else:
        (zmin, zmax) = zrange
    
    zrange = (zmin, zmax)
    print(zrange)
    fig, ax = plt.subplots(figsize=(5,5))
    fig, ax, im = d.plot(fig=fig, ax=ax, zrange=zrange, **kwargs)

    edges = skimage.feature.canny(
        image=d.data,
        sigma= sigmainit,
        low_threshold=lowthresinit,
        high_threshold=highthresinit,
        )
    edges = np.invert(edges)
    masked_d = np.ma.masked_array(np.zeros((d.image.size["pixels"]["x"],d.image.size["pixels"]["x"])), edges)

    ax.imshow( masked_d, cmap="binary_r", interpolation = 'none')


    sigma_slider = widgets.FloatSlider(description = 'sigma',
                    min = 0, max = 10, value = sigmainit)
    low_threshold_slider = widgets.FloatSlider(description = 'low threshold',
                    min = zmin, max = zmax, step=0.01, value = lowthresinit)
    high_threshold_slider = widgets.FloatSlider(description = 'high threshold',
                    min = zmin, max = zmax , step=0.01, value = highthresinit)


    def update(sigma, low_threshold, high_threshold):
        ax.clear()
        _ = d.plot(fig=fig, ax=ax, show_cbar=False)
        edges = skimage.feature.canny(
            image=d.data,
            sigma= sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        
        ms = np.ma.masked_where(edges == False,  np.zeros((d.image.size["pixels"]["x"],d.image.size["pixels"]["x"])))
        ax.imshow(ms, cmap="binary_r", interpolation = 'none')


    widgets.interact(update, sigma = sigma_slider, 
                        low_threshold = low_threshold_slider,
                        high_threshold = high_threshold_slider)

def plot_map(axis, img, show_xy_ticks=True, show_cbar=True, x_vec=None, y_vec=None,
             num_ticks=4, stdevs=None, cbar_label=None, tick_font_size=None, infer_aspect=False, **kwargs):
    """
    Plots an image within the given axis with a color bar + label and appropriate X, Y tick labels.
    This is particularly useful to get readily interpretable plots for papers

    Parameters
    ----------
    axis : matplotlib.axes.Axes object
        Axis to plot this image onto
    img : 2D numpy array with real values
        Data for the image plot
    show_xy_ticks : bool, Optional, default = None, shown unedited
        Whether or not to show X, Y ticks
    show_cbar : bool, optional, default = True
        Whether or not to show the colorbar
    x_vec : 1-D array-like or Number, optional
        if an array-like is provided, these will be used for the tick values on the X axis
        if a Number is provided, this will serve as an extent for tick values in the X axis.
        For example x_vec=1.5 would cause the x tick labels to range from 0 to 1.5
    y_vec : 1-D array-like or Number, optional
        if an array-like is provided - these will be used for the tick values on the Y axis
        if a Number is provided, this will serve as an extent for tick values in the Y axis.
        For example y_vec=225 would cause the y tick labels to range from 0 to 225
    num_ticks : unsigned int, optional, default = 4
        Number of tick marks on the X and Y axes
    stdevs : unsigned int (Optional. Default = None)
        Number of standard deviations to consider for plotting.  If None, full range is plotted.
    cbar_label : str, optional, default = None
        Labels for the colorbar. Use this for something like quantity (units)
    tick_font_size : unsigned int, optional, default = None
        Font size to apply to x, y, colorbar ticks and colorbar label
    infer_aspect : bool, Optional. Default = False
        Whether or not to adjust the aspect ratio of the image based on the provided x_vec and y_vec
        The values of x_vec and y_vec will be assumed to have the same units.
    kwargs : dictionary
        Anything else that will be passed on to matplotlib.pyplot.imshow

    Returns
    -------
    im_handle : handle to image plot
        handle to image plot
    cbar : handle to color bar
        handle to color bar

    Note
    ----
    The origin of the image will be set to the lower left corner. Use the kwarg 'origin' to change this

    """
    if not isinstance(axis, mpl.axes.Axes):
        raise TypeError('axis must be a matplotlib.axes.Axes object')
    if not isinstance(img, np.ndarray):
        raise TypeError('img should be a numpy array')
    if not img.ndim == 2:
        raise ValueError('img should be a 2D array')
    if not isinstance(show_xy_ticks, bool):
        raise TypeError('show_xy_ticks should be a boolean value')
    if not isinstance(show_cbar, bool):
        raise TypeError('show_cbar should be a boolean value')
    # checks for x_vec and y_vec are done below
    if num_ticks is not None:
        if not isinstance(num_ticks, int):
            raise TypeError('num_ticks should be a whole number')
        if num_ticks < 2:
            raise ValueError('num_ticks should be at least 2')
    if stdevs is not None:
        data_mean = np.mean(img)
        data_std = np.std(img)
        kwargs.update({'clim': [data_mean - stdevs * data_std, data_mean + stdevs * data_std]})

    kwargs.update({'origin': kwargs.pop('origin', 'lower')})

    if show_cbar:
        vector = np.squeeze(img)
        if not isinstance(vector, np.ndarray):
            raise TypeError('vector should be of type numpy.ndarray. Provided object of type: {}'.format(type(vector)))
        if np.max(np.abs(vector)) == np.max(vector):
            exponent = np.log10(np.max(vector))
        else:
            # negative values
            exponent = np.log10(np.max(np.abs(vector)))
        
        y_exp = int(np.floor(exponent))
        z_suffix = ''
        if y_exp < -2 or y_exp > 3:
            img = np.squeeze(img) / 10 ** y_exp
            z_suffix = ' x $10^{' + str(y_exp) + '}$'

    assert isinstance(show_xy_ticks, bool)

    ########################################################################################################

    def set_ticks_for_axis(tick_vals, is_x):
        if is_x:
            tick_vals_var_name = 'x_vec'
            tick_set_func = axis.set_xticks
            tick_labs_set_func = axis.set_xticklabels
        else:
            tick_vals_var_name = 'y_vec'
            tick_set_func = axis.set_yticks
            tick_labs_set_func = axis.set_yticklabels

        img_axis = int(is_x)
        img_size = img.shape[img_axis]
        chosen_ticks = np.linspace(0, img_size - 1, num_ticks, dtype=int)

        if tick_vals is not None:
            if isinstance(tick_vals, (int, float)):
                if tick_vals > 0.01:
                    tick_labs = [str(np.round(ind * tick_vals / img_size, 2)) for ind in chosen_ticks]
                else:
                    tick_labs = ['{0:.1E}'.format(ind * tick_vals / img_size) for ind in chosen_ticks]
                    print(tick_labs)
                tick_vals = np.linspace(0, tick_vals, img_size)
            else:
                if not isinstance(tick_vals, (np.ndarray, list, tuple, range)) or len(tick_vals) != img_size:
                    raise ValueError(
                        '{} should be array-like with shape equal to axis {} of img'.format(tick_vals_var_name,
                                                                                            img_axis))
                if np.max(tick_vals) > 0.01:
                    tick_labs = [str(np.round(tick_vals[ind], 2)) for ind in chosen_ticks]
                else:
                    tick_labs = ['{0:.1E}'.format(tick_vals[ind]) for ind in chosen_ticks]
        else:
            tick_labs = [str(ind) for ind in chosen_ticks]

        tick_set_func(chosen_ticks)
        tick_labs_set_func(tick_labs)

        if tick_font_size is not None:
            for tick in axis.xaxis.get_major_ticks():
                tick.label.set_fontsize(tick_font_size)
            for tick in axis.yaxis.get_major_ticks():
                tick.label.set_fontsize(tick_font_size)    

        return tick_vals

    ########################################################################################################

    if show_xy_ticks is True or x_vec is not None:
        x_vec = set_ticks_for_axis(x_vec, True)
    else:
        axis.set_xticks([])

    if show_xy_ticks is True or y_vec is not None:
        y_vec = set_ticks_for_axis(y_vec, False)
    else:
        axis.set_yticks([])

    if infer_aspect:
        # Aspect ratio determined by this function will take precedence.
        _ = kwargs.pop('infer_aspect', None)

        """
        At this stage, if x_vec and y_vec are not None, they should be arrays.
        
        This will be very useful when one dimension is coarsely sampled while another is finely sampled
        and we want to visualize the image with the physically correct aspect ratio.
        This CANNOT be performed automatically due to potentially incompatible units which are unknown to this func.
        """

        if x_vec is not None or y_vec is not None:
            x_range = x_vec.max() - x_vec.min()
            y_range = y_vec.max() - y_vec.min()
            kwargs.update({'aspect': (y_range / x_range) * (img.shape[1] / img.shape[0])})

    im_handle = axis.imshow(img, **kwargs)

    cbar = None
    if not isinstance(show_cbar, bool):
        show_cbar = False

    if show_cbar:
        cbar = plt.colorbar(im_handle, ax=axis, orientation='vertical',
                            fraction=0.046, pad=0.04, use_gridspec=True)
        # cbar = axis.cbar_axes[count].colorbar(im_handle)

        if cbar_label is not None:
            if not isinstance(cbar_label, str):
                raise TypeError('cbar_label should be a string')

            if tick_font_size is not None:
                cbar.set_label(cbar_label + z_suffix)
            else:
                cbar.set_label(cbar_label + z_suffix, fontsize=tick_font_size)
        else:
            if z_suffix != '':
                cbar.set_label(z_suffix)

        if tick_font_size is not None:
            cbar.ax.tick_params(labelsize=tick_font_size)
    return im_handle, cbar        