import os
import h5py
import pyUSID as usid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pycroscopy as px
from scipy.optimize import curve_fit


class AsylumData(object):
    """
    Class that handles standard Asylum PFM files. 

    Parameters:

    
    """

    def __init__(self, fname):

        if fname.endswith(".ibw"):
            raise ValueError("Need to convert .ibw file to .h5 beforehand. use convert_to_h5(directory)")
        elif fname.endswith(".h5"):
            self.file = h5py.File( fname, mode='r' )
        else:
            raise ValueError("filetype needs to be .h5")
        
        self.filename = os.path.basename(fname)
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
                "Frequency" : self.frequency
            }
        )
        self.pos_max = usid.hdf_utils.get_attributes(self.file['Measurement_000'])['ScanSize']/10**(-6)
        self.x_vec = np.linspace(0, self.pos_max, len(self.topography))
        self.mask = None

    def get_map_range(self, channel):
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
        # Check if channel is a string specifying the PFM channel to plot
        if type(channel) is str:
            if channel in self.channels.keys():
                data = self.channels[channel]
                data_title = channel
            else:
                raise ValueError("Channel to plot needs to be one of {}".format(self.channels.keys()))

        # Check if it is the data array of the PFM channel to plot 
        elif isinstance(channel, np.ndarray):
#            if channel in self.channels.values():
            data = channel
            #data_title = list(self.channels.keys())[list(self.channels.values()).index(data)]
#            else:
#                raise ValueError("Channel data does not match loaded channels in class!")
        
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
            elif channel in ["Amplitude_1", "Amplitude_2"]:
                factors[idx] = 1e-12

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
    trans = px.io.translators.igor_ibw.IgorIBWTranslator()
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