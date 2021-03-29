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

    def __init__(self, fname, data_dir=None):

        if fname.endswith(".ibw"):
            raise ValueError("Need to convert .ibw file to .h5 beforehand. use convert_to_h5(directory)")
        elif fname.endswith(".h5"):
            self.file = h5py.File( os.path.join(data_dir,fname), mode='r' )
        else:
            raise ValueError("filetype needs to be .h5")
        
        self.filename = os.path.basename(os.path.join(data_dir, fname))
        d = self._get_channels(fmt="arr")
        self.topography = d[0]
        self.amplitude_1 = d[1]
        self.amplitude_2 = d[2]
        self.phase_1 = d[3]
        self.phase_2 = d[4]
        self.frequency = d[5] 
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

    def get_map_range(self, channel):
        # Check if channel is a string specifying the PFM channel to plot
        if type(channel) is str:
            if channel in self.channels.keys():
                data = self.channels[channel]
            else:
                raise ValueError("Channel to plot needs to be one of {}".format(self.channels.keys()))

        # Check if it is the data array of the PFM channel to plot 
        elif isinstance(channel, np.ndarray):
            if channel in channels.values():
                data = channel
            else:
                raise ValueError("Channel data does not match loaded channels in class!")

        data_hist = np.hist(data)


    def _get_channels(self, fmt="arr"):
        """
        Function that gets the different channels from an Asylum file and returns them 

        TODO: Need to make the assignment of channels more generic, and check for extra channels
        in a better way (in case the user has flattened the topography data or similar)
        """

        topo1 = usid.USIDataset(self.file['Measurement_000/Channel_000/Raw_Data']) 
        ampl1 = usid.USIDataset(self.file['Measurement_000/Channel_001/Raw_Data']) 
        ampl2 = usid.USIDataset(self.file['Measurement_000/Channel_002/Raw_Data']) 
        phase1 = usid.USIDataset(self.file['Measurement_000/Channel_003/Raw_Data']) 
        phase2 = usid.USIDataset(self.file['Measurement_000/Channel_004/Raw_Data']) 
        frequency = usid.USIDataset(self.file['Measurement_000/Channel_005/Raw_Data'])
        
        
        if fmt == 'arr':
            topo1_nd = np.transpose(topo1.get_n_dim_form().squeeze())
            ampl1_nd = np.transpose(ampl1.get_n_dim_form().squeeze())
            ampl2_nd = np.transpose(ampl2.get_n_dim_form().squeeze())
            phase1_nd = np.transpose(phase1.get_n_dim_form().squeeze())
            phase2_nd = np.transpose(phase2.get_n_dim_form().squeeze())
            freq_nd = np.transpose(frequency.get_n_dim_form().squeeze())
        
            if len(self.file['Measurement_000']) >10:
                topo2 = usid.USIDataset(self.file['Measurement_000/Channel_006/Raw_Data']) 
                topo2_nd = np.transpose(topo2.get_n_dim_form().squeeze())
            
                return [topo1_nd, ampl1_nd, phase1_nd, ampl2_nd, phase2_nd, topo2_nd]
            return [topo1_nd, ampl1_nd, phase1_nd, ampl2_nd, phase2_nd, freq_nd]
            
        
        if len(self.file['Measurement_000']) >10:
            topo2 = usid.USIDataset(self.file['Measurement_000/Channel_006/Raw_Data']) 
            topo2_nd = np.transpose(topo2.get_n_dim_form().squeeze())
            
            return [topo1, ampl1, phase1, ampl2, phase2, frequency, topo2]
        
        return [topo1, ampl1, phase1, ampl2, phase2, frequency]

    def single_image_plot(self, channel="Topography", cmap=None, zrange=None, axis=None, fig=None, **kwargs):
        # Check if channel is a string specifying the PFM channel to plot
        if type(channel) is str:
            if channel in self.channels.keys():
                data = self.channels[channel]
                data_title = channel
            else:
                raise ValueError("Channel to plot needs to be one of {}".format(self.channels.keys()))

        # Check if it is the data array of the PFM channel to plot 
        elif isinstance(channel, np.ndarray):
            if channel in channels.values():
                data = channel
                data_title = self.channels.index(data)
            else:
                raise ValueError("Channel data does not match loaded channels in class!")


        if axis is None:
            fig, axis = plt.subplots()   

        if zrange is None:
            
            usid.plot_utils.plot_map(axis, data, cmap=cmap, x_vec = self.x_vec, y_vec= self.x_vec)
            #axis.set_title(title)    
            axis.set_xlabel('X ($\mathrm{\mu}$m)')
            axis.set_ylabel('Y ($\mathrm{\mu}$m)')
        else:
            
            usid.plot_utils.plot_map(axis, data, cmap=cmap, vmin=zrange[0], vmax=zrange[1], x_vec = self.x_vec, y_vec=self.x_vec, **kwargs)
            #axis.set_title(title)        
            axis.set_xlabel('X ($\mathrm{\mu}$m)')
            axis.set_ylabel('Y ($\mathrm{\mu}$m)')

        return fig, axis

    
def multi_image_plot(images, experiment, titles=None, cmap=None, 
                     zrange=None, axis=None, fig=None,  gs=None, 
                     posn=None, ntick=4, **kwargs):
    """
    Handy function that plots three image channels side by side with colorbars

    Parameters
    ----------
    images     : list or array-like
                List of three images defined as 2D numpy arrays
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
    if zrange is None:
        ph1 = -90
        ph2 = ph1 + 360
        zrange = [ (-2, 2), (0, 200),(ph1,ph2) ] # Preset height, amplitude, and phase channel max/min

    #[t1, a1, p1, a2, p2] = images
    #channel2 = [ t1, a2, p2 ]
    titles = ['topo', 'ampl', 'phase']
    topo, ampl, phase = images
    if axis is None:

        fig, axis = plt.subplots(nrows=1, ncols=3, figsize=(12,4))   
        gs = GridSpec(1, 3)

    xvec = np.linspace( 0, usid.hdf_utils.get_attributes(experiment['Measurement_000'])['ScanSize']/10**(-6), len(topo))
    axes = []



    ax1 = fig.add_subplot(gs[0])
    usid.plot_utils.plot_map(
        ax1, images[0]/10**(-9), stdevs=3, num_ticks=4, 
        vmin=zrange[0][0], vmax=zrange[0][1], cmap=cmap, 
        x_vec = xvec, y_vec = xvec
        )
    ax1.set_xlabel('X ($\mathrm{\mu}$m)')
    ax1.set_ylabel('Y ($\mathrm{\mu}$m)')

    ax2 = fig.add_subplot(gs[1])
    usid.plot_utils.plot_map(
        ax2, images[1]/10**(-12), stdevs=3, num_ticks=4, 
        vmin=zrange[1][0], vmax=zrange[1][1], cmap=cmap, 
        x_vec = xvec, y_vec=xvec
        )
    ax2.set_xlabel('X ($\mathrm{\mu}$m)')

    ax3 = fig.add_subplot(gs[2])
    usid.plot_utils.plot_map(
        ax3, images[2], stdevs=3, num_ticks=4, 
        vmin=zrange[2][0], vmax=zrange[2][1], cmap=cmap, 
        x_vec = xvec, y_vec=xvec
        )
    ax3.set_xlabel('X ($\mathrm{\mu}$m)')

    axes = [ax1, ax2, ax3]
    return fig, axes

'''
    if zrange is None:
        
        for pos, img, title in zip(gs, images, titles):
            axis = fig.add_subplot(pos)
            usid.plot_utils.plot_map(axis, img, stdevs=3, num_ticks=4,
                                     cmap=cmap, x_vec = xvec, y_vec=xvec)
            axis.set_title(title, fontsize=12) 
            axis.set_xlabel('X ($\mathrm{\mu}$m)')
            if pos == gs[0]:
                axis.set_ylabel('Y ($\mathrm{\mu}$m)')
            axes.append(axis)
    else:
        for pos, img, title, zrange in zip(gs, images, titles, zrange):
            axis = fig.add_subplot(pos)
            usid.plot_utils.plot_map(axis, img, cmap=cmap, num_ticks=ntick,
                                     vmin=zrange[0], vmax=zrange[1],
                                     x_vec = xvec, y_vec=xvec)
            axis.set_title(title)        
            axis.set_xlabel('X ($\mathrm{\mu}$m)')
            if pos == gs[0]:
                axis.set_ylabel('Y ($\mathrm{\mu}$m)')
            axes.append(axis)
'''


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