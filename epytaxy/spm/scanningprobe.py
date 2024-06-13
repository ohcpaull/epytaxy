import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit
import scipy
from ipywidgets import interact 
from tqdm import tqdm
import pySPM
from pySPM.SPM import SPM_image

import ipywidgets as widgets
import skimage.feature

from matplotlib.patches import ConnectionPatch

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
    plot_map,
)

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