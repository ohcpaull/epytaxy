import os
import h5py
import pyUSID as usid
import numpy as np
import matplotlib.pyplot as plt
import pycroscopy as px


def get_channels( file, fmt='raw' ):

    topo1 = usid.USIDataset(file['Measurement_000/Channel_000/Raw_Data']) 
    ampl1 = usid.USIDataset(file['Measurement_000/Channel_001/Raw_Data']) 
    phase1 = usid.USIDataset(file['Measurement_000/Channel_003/Raw_Data']) 
    ampl2 = usid.USIDataset(file['Measurement_000/Channel_002/Raw_Data']) 
    phase2 = usid.USIDataset(file['Measurement_000/Channel_004/Raw_Data']) 
    
    
    if fmt == 'arr':
        topo1_nd = np.transpose(topo1.get_n_dim_form().squeeze())
        ampl1_nd = np.transpose(ampl1.get_n_dim_form().squeeze())
        ampl2_nd = np.transpose(ampl2.get_n_dim_form().squeeze())
        phase1_nd = np.transpose(phase1.get_n_dim_form().squeeze())
        phase2_nd = np.transpose(phase2.get_n_dim_form().squeeze())
    
        if len(file['Measurement_000']) >10:
            topo2 = usid.USIDataset(file['Measurement_000/Channel_006/Raw_Data']) 
            topo2_nd = np.transpose(topo2.get_n_dim_form().squeeze())
        
            return [topo1_nd, ampl1_nd, phase1_nd, ampl2_nd, phase2_nd, topo2_nd]
        return [topo1_nd, ampl1_nd, phase1_nd, ampl2_nd, phase2_nd]
        
    
    if len(file['Measurement_000']) >10:
        topo2 = usid.USIDataset(file['Measurement_000/Channel_006/Raw_Data']) 
        topo2_nd = np.transpose(topo2.get_n_dim_form().squeeze())
        
        return [topo1, ampl1, phase1, ampl2, phase2, topo2]
    
    return [topo1, ampl1, phase1, ampl2, phase2]
    
def single_image_plot(image, title, xvec, cmap=None, zrange=None, axis=None, fig=None, posn=None, **kwargs):
    if axis is None:
        fig, axis = plt.subplots()   
        #gs = gridspec.GridSpec(1)

    #xvec = np.linspace( 0, usid.hdf_utils.get_attributes(experiment[i]['Measurement_000'])['ScanSize']/10**(-6), len(image[0]))
    
    if zrange is None:
        
        usid.plot_utils.plot_map(axis, image, cmap=cmap, x_vec = xvec, y_vec=xvec)
        #axis.set_title(title)    
        axis.set_xlabel('X ($\mathrm{\mu}$m)')
        axis.set_ylabel('Y ($\mathrm{\mu}$m)')
    else:
        
        usid.plot_utils.plot_map(axis, image, cmap=cmap, vmin=zrange[0], vmax=zrange[1], x_vec = xvec, y_vec=xvec, **kwargs)
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
    ph1 = -90
    ph2 = ph1 + 360
    z_range = [ (-2, 2), (0, 0.7),(ph1,ph2) ] # Preset height, amplitude, and phase channel max/min

    #[t1, a1, p1, a2, p2] = images
    #channel2 = [ t1, a2, p2 ]
    titles = ['topo', 'ampl', 'phase']
    if axis is None:

        fig, axis = plt.subplots(nrows=1, ncols=3, figsize=(12,4))   
        gs = gridspec.GridSpec(1, 3)

    xvec = np.linspace( 0, usid.hdf_utils.get_attributes(experiment['Measurement_000'])['ScanSize']/10**(-6), len(images[0]))
    axes = []
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


    return fig, axes

def convert_to_h5( directory ):
    trans = px.io.translators.igor_ibw.IgorIBWTranslator()
    c = 1
    for file in os.listdir( directory ):
        if file.endswith(".ibw"):
            tmp = trans.translate( directory + file)
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