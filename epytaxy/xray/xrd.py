# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:00:40 2018

In this version I graph the output as H K L parameters

@author: oliver
"""

import os #this deals with file directories
import io
import xrayutilities as xu #this is the main package that is used
import numpy as np #useful mathematics package
import tkinter as tk #user interface for file IO
from tkinter import filedialog #file dialogue from TKinter
import matplotlib.pyplot as plt #plotting package
import matplotlib as mpl #plotting
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import rcParams, cm
import refnx.util.general as general
import refnx.util.ErrorProp as EP
import xml.etree.ElementTree as et
from refnx.dataset import ReflectDataset
import re
from itertools import islice
import zipfile

# mm
XRR_BEAMWIDTH_SD = 0.019449

rcParams.update({'figure.autolayout': True})
#%matplotlib notebook
#%matplotlib inline

#__all__ = ["lattice", "find_nearest"]

class ReciprocalSpaceMap:
    """
    X-ray diffraction reciprocal space map. 

    This class reads PANAlytical .xrdml files and Rigaku .ras files. When
    constructing an instance of this object, one needs to know the reciprocal
    lattice reflection that was measured, the in-plane and out-of-plane substrate
    directions to construct the scattering plane, and the substrate material to
    get expected RSM points to correct data for experimental omega and 2theta
    offsets. 

    Parameters
    ----------
    ref_hkl         :   3-tuple (h,k,l)
        Reciprocal lattice vector of the reciprocal lattice reflection that is measured.
    ip_hkl          :   3-tuple
        Reciprocal lattice vector of the in-plane substrate direction that is also along
        the x-ray propagation direction.
    oop_hkl         :   3-tuple
        Reciprocal lattice vector of the out-of-plane substrate normal direction. 
    substrate_mat   :   str, {"STO", "LAO", "LSAT", "YAO"}
        Abbreviation of substrate compound
    filepath        :   str, optional
        Filepath for the reciprocal space map file. If None, this loads an empty 
        ReciprocalSpaceMap that can be populated with the `self.load_sub`
        function.

    Attributes
    ----------
    omega   :   np.ndarray or list of np.ndarrays



    Notes
    ----------
    Some substrates are orthorhombic, and as such the orthorhombic
    reciprocal lattice vectors must be used. 
    """
    def __init__(self, ref_hkl, ip_hkl, oop_hkl, substrate_mat, filepath=None, **kwargs):
        # omega 2theta and data arrays. Omega and 2theta are 1D arrays while data is a 2D array with dimensions of omega and 2theta
        self.omega = []
        self.tt = []
        self.data = []
        self.meta = {}
        self.delta = []
        self.theoOm, self.theoTT = [], []
        self.qx, self.qy, self.qz = [], [], []
        self.omCorr, self.ttCorr = [], []
        self.gridder = []
        self.rsmRef = []
        self.iHKL = ip_hkl
        self.oHKL = oop_hkl
        self.h, self.k, self.l = 0, 0, 0
        self.maxInt = 0   
        self.p = []
        self.substrateMat = substrate_mat
        self.rsmRef = ref_hkl
        self.data_raw = []
        self.axis_raw = []
        self.omega1D = []
        self._num_datasets = 0

        
        [self.substrate, self.hxrd] = self.initialise_substrate( wl='CuKa12', **kwargs )

        if filepath:
            self.load_sub( filepath, **kwargs)

    def load_sub( self, filepath = None, delta=None, verbose=False, **kwargs):

        self.filepath = filepath
        if filepath is None:
            self.filepath = self.open_dialogue()   # open GUI for user to select file 
        b = self.filepath.index('.') # find the period that separates filename and extension
        a = self.filepath.rfind('/')  # find the slash that separates the directory and filename
        ext = self.filepath.split(".")[-1] # get extension and determine whether it is RAS or XRDML
        self.filename = self.filepath[(a+1):(b)]

        if ext == 'xrdml':
            (omega, tt, data) = self.xrdml_file(self.filepath)
            self.omega.append(omega)
            self.tt.append(tt)
            self.data.append(data)
            self.p, [qx, qy, qz] = self.align_sub_peak(delta=delta, verbose=verbose)
        elif ext == 'ras':
            (omega, tt, data) = self.ras_file(self.filepath)
            self.omega.append(omega)
            self.tt.append(tt)
            self.data.append(data)
            self.p, [qx, qy, qz] = self.align_sub_peak(delta=delta, verbose=verbose)
        elif ext == "txt":
            d = self.text_file(self.filepath)
            m = d["Metadata"]
            self.omega.append(d[m["axis type"][0]])
            self.tt.append(d[m["axis type"][1]])
            self.data.append(d["Data"])
            qx = np.zeros(d["Qy"].shape)
            qy = d["Qy"]
            qz = d["Qz"]
        else:
            print('filetype not supported.') 
        
        self._update_q_ranges(qx, qy, qz)
        self.qx.append(qx)
        self.qy.append(qy)
        self.qz.append(qz)
        self._num_datasets += 1

    def load_film( self, filepath=None, delta=None, verbose=False ):
        """
        Load film RSM file separately from substrate

        Applies same experimental offset as substrate file

        Parameters
        ----------
        filepath    :   str
                filepath to film RSM file
        """
        self.multi_file = True

        if delta is None:
            delta = self.delta

        if filepath is None:
            film_file = self.open_dialogue()
        else:
            film_file = filepath
        if filepath.endswith(".xrdml"):
            (f_omega, f_tt, f_data) = self.xrdml_file(film_file)
            qx, qy, qz = self.hxrd.Ang2Q(f_omega, f_tt, delta=delta)
        elif filepath.endswith(".ras"):
            (f_omega, f_tt, f_data) = self.ras_file(film_file)
            qx, qy, qz = self.hxrd.Ang2Q(f_omega, f_tt, delta=delta)
        elif filepath.endswith("txt"):
            d = self.text_file(filepath)
            m = d["Metadata"]
            f_omega = d[m["axis type"][0]]
            f_tt = d[m["axis type"][1]]
            f_data = d["Data"]
            qx = np.zeros(d["Qy"].shape)
            qy = d["Qy"]
            qz = d["Qz"]
        
        print(f_omega.shape)
        print(f_tt.shape)
        print(f_data.shape)

        self.omega.append(f_omega)
        self.tt.append(f_tt)
        self.data.append(f_data)

        self._update_q_ranges(qx, qy, qz)

        self.qx.append(qx)
        self.qy.append(qy)
        self.qz.append(qz)
        self._num_datasets += 1
        
        return f_omega, f_tt, f_data

    def xrdml_file(self, file):
        xrdml_file = xu.io.XRDMLFile(file)
        d = xrdml_file.scan.ddict

        twotheta = d["2Theta"]
        omega = np.array([np.full(np.shape(twotheta)[1], i) for i in d["Omega"]])
        data = d["detector"]
        d.pop("detector")
        d.pop("2Theta")
        d.pop("Omega")

        self.meta = d
        #data = xrdtools.read_xrdml(file)
        #om = data['Omega']
        #tt = data['2Theta']
        
        #for key, val in data.items():
        #    if key in ["filename", "sample", "status", "comment", "substrate", "measType", "hkl", "time", "xunit", "yunit", "date"]:
        #        self.meta[key] = val

        return (np.transpose(omega),
                np.transpose(twotheta),
                np.transpose(data))

    def ras_file(self, file):
        # Read RAS data to object
        rasFile = xu.io.rigaku_ras.RASFile(file)
        self.file = rasFile
        self.scanaxis = rasFile.scans[1].scan_axis
        self.step_size = rasFile.scans[1].meas_step
        self.measureSpeed= rasFile.scans[1].meas_speed
        self.data_count = rasFile.scans[1].length
        time_per_step = self.data_count / self.measureSpeed
        # Read raw motor position and intensity data to large 1D arrays

        omttRaw, data = xu.io.getras_scan(rasFile.filename+'%s', '', self.scanaxis)

        counts = np.array(data["int"] * data["att"] / time_per_step)

        intensities = counts.reshape(len(rasFile.scans), rasFile.scan.length)
        # Read omega data from motor positions at the start of each 2theta-Omega scan
        omga = [rasFile.scans[i].init_mopo['Omega'] for i in range(0, len(rasFile.scans))]
        self.axis_raw.append(omttRaw)
        self.data_raw.append(data)
        # Convert 2theta-omega data to 1D array
        tt_1d = np.array([omttRaw[i] for i in range(0, rasFile.scans[0].length)])

        if self.scanaxis == 'TwoThetaOmega' and self._num_datasets == 0: # If RSM is 2theta/omega vs omega scan, adjust omega values in 2D matrix
            
            omga = [[omga[i] + (n * self.step_size/2) for n in range(0,len(tt_1d))] for i in range(0,len(omga))]
            tt = [[tt_1d[i] for i in range(0,len(tt_1d))] for j in range(0, len(omga))]
        else:
            omga, tt = np.meshgrid(omga, tt_1d)
        tt = np.array(tt)
        om = np.array(omga)
        
        return (np.transpose(om), np.transpose(tt), np.transpose(intensities))

    def text_file(self, file):
        """
        Loads a ASCII 2D RSM file with Numpy.
        
        Parameters
        ----------
        txt_file : .txt file from rigaku 
        
        Output
        ----------
        (omega, tt, data) : 3-tuple of np.arrays
            Three 2D arrays for the omega, two-theta, 
            and intensity values of the reciprocal 
            space map measurement
        """
        
        f = open(file, "r")
        lines = f.readlines()
        
        
        data = []
        axis1 = []
        axis2 = []
        

        metadata = {}
        for line in lines[1:]:
            if "#" in line and ":" in line:
                # If # and : is in the line then it is a header comment 
                key, val = line.split(":")
                key = key.strip()
                if len(val.split(" ")) > 2:
                    # If there are multiple values in the comment
                    # line, put them in an array
                    val = val.split(" ")
                    val = val[1:]
                
                metadata[key[2:]] = val
            else:
                rel_om, tth, dat, _ =  line.split("\t")
                om = float(rel_om) + float(metadata["gonio origin"][0])
                axis1.append(float(om))
                axis2.append(float(tth))
                data.append(float(dat))

        om_length = int(metadata["number of data"][0])
        tth_length = int(metadata["number of data"][1])
        np_ax1 = np.array(axis1).reshape(tth_length, om_length)
        np_ax2 = np.array(axis2).reshape(tth_length, om_length)
        if "Chi" in metadata["axis type"][0]:
            # if True, then assume a 2D detector is used
            np_qy = 2 * (2 * np.pi / 1.54) * np.sin(np.radians(np_ax2/2)) * np.sin(np.radians(np_ax1))
            np_qz = 2 * (2 * np.pi / 1.54) * np.sin(np.radians(np_ax2/2)) * np.cos(np.radians(np_ax1))
        else:
            np_qy = 2*np.pi/1.54 * (np.cos(np.radians(np_ax2 - np_ax1)) - np.cos(np.radians(np_ax1)))
            np_qz = 2*np.pi/1.54 * (np.sin(np.radians(np_ax2 - np_ax1)) + np.sin(np.radians(np_ax1)))
        
        np_data = np.array(data).reshape(tth_length, om_length)
        d = {metadata["axis type"][0] : np_ax1, metadata["axis type"][1] : np_ax2, "Qy" : np_qy, "Qz" : np_qz, "Data" : np_data, "Metadata" : metadata}

        self.meta = d["Metadata"]
                        
        return d

    def plot2d(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        om = np.array(self.omega)
        tt = np.array(self.tt)
        if(om.ndim == 1 and tt.ndim == 1):
            om, tt = np.meshgrid(self.omega, self.tt)
        elif (om.ndim != tt.ndim):
            print('Error. omega and twotheta arrays must have same dimension. dim(omega) = ' \
                  + str(om.shape) + 'and dim(tt) = ' + str(tt.shape))
        '''    
        if self.scanaxis == "TwoThetaOmega":
            om1D = np.ravel(om)
            tt1D = np.ravel(tt)
            dat1D = np.ravel(self.data)
            ax.scatter(om1D,tt1D,c=dat1D,cmap='jet')
        '''    

        for i in range(self._num_datasets):
            a = ax.contourf(self.omega[i],self.tt[i], np.log10(self.data[i]), **kwargs)

        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(r'2$\theta$-$\omega$')
        fig.show()
        return ax
    
    def plotQ(self, xGrid, yGrid, dynLow, dynHigh, fig=None, ax=None, nlev = None, show_title=False, axlabels='hkl', **kwargs):
        """
        Plots X-ray diffraction mesh scan in reciprocal space. 
        First needs to nonlinearly regrid the angular data into reciprocal space
        and plot this.

        Parameters
        ----------
        xGrid   :   int
                number of x bins for regridding data
        yGrid   :   int
                number of y bins for regridding data
        dynLow  :   float
                soft lower limit for intensity colorscale. Equivalent to 10^-(dynLow)
        dynHigh :   float
                soft upper limit for intensity colorscale. Equivalent to 10^(dynHigh)
        """

        if ax is None or fig is None:
            fig, ax = plt.subplots()

        if show_title == True:
            ax.set_title( self.filename )

        if "cmap" not in kwargs.keys():
            # Generate new colormap for RSM display
            rainbow = cm.get_cmap('magma_r', 256)
            newcolors = rainbow(np.linspace(0, 1, 256))
            white = np.array([1, 1, 1, 1])
            newcolors[:20, :] = white
            kwargs["cmap"] = ListedColormap(newcolors)
        

        self.gridder = xu.Gridder2D(xGrid,yGrid)
        self.gridder.KeepData(True)
        self.gridder.dataRange(
            self._gridder_ymin, 
            self._gridder_ymax, 
            self._gridder_zmin,
            self._gridder_zmax,
        )
        # If multiple datasets have been loaded, add each one to gridder
        for d in range(self._num_datasets):
            x = self.qy[d].ravel()
            y = self.qz[d].ravel()
            z = self.data[d].ravel()
            self.gridder(x,y,z)
        
        intHigh = np.argmax(self.gridder.data)
        intMin = np.argmin(self.gridder.data)
        dynhigh = np.rint(np.log10(intHigh))



        INT = xu.maplog(self.gridder.data, dynLow, dynHigh)
        im = ax.contourf(self.gridder.xaxis, self.gridder.yaxis, np.transpose(INT), nlev, **kwargs)
        for c in im.collections:
            c.set_edgecolor("face")
        if axlabels == 'ipoop':
            xlabel = '$q_{ip}$'
            ylabel = '$q_{oop}$'
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            xlabel = [ '\\bar{'+str(-i)+'}' if i < 0 else str(i) for i in self.iHKL ]
            ylabel = [ '\\bar{'+str(-i)+'}' if i < 0 else str(i) for i in self.oHKL ]
            ax.set_xlabel(r'$q_{[' + xlabel[0] + '' + xlabel[1] + '' + xlabel[2] + ']}$')
            ax.set_ylabel(r'$q_{[' + ylabel[0] + '' + ylabel[1] + '' + ylabel[2] + ']}$')
            ax.tick_params(axis='both', which='major')
        
        return fig, ax
        
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def initialise_substrate(self, geometry='hi_lo', **kwargs):
        La = xu.materials.elements.La
        Al = xu.materials.elements.Al
        O =  xu.materials.elements.O2m
        Sr = xu.materials.elements.Sr
        Ti = xu.materials.elements.Ti
        Ta = xu.materials.elements.Ta
        Y =  xu.materials.elements.Y
        Dy = xu.materials.elements.Dy
        Sc = xu.materials.elements.Sc

        energy = 1240/0.154

        if self.substrateMat == "LAO":
            substrate = xu.materials.Crystal(
                "LaAlO3", 
                xu.materials.SGLattice(221, 3.784, atoms=[La, Al, O], pos=['1a', '1b', '3c'])
            )
        elif self.substrateMat == "STO":
            substrate = xu.materials.SrTiO3
        elif self.substrateMat == "LSAT": # need to make an alloy of LaAlO3 and Sr2AlTaO6
            mat1 = xu.materials.Crystal(
                "LaAlO3", xu.materials.SGLattice(221, 3.79, atoms=[La, Al, O], pos=['1a', '1b', '3c'])
            )
            mat2 = xu.materials.Crystal(
                "Sr2AlTaO6", 
                xu.materials.SGLattice(221, 3.898, atoms=[Sr, Al, Ta, O], pos=['8c', '4a', '4b', '24c'])
            )
            substrate = xu.materials.CubicAlloy(mat1, mat2, 0.71)

        elif self.substrateMat == "YAO":
            print(
                "Warning: YAlO3 is an orthorhombic substrate. Remember to take this into account\
                when inputting measured RSM reflections."
            )
            substrate = xu.materials.Crystal(
                "YAlO3", 
                xu.materials.SGLattice(62, 5.18, 5.33, 7.37, atoms=[Y, Al, O], pos=['4c', '4b', '4c', '8d'])
            )

        elif self.substrateMat == "DSO":
            substrate = xu.materials.Crystal(
                "DyScO3", 
                xu.materials.SGLattice(62, 5.44, 5.71, 7.89, atoms=[Dy, Sc, O], pos=['4c', '4b', '4c', '8d'])
            )

        hxrd = xu.HXRD(
            substrate.Q(int(self.iHKL[0]), int(self.iHKL[1]), int(self.iHKL[2])),
            substrate.Q(int(self.oHKL[0]), int(self.oHKL[1]), int(self.oHKL[2])), 
            en=energy, 
            geometry = geometry, 
            **kwargs
        )
        return [substrate, hxrd]
    
    def align_sub_peak(self, frange=None, delta=None, verbose=False, **kwargs):
        nchannel = 255
        chpdeg = nchannel/2.511 #2.511 degrees is angular acceptance of xrays to detector
        center_ch = 128

        indexMax = np.argmax(self.data)
        self.maxInt = indexMax

        data = np.array(self.data)
        tupleIndex = np.unravel_index( indexMax, (len(self.data[0][:,0]), len(self.data[0][0,:])) )
        
        
        [self.theoOm, dummy, dummy, self.theoTT] = self.hxrd.Q2Ang(self.substrate.Q(self.rsmRef))
        exp_tt = self.tt[0][tupleIndex[0],tupleIndex[1]]
        exp_om = self.omega[0][tupleIndex[0],tupleIndex[1]]

        plot = False
        if verbose:
            plot = True

        if frange:
            exp_om, exp_tt, p, cov = xu.analysis.fit_bragg_peak( self.omega[0], self.tt[0], self.data[0], self.theoOm, self.theoTT, self.hxrd, plot=plot, frange=frange)
        else:
            exp_om, exp_tt, p, cov = xu.analysis.fit_bragg_peak( self.omega[0], self.tt[0], self.data[0], exp_om, exp_tt, self.hxrd, plot=plot)

        self.hxrd.Ang2Q.init_linear('y+', center_ch, nchannel, chpdeg=chpdeg) 

        if delta is None:
            self.delta = (exp_om - self.theoOm, exp_tt - self.theoTT)
        else:
             self.delta = delta

        if verbose:
            print('experimental omega = ' + str(exp_om))
            print('experimental tt = ' + str(exp_tt))
            print('theoretical omega = ' + str(self.theoOm))
            print('theoretical 2theta = ' + str(self.theoTT))

            print('delta = ' + str( self.delta ) )

        [qx, qy, qz] = self.hxrd.Ang2Q(self.omega[0], self.tt[0], delta=self.delta )
        
        return p, [qx, qy, qz]

    def fit_zoom_peak( self, angPlot, *kwargs ):

        yminInd =  ( np.abs(self.tt[:,0] - angPlot.get_ylim()[0]) ).argmin()
        ymaxInd =  ( np.abs(self.tt[:,0] - angPlot.get_ylim()[1]) ).argmin()
        xminInd =  ( np.abs(self.omega[0, :] - angPlot.get_xlim()[0]) ).argmin()
        xmaxInd =  ( np.abs(self.omega[0, :] - angPlot.get_xlim()[1]) ).argmin() 

        fitRange = [self.data[0, xminInd], self.data[0, xmaxInd], self.data[yminInd, 0], self.data[ymaxInd, 0]]

        tupleIndex = np.unravel_index(np.argmax(self.data[yminInd:ymaxInd, xminInd:xmaxInd].flatten()), \
                     (len(self.data[yminInd:ymaxInd, 0]), len(self.data[0, xminInd:xmaxInd])))

        cropOm = self.omega[yminInd:ymaxInd, xminInd:xmaxInd]
        cropTT = self.tt[yminInd:ymaxInd, xminInd:xmaxInd]
        cropData = self.data[yminInd:ymaxInd, xminInd:xmaxInd]

        xC = cropOm[tupleIndex]
        yC = cropTT[tupleIndex]
        amp = self.data[tupleIndex]
        xSigma = 0.1
        ySigma = 0.1
        angle = 0
        background = 1
        self.p = [xC, yC, xSigma, ySigma, amp, background, angle]

        fitRange = [self.omega[0,xminInd], self.omega[0,xmaxInd], self.tt[yminInd, 0], self.tt[ymaxInd, 0] ]
        print(fitRange)

        fitParams, cov = xu.math.fit.fit_peak2d(self.omega[:,0], self.tt[:,0], self.data, self.p, fitRange, xu.math.Gauss2d)
        
        cl = angPlot.contour(self.omega[0,xminInd:xmaxInd], self.tt[yminInd:ymaxInd,0], \
                 np.log10(xu.math.Gauss2d(self.omega[yminInd:ymaxInd, xminInd:xmaxInd], \
                 self.tt[yminInd:ymaxInd,xminInd:xmaxInd], *fitParams)), 8, colors='k', linestyles='solid')

        return cl, fitParams, cov

    def fit_zoom_Qpeak( self, angPlot, verbose=False, *kwargs ):

        yminInd =  ( np.abs(self.gridder.yaxis[:] - angPlot.get_ylim()[0]) ).argmin()
        ymaxInd =  ( np.abs(self.gridder.yaxis[:] - angPlot.get_ylim()[1]) ).argmin()
        xminInd =  ( np.abs(self.gridder.xaxis[:] - angPlot.get_xlim()[0]) ).argmin()
        xmaxInd =  ( np.abs(self.gridder.xaxis[:] - angPlot.get_xlim()[1]) ).argmin() 
        crop_qy = self.gridder.xaxis[xminInd:xmaxInd]
        crop_qz = self.gridder.yaxis[yminInd:ymaxInd]
        crop_data = self.gridder.data[xminInd:xmaxInd,yminInd:ymaxInd]

        fitRange = [self.gridder.xaxis[xminInd], self.gridder.xaxis[xmaxInd], self.gridder.yaxis[yminInd], self.gridder.yaxis[ymaxInd]]
        print(fitRange)


        max_idx = np.unravel_index(
            np.argmax(crop_data.flatten()),
            shape=(len(crop_qy), len(crop_qz))
        )


        crop_qy_mesh, crop_qz_mesh = np.meshgrid(crop_qy, crop_qz)
        x_mesh, y_mesh = np.meshgrid( self.gridder.xaxis, self.gridder.yaxis)
        xC = crop_qy[max_idx[0]]
        yC = crop_qz[max_idx[1]]
        amp = self.gridder.data[max_idx]
        xSigma = 0.01
        ySigma = 0.01
        angle = 0
        background = 1
        self.p = [xC, yC, xSigma, ySigma, amp, background, angle]

        pfit, cov = xu.math.fit.fit_peak2d(x_mesh, y_mesh, self.gridder.data.T, self.p, fitRange, xu.math.Gauss2d)

        if verbose:
            print('------------- DEBUGGING -----------')
            print('QxGrid size = ' + str(crop_qy_mesh.shape))
            print('QzGrid size = ' + str(crop_qz_mesh.shape))
            print('cropData size = ' + str(crop_data.shape))
            print('fit params = ' + str(pfit))
        
        cl = angPlot.contour( 
            self.gridder.xaxis, 
            self.gridder.yaxis, 
            xu.math.Gauss2d( x_mesh, y_mesh, *pfit), 
            levels=8, 
            colors='k', 
            linestyles='solid'
        )

        return cl, pfit, cov

    def _update_q_ranges(self, qx, qy, qz):
        """
        Checks min and max q range of loaded data to see if it exceeds
        previously loaded data (if there is any) and updates global
        min and max

        Parameters
        ----------
        qx  :   
        """

        xmin = min(qx.flatten())
        xmax = max(qx.flatten())
        ymin = min(qy.flatten())
        ymax = max(qy.flatten())
        zmin = min(qz.flatten())
        zmax = max(qz.flatten())
        print(ymin)
        print(ymax)
        if self._num_datasets > 0:
            if xmin < self._gridder_xmin:
                self._gridder_xmin = xmin
            elif xmax > self._gridder_xmax:
                self._gridder_xmax = xmax
            if ymin < self._gridder_ymin:
                self._gridder_ymin = ymin
            elif ymax > self._gridder_ymax:
                self._gridder_ymax = ymax
            if zmin < self._gridder_zmin:
                self._gridder_zmin = zmin
            elif zmax > self._gridder_zmax:
                self._gridder_zmax = zmax
        else:
            self._gridder_xmin = xmin
            self._gridder_xmax = xmax
            self._gridder_ymin = ymin
            self._gridder_ymax = ymax
            self._gridder_zmin = zmin
            self._gridder_zmax = zmax

    def open_dialogue(self):
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename()
        filename = os.path.basename(filepath)
        return filepath

    def to_csv(self, fname=None):
        if fname is None:
            fname = self.filename + "_CSVoutput.csv"

        tmp = np.array(
            [self.omega.flatten(), 
            self.tt.flatten(), 
            self.data.flatten()]
            )
        np.savetxt(fname, tmp, delimiter=',')


class BRMLFile:
    """
    Bruker D8 .brml file reader for 1-dimensional 2theta-omega x-ray data. 


    Parameters
    ----------
    file_name : str
        string
    """

    def __init__(self, file_name):
        self.scan_axes = {}
        
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.printdir()
            c = zip_ref.read("Experiment0/RawData0.xml")

            self.scan_params, self.fixed_params = self.parse_scan(c)

            self.two_theta = self.scan_params.pop("two_theta")
            self.omega = self.scan_params.pop("omega")
            self.counts = self.scan_params.pop("counts")

    def parse_scan(self, xml_file):
        root = et.fromstring(xml_file)

        ns = {'brml': "http://www.w3.org/2001/XMLSchema",
                  'fixed' : "http://www.w3.org/2001/XMLSchema-instance"}

        query = {
            "start_time" : "TimeStampStarted",
            "end_time" : "TimeStampEnded",
            "intensities": "DataRoutes/DataRoute/Datum",
            "twotheta_start" : "DataRoutes/DataRoute/ScanInformation/ScanAxes/ScanAxisInfo[@AxisId='TwoTheta']/Start",
            "twotheta_end" : "DataRoutes/DataRoute/ScanInformation/ScanAxes/ScanAxisInfo[@VisibleName='2Theta']/Stop",
            "twotheta_incr": "DataRoutes/DataRoute/ScanInformation/ScanAxes/ScanAxisInfo[@VisibleName='2Theta']/Increment",
            "omega_start" : "DataRoutes/DataRoute/ScanInformation/ScanAxes/ScanAxisInfo[@VisibleName='Omega']/Start",
            "omega_end" : "DataRoutes/DataRoute/ScanInformation/ScanAxes/ScanAxisInfo[@VisibleName='Omega']/Stop",
            "omega_incr": "DataRoutes/DataRoute/ScanInformation/ScanAxes/ScanAxisInfo[@VisibleName='Omega']/Increment",
        }
            
        scanp = {key: [i.text for i in root.findall(value, ns)] for key, value in query.items()}

        two_theta = np.array([float(point.split(",")[2]) for point in scanp["intensities"]])
        omega = np.array([float(point.split(",")[3]) for point in scanp["intensities"]])
        counts = np.array([float(point.split(",")[-1]) for point in scanp["intensities"]])
        scanp.pop("intensities")
        scanp.update({
            "two_theta" : two_theta,
            "omega" : omega,
            "counts" : counts,
        })

        fixp = {}
        for item in root.findall("FixedInformation/Drives/"):
            motor_name = item.get("LogicName")
            position = item.find("Position")
            pos_val = position.get("Value")
            fixp.update({motor_name : pos_val})

        return scanp, fixp




def ras_file(file):
    """
    Reads a 1-dimensional Rigaku RAS file.

    Parameters
    ----------
    file    :   str
        Filepath to the .ras 1D file
    
    Returns
    ----------
    d   :   dict
        Dictionary with relevant information about the .ras file to plot
        the data.
    
    """

    # Read RAS data to object
    ras_file = xu.io.rigaku_ras.RASFile(file)

    d = {
        "scan_axis" : ras_file.scans[0].scan_axis,
        "step_size" : ras_file.scans[0].meas_step,
        "speed"     : ras_file.scans[0].meas_speed,
        "length"    : ras_file.scans[0].length,
    }
    
    # Read raw motor position and intensity data to large 1D arrays
    
    ax1, data = xu.io.getras_scan(ras_file.filename+'%s', '', d["scan_axis"])
    
    #d["int"] = np.array(data["int"])
    #d["att"] = np.array(data["att"])
    d["counts"] = np.array(data["int"]) * np.array(data["att"])
    d["2theta"] = np.array(data["TwoThetaOmega"])
    
    
    # Read omega data from motor positions at the start of each 2theta-Omega scan
    d["omega"] = np.array([ras_file.scans[i].init_mopo['Omega'] for i in range(len(ras_file.scans))])

    return d

def xrdml_file(file, data_dir="."):
    """
    Reads a PANAlytical XRDML file

    Parameters
    ---------
    file    :   str
        Filename of .xrdml file
    data_dir :  str
        Data directory of the .xrdml file

    Returns
    ---------
    d       :   dict
        A dictionary that holds
            - 2theta
            - omega
            - int
            - 
    """
    xrdml_file = xu.io.XRDMLFile(file, path=data_dir)
    d = xrdml_file.scan.ddict

    return d

