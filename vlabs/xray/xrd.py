# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:00:40 2018

In this version I graph the output as H K L parameters

@author: oliver
"""

import os #this deals with file directories
import xrayutilities as xu #this is the main package that is used
import numpy as np #useful mathematics package
import tkinter as tk #user interface for file IO
from tkinter import filedialog #file dialogue from TKinter
import matplotlib.pyplot as plt #plotting package
import matplotlib as mpl #plotting
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import rcParams, cm
import xrdtools
import refnx.util.general as general
import refnx.util.ErrorProp as EP
import xml.etree.ElementTree as et
from refnx.dataset import ReflectDataset
import re
from itertools import islice

# mm
XRR_BEAMWIDTH_SD = 0.019449

rcParams.update({'figure.autolayout': True})
#%matplotlib notebook
#%matplotlib inline

#__all__ = ["lattice", "find_nearest"]

class ReciprocalSpaceMap(object):
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
        self.axisraw = []
        self.omega1D = []
        self._num_datasets = 0

        
        [self.substrate, self.hxrd] = self.initialise_substrate( wl='CuKa12', **kwargs )

        if filepath:
            self.load_sub( filepath, **kwargs)

    def load_sub( self, filepath = None, delta=None, verbose=False):

        self.filepath = filepath
        if filepath is None:
            self.filepath = self.open_dialogue()   # open GUI for user to select file 
        b = self.filepath.index('.') # find the period that separates filename and extension
        a = self.filepath.rfind('/')  # find the slash that separates the directory and filename
        ext = self.filepath[(b+1):] # get extension and determine whether it is RAS or XRDML
        self.filename = self.filepath[(a+1):(b)]

        if ext == 'xrdml':
            (omega, tt, data) = self.xrdml_file(self, self.filepath)
            self.omega.append(omega)
            self.tt.append(tt)
            self.data.append(data)
        elif ext == 'ras':
            (omega, tt, data) = self.ras_file(self, self.filepath)
            self.omega.append(omega)
            self.tt.append(tt)
            self.data.append(data)
        else:
            print('filetype not supported.') 
        self.p, [qx, qy, qz] = self.align_sub_peak(delta=delta, verbose=verbose)
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

        if filepath is None:
            film_file = self.open_dialogue()
        else:
            film_file = filepath
        if filepath.endswith(".xrdml"):
            (f_omega, f_tt, f_data) = self.xrdml_file( self, film_file )
        elif filepath.endswith(".ras"):
            (f_omega, f_tt, f_data) = self.ras_file( self, film_file )
            
        self.omega.append(f_omega)
        self.tt.append(f_tt)
        self.data.append(f_data)

        if not delta:
            delta = self.delta
        qx, qy, qz = self.hxrd.Ang2Q(f_omega, f_tt, delta=[delta[0], delta[1]])
        self._update_q_ranges(qx, qy, qz)

        self.qx.append(qx)
        self.qy.append(qy)
        self.qz.append(qz)
        self._num_datasets += 1
        
        return f_omega, f_tt, f_data

    @staticmethod
    def xrdml_file(self, file):
        xrdml_file = xu.io.XRDMLFile(file, path=self.filepath)
        d = xrdml_file.scan.ddict

        twotheta = d["2Theta"]
        omega = np.array([np.full(np.shape(twotheta)[1], i) for i in d["Omega"]])
        data = d["detector"]
        d.pop("detector")
        d.pop("2theta")
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

    @staticmethod
    def ras_file(self, file):
        # Read RAS data to object
        rasFile = xu.io.rigaku_ras.RASFile(file)
        
        self.scanaxis = rasFile.scans[1].scan_axis
        self.stepSize = rasFile.scans[1].meas_step
        self.measureSpeed= rasFile.scans[1].meas_speed
        self.dataCount = rasFile.scans[1].length
        # Read raw motor position and intensity data to large 1D arrays

        omttRaw, data = xu.io.getras_scan(rasFile.filename+'%s', '', self.scanaxis)

        npinte = np.array(data['int'])
        intensities = npinte.reshape(len(rasFile.scans), rasFile.scans[0].length)
        # Read omega data from motor positions at the start of each 2theta-Omega scan
        om = [rasFile.scans[i].init_mopo['Omega'] for i in range(0, len(rasFile.scans))]
        self.axisraw = omttRaw
        # Convert 2theta-omega data to 1D array
        tt = [omttRaw.data[i] for i in range(0, rasFile.scans[0].length)]
        
        if self.scanaxis == 'TwoThetaOmega': # If RSM is 2theta/omega vs omega scan, adjust omega values in 2D matrix
            omga = [[om[i] + (n * self.stepSize/2) for n in range(0,len(tt))] for i in range(0,len(om))]
            omga = np.array(omga)
            self.omega1D = omga
            ttheta = np.array(tt)
            tt = [[ttheta[i] for i in range(0,len(ttheta))] for j in range(0, len(omga))]
            tt = np.array(tt)
        
        return (np.transpose(omga), np.transpose(tt), np.transpose(intensities))
        
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

        
        a = ax.contourf(self.omega[0],self.tt[0], np.log10(self.data[0]), **kwargs)

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
        levels = np.linspace(1, dynhigh, num=20)
        levels = 10**(levels)
        #print(levels)
        ax.contourf(self.gridder.xaxis, self.gridder.yaxis, np.transpose(INT), nlev, **kwargs)

        if axlabels == 'ipoop':
            xlabel = '$Q_{ip}$'
            ylabel = '$Q_{oop}$'
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            xlabel = [ '\\bar{'+str(-i)+'}' if i < 0 else str(i) for i in self.iHKL ]
            ylabel = [ '\\bar{'+str(-i)+'}' if i < 0 else str(i) for i in self.oHKL ]
            ax.set_xlabel(r'$Q_{[' + xlabel[0] + '' + xlabel[1] + '' + xlabel[2] + ']}$')
            ax.set_ylabel(r'$Q_{[' + ylabel[0] + '' + ylabel[1] + '' + ylabel[2] + ']}$')
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
        energy = 1240/0.154
        

        while (self.substrateMat != 'LAO' and self.substrateMat != 'STO' and self.substrateMat != 'LSAT' and self.substrateMat != 'YAO'):
            self.substrateMat = input('Sample substrate (LAO, STO or LSAT)?')
            print(self.substrateMat)
            print('input valid substrate material')
            


        if self.substrateMat == "LAO":
            substrate = xu.materials.Crystal("LaAlO3", xu.materials.SGLattice(221, 3.784, \
                            atoms=[La, Al, O], pos=['1a', '1b', '3c']))
            hxrd = xu.HXRD(substrate.Q(int(self.iHKL[0]), int(self.iHKL[1]), int(self.iHKL[2])), \
                           substrate.Q(int(self.oHKL[0]), int(self.oHKL[1]), int(self.oHKL[2])), en=energy, geometry = geometry)
        elif self.substrateMat == "STO":
            substrate = xu.materials.SrTiO3

            hxrd = xu.HXRD(substrate.Q(int(self.iHKL[0]), int(self.iHKL[1]), int(self.iHKL[2])), \
                           substrate.Q(int(self.oHKL[0]), int(self.oHKL[1]), int(self.oHKL[2])), en=energy)
        elif self.substrateMat == "LSAT": # need to make an alloy of LaAlO3 and Sr2AlTaO6
            mat1 = xu.materials.Crystal("LaAlO3", xu.materials.SGLattice(221, 3.79, \
                           atoms=[La, Al, O], pos=['1a', '1b', '3c']))
            mat2 = xu.materials.Crystal("Sr2AlTaO6", xu.materials.SGLattice(221, 3.898,\
                           atoms=[Sr, Al, Ta, O], pos=['8c', '4a', '4b', '24c']))
            substrate = xu.materials.CubicAlloy(mat1, mat2, 0.71)
            hxrd = xu.HXRD(substrate.Q(int(self.iHKL[0]), int(self.iHKL[1]), int(self.iHKL[2])), \
                           substrate.Q(int(self.oHKL[0]), int(self.oHKL[1]), int(self.oHKL[2])), en=energy)
        elif self.substrateMat == "YAO":
            print(
                "Warning: YAlO3 is an orthorhombic substrate. Remember to take this into account\
                when inputting measured RSM reflections."
            )
            substrate = xu.materials.Crystal("YAlO3", xu.materials.SGLattice(62, 5.18, 5.33, 7.37,\
                            atoms=[Y, Al, O], pos=['4c', '4b', '4c', '8d']))
            hxrd = xu.HXRD(substrate.Q(int(self.iHKL[0]), int(self.iHKL[1]), int(self.iHKL[2])), \
                           substrate.Q(int(self.oHKL[0]), int(self.oHKL[1]), int(self.oHKL[2])), en=energy, **kwargs)
           
        return [substrate, hxrd]
    
    def align_sub_peak(self, delta=None, verbose=False):
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

    def fit_zoom_Qpeak( self, angPlot, *kwargs ):

        yminInd =  ( np.abs(self.gridder.yaxis[:] - angPlot.get_ylim()[0]) ).argmin()
        ymaxInd =  ( np.abs(self.gridder.yaxis[:] - angPlot.get_ylim()[1]) ).argmin()
        xminInd =  ( np.abs(self.gridder.xaxis[:] - angPlot.get_xlim()[0]) ).argmin()
        xmaxInd =  ( np.abs(self.gridder.xaxis[:] - angPlot.get_xlim()[1]) ).argmin() 
        cropQx = self.gridder.xaxis[xminInd:xmaxInd]
        cropQz = self.gridder.yaxis[yminInd:ymaxInd]

        fitRange = [self.gridder.xaxis[xminInd], self.gridder.xaxis[xmaxInd], self.gridder.yaxis[yminInd], self.gridder.yaxis[ymaxInd]]
        print(fitRange)


        tupleIndex = np.unravel_index(
            np.argmax(self.gridder.data[xminInd:xmaxInd,yminInd:ymaxInd].flatten()),
            shape=(len(self.gridder.data[xminInd:xmaxInd, 0]), len(self.gridder.data[0,yminInd:ymaxInd]))
        )


        cropData = self.gridder.data[yminInd:ymaxInd, xminInd:xmaxInd]
        cropQxGrid, cropQzGrid = np.meshgrid(cropQx, cropQz)
        xGrid, yGrid = np.meshgrid( self.gridder.xaxis, self.gridder.yaxis)
        xC = cropQx[tupleIndex[0]]
        yC = cropQz[tupleIndex[1]]
        amp = self.gridder.data[tupleIndex]
        xSigma = 0.01
        ySigma = 0.01
        angle = 0
        background = 1
        self.p = [xC, yC, xSigma, ySigma, amp, background, angle]

        fitParams, cov = xu.math.fit.fit_peak2d(xGrid, yGrid, self.gridder.data.T, self.p, fitRange, xu.math.Gauss2d)


        print('------------- DEBUGGING -----------')
        print('QxGrid size = ' + str(cropQxGrid.shape))
        print('QzGrid size = ' + str(cropQzGrid.shape))
        print('cropData size = ' + str(cropData.shape))
        print('fit params = ' + str(fitParams))
        cl = angPlot.contour( self.gridder.xaxis, self.gridder.yaxis, \
                 np.log10(xu.math.Gauss2d( xGrid, \
                 yGrid, *fitParams)), 8, colors='k', linestyles='solid')

        return cl, fitParams, cov

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

        if self._num_datasets:
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

def xrdml_file(file):
        """
        Reads a PANAlytical XRDML file

        Parameters
        ---------
        file    :   str
            Filepath to .xrdml file

        Returns
        ---------
        d       :   dict
            A dictionary that holds
                - 2theta
                - omega
                - int
                - 
        """
        d = {}
        d = xrdtools.read_xrdml(file)

        return d

def reduce_xrr(f, bkg=None, scale=None, sample_length=None, throwaway=0):
    """
    Reduces a X-ray file. Current supported file formats are
    PANAlytical XRDML and Rigaku RAS filetypes.

    Parameters
    ----------
    f: file-like object or string
        The specular reflectivity (XRDML or RAS) file of interest
    bkg: list
        A list of file-like objects or strings that contain background
        measurements. The background is assumed to have the same number of
        points as the specular reflectivity curve.  The backgrounds are
        averaged and subtracted from the specular reflectivity
    scale: float, None
        The direct beam intensity (cps). If `scale is None` then the dataset
        is scaled by the point with maximum intensity below Q = 0.0318 (Q_crit
        for Si at 8.048 keV).
    sample_length: None or float
        If None then no footprint correction is done. Otherwise the transverse
        footprint of the sample (mm).

    Returns
    -------
    dataset: refnx.dataset.ReflectDataset
        The specular reflectivity as a function of momentum transfer, Q.
    """
    if f.endswith(".xrdml"):
        spec = _parse_xrdml_file(f)
    elif f.endswith(".ras"):
        spec = _parse_ras_file(f)

    reflectivity = (spec["intensities"][throwaway:] + 1) / spec["count_time"]
    reflectivity_s = (
        np.sqrt(spec["intensities"][throwaway:]) / spec["count_time"]
    )

    # do the background subtraction
    if bkg is not None:
        bkgds = [parse_xrdml_file(fi) for fi in bkg]

        bkgd_refs = np.r_[[bkgd["intensities"] for bkgd in bkgds]]
        bkgd_refs_s = np.r_[
            [
                np.sqrt(bkgd["intensities"]) / bkgd["count_time"]
                for bkgd in bkgds
            ]
        ]
        bkgd_refs_var = bkgd_refs_s ** 2
        weights = 1.0 / bkgd_refs_var
        numerator = np.sum(bkgd_refs * weights, axis=0)
        denominator = np.sum(weights, axis=0)

        total_bkgd = numerator / denominator
        total_bkgd_s = np.sqrt(1 / denominator)

        reflectivity, reflectivity_s = EP.EPsub(
            reflectivity[throwaway:],
            reflectivity_s[throwaway:],
            total_bkgd,
            total_bkgd_s,
        )

    # work out the Q values
    qx, qy, qz = general.q2(
        spec["omega"][throwaway:],
        spec["twotheta"][throwaway:],
        np.zeros_like(spec["omega"][throwaway:]),
        spec["wavelength"],
    )

    # do a footprint correction
    if sample_length is not None:
        footprint_correction = general.beamfrac(
            np.array([XRR_BEAMWIDTH_SD]) * 2.35,
            np.array([sample_length]),
            spec["omega"][throwaway:],
        )
        reflectivity /= footprint_correction
        reflectivity_s /= footprint_correction

    # divide by the direct beam intensity
    # assumes that the direct beam intensity is enormous, so the counting
    # uncertainties in the scale factor are negligible.
    if scale is None:
        # no scale factor was specifed, so normalise by highest intensity point
        # below Qc for Silicon at 8.048 keV
        below_qc = qz[qz < 0.0318]
        if len(below_qc):
            scale = np.max(reflectivity[qz < 0.0318])

    reflectivity /= scale
    reflectivity_s /= scale

    d = ReflectDataset(data=(qz, reflectivity, reflectivity_s))

    return d

def _parse_ras_file(f):
    """
    Parses a RAS file. Adapted from xrayutilities
    (see https://xrayutilities.sourceforge.io/)

    Parameters
    ----------
    f: file-like object or string

    Returns
    -------
    d: dict
        A dictionary containing the RAS file information.  The following keys
        are used:

        'intensities' - np.ndarray
            Intensities
        'twotheta' - np.ndarray
            Two theta values
        'omega' - np.ndarray
            Omega values
        'count_time' - float
            How long each point was counted for
        'wavelength' - float
            Wavelength of X-ray radiation
    """
    re_measstart = re.compile(r"^\*RAS_DATA_START")
    re_measend = re.compile(r"^\*RAS_DATA_END")
    re_headerstart = re.compile(r"^\*RAS_HEADER_START")
    re_headerend = re.compile(r"^\*RAS_HEADER_END")
    re_datastart = re.compile(r"^\*RAS_INT_START")
    re_scanaxis = re.compile(r"^\*MEAS_SCAN_AXIS_X_INTERNAL")
    re_initmopovalue = re.compile(r"^\*MEAS_COND_AXIS_POSITION")
    re_datacount = re.compile(r"^\*MEAS_DATA_COUNT")
    re_measspeed = re.compile(r"^\*MEAS_SCAN_SPEED ")
    re_measstep = re.compile(r"^\*MEAS_SCAN_STEP ")
    re_wavelength = re.compile(r"^\*HW_XG_WAVE_LENGTH_ALPHA1")

    keys, position = {}, {}
    with open(f, mode="rb") as fid:
        while True:
            line = fid.readline()
            line = line.decode("ascii", "ignore")
            d = dict()
            if re_measstart.match(line):
                continue
            elif re_headerstart.match(line):
                offset = fid.tell()
                for line in fid:
                    offset += len(line)
                    line = line.decode("ascii", "ignore")
                    if re_initmopovalue.match(line):
                        idx = int(line.split("-", 1)[-1].split()[0])
                        mopos = line.split(" ", 1)[-1].strip().strip('"')
                        try:
                            mopos = float(mopos)
                        except ValueError:
                            pass
                        position[idx] = mopos
                    elif re_wavelength.match(line):
                        m = line.split(" ", 1)[-1].strip()
                        wavelength = m.strip('""')
                    elif re_scanaxis.match(line):
                        scan_axis = line.split(" ", 1)[-1].strip().strip('"')
                    elif re_datacount.match(line):
                        length = line.split(" ", 1)[-1].strip().strip('"')
                        length = int(float(length))
                    elif re_measspeed.match(line):
                        speed = line.split(" ", 1)[-1].strip().strip('"')
                        meas_speed = float(speed)
                    elif re_measstep.match(line):
                        step = line.split(" ", 1)[-1].strip().strip('"')
                        meas_step = float(step)
                    elif re_headerend.match(line):
                        break

            line = fid.readline()
            line = line.decode("ascii", "ignore")
            offset = fid.tell()
            if re_datastart.match(line):
                lines = islice(fid, length)
                data = np.genfromtxt(lines)
                data = np.rec.fromrecords(
                    data, names=[scan_axis, "int", "att"]
                )
                fid.seek(offset)
                lines = islice(fid, length)
                dlength = np.sum([len(line) for line in lines])
                fid.seek(offset + dlength)
            elif re_measend.match(line) or line in (None, ""):
                break

        init_mopo = {}
        for k in keys:
            init_mopo[keys[k]] = position[k]
        fid.seek(offset)

    d["intensities"] = data["int"] * data["att"]
    d["twotheta"] = data["TwoThetaOmega"]
    d["omega"] = data["TwoThetaOmega"] / 2
    d["count_time"] = (1 / meas_speed) * meas_step
    d["wavelength"] = float(wavelength)

    return d

def _parse_xrdml_file(f):
    """
    Parses an XRML file

    Parameters
    ----------
    f: file-like object or string

    Returns
    -------
    d: dict
        A dictionary containing the XRDML file information.  The following keys
        are used:

        'intensities' - np.ndarray
            Intensities
        'twotheta' - np.ndarray
            Two theta values
        'omega' - np.ndarray
            Omega values
        'count_time' - float
            How long each point was counted for
        'wavelength' - float
            Wavelength of X-ray radiation
    """
    tree = et.parse(f)
    root = tree.getroot()
    ns = {"xrdml": "http://www.xrdml.com/XRDMeasurement/1.0"}

    query = {
        "intensities": ".//xrdml:intensities",
        "twotheta_start": ".//xrdml:positions[@axis='2Theta']"
        "/xrdml:startPosition",
        "twotheta_end": ".//xrdml:positions[@axis='2Theta']"
        "/xrdml:endPosition",
        "omega_start": ".//xrdml:positions[@axis='Omega']"
        "/xrdml:startPosition",
        "omega_end": ".//xrdml:positions[@axis='Omega']" "/xrdml:endPosition",
        "cnt_time": ".//xrdml:commonCountingTime",
        "kAlpha1": ".//xrdml:kAlpha1",
        "kAlpha2": ".//xrdml:kAlpha2",
        "ratio": ".//xrdml:ratioKAlpha2KAlpha1",
    }

    res = {key: root.find(value, ns).text for key, value in query.items()}

    kAlpha1 = float(res["kAlpha1"])
    kAlpha2 = float(res["kAlpha2"])
    ratio = float(res["ratio"])
    wavelength = (kAlpha1 + ratio * kAlpha2) / (1 + ratio)

    d = dict()

    intensities = np.fromstring(res["intensities"], sep=" ")
    n_pnts = intensities.size
    d["intensities"] = intensities
    d["twotheta"] = np.linspace(
        float(res["twotheta_start"]), float(res["twotheta_end"]), n_pnts
    )
    d["omega"] = np.linspace(
        float(res["omega_start"]), float(res["omega_end"]), n_pnts
    )
    d["count_time"] = float(res["cnt_time"])
    d["wavelength"] = wavelength

    return d
