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
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
#%matplotlib notebook
#%matplotlib inline

#__all__ = ["lattice", "find_nearest"]

class RSMtools(object):
    
    def __init__(self, refHKL, iHKL, oHKL, substrateMat, filepath=None, **kwargs):
        # omega 2theta and data arrays. Omega and 2theta are 1D arrays while data is a 2D array with dimensions of omega and 2theta
        self.omega = []
        self.tt = []
        self.data = []
        self.meta = []
        self.delta = []
        self.theoOm, self.theoTT = [], []
        self.qx, self.qy, self.qz = [], [], []
        self.omCorr, self.ttCorr = [], []
        self.gridder = []
        self.rsmRef = []
        self.iHKL = iHKL
        self.oHKL = oHKL
        self.h, self.k, self.l = 0, 0, 0
        self.maxInt = 0   
        self.p = []
        self.substrateMat = substrateMat
        self.rsmRef = refHKL
        self.axisraw = []
        self.omega1D = []
            
        [self.substrate, self.hxrd] = self.initialise_substrate( wavelength='CuKa12', **kwargs )

        if filepath:
            self.load_sub( filepath )


    def load_sub( self, filepath = None, refHKL = None ):

        self.filepath = filepath
        if filepath is None:
            self.filepath = self.openDialogue()   # open GUI for user to select file 
        b = self.filepath.index('.') # find the period that separates filename and extension
        a = self.filepath.rfind('/')  # find the slash that separates the directory and filename
        ext = self.filepath[(b+1):] # get extension and determine whether it is RAS or XRDML
        self.filename = self.filepath[(a+1):(b)]

        if ext == 'xrdml':
            (self.omega, self.tt, self.data) = self.xrdml_file(self, self.filepath)
        elif ext == 'ras':
            (self.omega, self.tt, self.data) = self.ras_file(self, self.filepath)
        else:
            print('filetype not supported.')
        if refHKL is None:    
            self.p, [self.qx, self.qy, self.qz] = self.align_sub_peak()
        else:
            self.p, [self.qx, self.qy, self.qz] = self.align_sub_peak( refHKL )

    def load_film( self, filepath = None ):
        if filepath is None:
            film_file = self.openDialogue()
        else:
            film_file = filepath
        b = film_file.index('.') # find the period that separates filename and extension
        ext = film_file[(b+1):] # get extension and determine whether it is RAS or XRDML
        a = film_file.rfind('/')  # find the period that separates the directory and filename
        (f_omega, f_tt, f_data) = self.xrdml_file( self, film_file )
        return f_omega, f_tt, f_data

        
    @staticmethod
    def xrdml_file(self, file):
        data = xrdtools.read_xrdml(file)
        om = data['Omega']
        tt = data['2Theta']
        
        return (np.transpose(om),
                np.transpose(tt),
                np.transpose(data['data']))

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
    
    def openDialogue(self):
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename()
        filename = os.path.basename(filepath)
        return filepath
    
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

        
        a = ax.contourf(om[0,:],tt[:,0], np.log10(self.data), **kwargs)

        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(r'2$\theta$-$\omega$')
        fig.show()
        return ax
    
    def plotQ(self, xGrid, yGrid, dynLow, dynHigh, fig=None, ax=None, nlev = None, show_title=False, axlabels='hkl', **kwargs):
        if ax is None or fig is None:
            fig, ax = plt.subplots()

        if show_title == True:
            ax.set_title( self.filename )
        
        
        self.gridder = xu.Gridder2D(xGrid,yGrid)
        self.gridder(self.qy, self.qz, self.data)
        intHigh = np.argmax(self.data)
        intMin = np.argmin(self.data)
        dynhigh = np.rint(np.log10(intHigh))

        INT = xu.maplog(self.gridder.data, dynLow, dynHigh)
        levels = np.linspace(1, dynhigh, num=20)
        levels = 10**(levels)
        #print(levels)
        ax.contourf(self.gridder.xaxis, self.gridder.yaxis, np.transpose(INT), nlev, **kwargs)

        if axlabels == 'ipoop':
            xlabel = '$Q_{ip}$'
            ylabel = '$Q_{oop}$'
            ax.set_xlabel(xlabel, fontsize=18)
            ax.set_ylabel(ylabel, fontsize=18)
        else:
            xlabel = [ '\\bar{'+str(-i)+'}' if i < 0 else str(i) for i in self.iHKL ]
            ylabel = [ '\\bar{'+str(-i)+'}' if i < 0 else str(i) for i in self.oHKL ]
            ax.set_xlabel(r'$Q_{[' + xlabel[0] + '' + xlabel[1] + '' + xlabel[2] + ']}$', fontsize=18)
            ax.set_ylabel(r'$Q_{[' + ylabel[0] + '' + ylabel[1] + '' + ylabel[2] + ']}$', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        return fig, ax

    def to_csv(self, fname=None):
        if fname is None:
            fname = self.filename + "_CSVoutput.csv"

        tmp = np.array(
            [self.omega.flatten(), 
            self.tt.flatten(), 
            self.data.flatten()]
            )
        np.savetxt(fname, tmp, delimiter=',')
        
    def findSubstratePeak(self, axis1, axis2, data):
        ''' Returns indices of axis1 and axis2 where the substrate peak is positioned (based off maximum counts)'''
        maxIndex = np.argmax(data)
        return np.unravel_index( maxIndex, (len(data[:,0]), len(data[0,:])))

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
            substrate = xu.materials.Crystal("YAlO3", xu.materials.SGLattice(221, 3.720, \
                            atoms=[Y, Al, O], pos=['1a', '1b', '3c']))
            hxrd = xu.HXRD(substrate.Q(int(self.iHKL[0]), int(self.iHKL[1]), int(self.iHKL[2])), \
                           substrate.Q(int(self.oHKL[0]), int(self.oHKL[1]), int(self.oHKL[2])), en=energy, **kwargs)
           
        return [substrate, hxrd]
    
    def align_sub_peak(self, delta = None):
        nchannel = 255
        chpdeg = nchannel/2.511 #2.511 degrees is angular acceptance of xrays to detector
        center_ch = 128
        energy = 1240/0.154

        indexMax = np.argmax(self.data)
        self.maxInt = indexMax

        data = np.array(self.data)
        tupleIndex = np.unravel_index( indexMax, (len(self.data[:,0]), len(self.data[0,:])) )
        
        
        [self.theoOm, dummy, dummy, self.theoTT] = self.hxrd.Q2Ang(self.substrate.Q(self.rsmRef))
        expTT = self.tt[tupleIndex[0],tupleIndex[1]]
        expOm = self.omega[tupleIndex[0],tupleIndex[1]]
        print('experimental omega = ' + str(expOm))
        print('experimental tt = ' + str(expTT))
        print('theoretical omega = ' + str(self.theoOm))
        print('theoretical 2theta = ' + str(self.theoTT))

        expOm, expTT, p, cov = xu.analysis.fit_bragg_peak( self.omega, self.tt, self.data, expOm, expTT, self.hxrd, plot=False)
        
        offset = (self.theoTT/2) + self.theoOm
        omnominal = (self.theoTT/2) + offset
        self.hxrd.Ang2Q.init_linear('y+', center_ch, nchannel, chpdeg=chpdeg) 

        if delta is None:
            self.delta = (expOm - self.theoOm, expTT - self.theoTT)
        else:
             self.delta = delta
        print('delta = ' + str( self.delta ) )
        
        

        [qx, qy, qz] = self.hxrd.Ang2Q(self.omega, self.tt, delta=[expOm - self.theoOm, expTT - self.theoTT] )
        
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


        tupleIndex = np.unravel_index(np.argmax(self.gridder.data[xminInd:xmaxInd,yminInd:ymaxInd].flatten()), \
                     (len(self.gridder.data[xminInd:xmaxInd, 0]), len(self.gridder.data[0,yminInd:ymaxInd])))


        cropData = self.gridder.data[yminInd:ymaxInd, xminInd:xmaxInd]
        cropQxGrid, cropQzGrid = np.meshgrid(cropQx, cropQz)
        xGrid, yGrid = np.meshgrid( self.gridder.xaxis, self.gridder.yaxis)
        xC = cropQx[tupleIndex[0]]
        yC = cropQz[tupleIndex[1]]
        amp = self.gridder.data[tupleIndex]
        xSigma = 0.1
        ySigma = 0.1
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


def ras_file( file ):
    # Read RAS data to object
    rasFile = xu.io.rigaku_ras.RASFile(file)
        
    self.scanaxis = rasFile.scans.scan_axis
    self.stepSize = rasFile.scans.meas_step
    self.measureSpeed= rasFile.scans.meas_speed
    self.dataCount = rasFile.scans.length
    # Read raw motor position and intensity data to large 1D arrays

    ax1, data = xu.io.getras_scan(rasFile.filename+'%s', '', self.scanaxis)

    npinte = np.array(data['int'])
    
    
    # Read omega data from motor positions at the start of each 2theta-Omega scan
    om = [rasFile.scans[i].init_mopo['Omega'] for i in range(0, len(rasFile.scans))]
    # Convert 2theta-omega data to 1D array
    
    return (np.transpose(omga), np.transpose(tt), np.transpose(intensities))