import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import pathlib
from lmfit.models import LorentzianModel, GaussianModel
from matplotlib.widgets import SpanSelector
from scipy.fft import fft, fftfreq
import seaborn as sns

class XPSpec:
    """
    Reads Ultraviolet Photoelectron Spectroscopy .VMS files. 
    Keeps metadata and calculates kinetic energy and binding energy.

    Contains a wrapper for `lmfit` fitting of the peaks in the data

    Parameters
    ----------
    file : str, os.path
        Filename of the UPS file
    data_dir : str, os.path
        Directory of the data
    """
    def __init__(self, file, data_dir="."):
        
        with open(os.path.join(data_dir, file)) as f:
            for i in range(23):
                f.readline()

            year = int(f.readline().strip())
            month = int(f.readline().strip())
            day = int(f.readline().strip())
            for i in range(13):
                f.readline()
           
            excitation_energy = f.readline().strip()
            for i in range(8):
                f.readline()
            work_function = f.readline().strip()
            for i in range(9):
                f.readline()
            units = f.readline().strip()
            start_energy = float(f.readline().strip())
            energy_step = float(f.readline().strip())

            for i in range(14):
                f.readline()
   
            data = f.read().splitlines() 
            
        self.metadata = {
            "date" : datetime.date(year, month, day),
            "excitation_energy" : float(excitation_energy),
            "energy_units" : units,
            "work_function" : float(work_function)
        }
        data = [float(i) for i in data[:-1]]
        energy = start_energy + np.arange(len(data)) * energy_step
        benergy = -self.metadata["excitation_energy"] + energy #+ self.metadata["work_function"]
        d = {
            "Kinetic Energy" : energy, 
            "Binding Energy" : benergy,
            "Counts" : data
        }

        self.data = pd.DataFrame(data=d)
        self.fitting = {}
    
    def binding_energy(self, kinetic_energy):
        return - self.metadata["excitation_energy"] + kinetic_energy
    
    def kinetic_energy(self, binding_energy):
        return binding_energy + self.metadata["excitation_energy"]

    def fit(self, model, params, fit_region=None, method="least_squares", fit_ID=None, xunits="KE"):
        """
        Fits experimental data to a model and parameters inputted by the user. 

        Parameters
        ----------
        model : `lmfit.models.Model`
            Can be a custom-defined model, but more likely a pre-defined standard 
            model like `lmfit.models.LorentzianModel` or `lmfit.models.GausianModel`. 
        params : `lmfit.parameter.Parameters`
            A set of parameters that are required to fit a model. Should include a 
            starting 'guess' value, 
        

        """
        if fit_ID == None:
            fit_ID = f"fit_{len(self.fitting)}"
        self.fitting[fit_ID] = {"model" : model, "params" : params, "res" : None}

        if xunits == "KE":
            units = "Kinetic Energy"
        elif xunits == "BE":
            units = "Binding Energy"
        else: 
            raise ValueError("Units not recognised!")

        if fit_region:
            region = self.data[self.data[units].between(fit_region[0], fit_region[1])]
        else:
            region = self.data

        self.fitting[fit_ID]["res"] = self.fitting[fit_ID]["model"].fit(region["Counts"], self.fitting[fit_ID]["params"], x=region[units], method=method)
        self.fitting[fit_ID]["fit_components"] = self.fitting[fit_ID]["res"].eval_components(x=self.data[units])
    
    def plot(self, fig=None, ax=None, xunits="KE", **kwargs):
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        if xunits == "BE":
            units = "Binding Energy"
        elif xunits == "KE":
            units = "Kinetic Energy"
        else: 
            raise ValueError("Units not recognised!")

        x = self.data[units]

        ax.plot(self.data[units], self.data["Counts"], ".", **kwargs)
        
        if len(self.fitting):
            # Plot each fit that has been performed
            for id in self.fitting.keys():
                colours = sns.color_palette("colorblind", len(self.fitting[id]["fit_components"]))
                # Check if fit is an array (only relevant for constant models)
                if not hasattr(self.fitting[id]["res"].eval(self.fitting[id]["res"].params, x=self.data[units]), "__len__"):
                    y = np.full(len(x), self.fitting[id]["res"].eval(self.fitting[id]["res"].params, x=self.data[units]))
                else:
                    y = self.fitting[id]["res"].eval(self.fitting[id]["res"].params, x=self.data[units])
                # plot best fit
                ax.plot(x, y, '-', color="red", label=f"{id} best fit")
                
                # plot fit components
                for idx, key in enumerate(self.fitting[id]["fit_components"].keys()):
                    if not hasattr(self.fitting[id]["fit_components"][key], "__len__"):
                        y = np.full(len(x), self.fitting[id]["fit_components"][key])
                    else:
                        y = self.fitting[id]["fit_components"][key]
                    ax.plot(x, y, "--", color=colours[idx], markersize=1, label=f"{id} {key}")
        
        ax.set(xlabel=units, ylabel="Intensity (a.u.)")
        plt.legend()
        return fig, ax


class UVSpec:
    """
    Reads Ultraviolet Photoelectron Spectroscopy .VMS files. 
    Keeps metadata and calculates kinetic energy and binding energy.

    Contains a wrapper for `lmfit` fitting of the peaks in the data

    Parameters
    ----------
    file : str, os.path
        Filename of the UPS file
    data_dir : str, os.path
        Directory of the data
    """

    def __init__(self, file, data_dir="."):
        
        with open(os.path.join(data_dir, file)) as f:
            for i in range(23):
                f.readline()

            year = int(f.readline().strip())
            month = int(f.readline().strip())
            day = int(f.readline().strip())
            for i in range(21):
                f.readline()
            work_function = f.readline().strip()
            for i in range(9):
                f.readline()
            units = f.readline().strip()
            excitation_energy = float(f.readline().strip())
            energy_step = float(f.readline().strip())           
            start_energy = excitation_energy
            

            for i in range(14):
                f.readline()
   
            data = f.read().splitlines() 
            
        self.metadata = {
            "date" : datetime.date(year, month, day),
            "excitation_energy" : float(excitation_energy),
            "energy_units" : units,
            "work_function" : float(work_function)
        }
        data = [float(i) for i in data[:-1]]
        energy = start_energy + np.arange(len(data)) * energy_step
        benergy = -self.metadata["excitation_energy"] + energy #+ self.metadata["work_function"]
        d = {
            "Kinetic Energy" : energy, 
            "Binding Energy" : benergy,
            "Counts" : data
        }

        self.data = pd.DataFrame(data=d)
    
    def binding_energy(self, kinetic_energy):
        return - self.metadata["excitation_energy"] + kinetic_energy
    
    def kinetic_energy(self, binding_energy):
        return binding_energy + self.metadata["excitation_energy"]

    def fit(self, model, params, fit_region=None, method="least_squares", fit_ID=None, xunits="KE"):
        """
        Fits experimental data to a model and parameters inputted by the user. 

        Parameters
        ----------
        model : `lmfit.models.Model`
            Can be a custom-defined model, but more likely a pre-defined standard 
            model like `lmfit.models.LorentzianModel` or `lmfit.models.GausianModel`. 
        params : `lmfit.parameter.Parameters`
            A set of parameters that are required to fit a model. Should include a 
            starting 'guess' value, 
        

        """
        if fit_ID == None:
            fit_ID = f"fit_{len(self.fitting)}"
        
        self.fitting[fit_ID] = {"model" : model, "params" : params, "res" : None}

        if xunits == "KE":
            units = "Kinetic Energy"
        elif xunits == "BE":
            units = "Binding Energy"
        else: 
            raise ValueError("Units not recognised!")

        if fit_region:
            region = self.data[self.data[units].between(fit_region[0], fit_region[1])]
        else:
            region = self.data

        self.fitting[fit_ID]["res"] = self.fitting[fit_ID]["model"].fit(region["Counts"], self.fitting[fit_ID]["params"], x=region[units], method=method)
        self.fitting[fit_ID]["fit_components"] = self.fitting[fit_ID]["res"].eval_components(x=self.data[units])
    
    def plot(self, fig=None, ax=None, xunits="KE", **kwargs):
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        if xunits == "BE":
            units = "Binding Energy"
        elif xunits == "KE":
            units = "Kinetic Energy"
        else: 
            raise ValueError("Units not recognised!")

        x = self.data[units]

        ax.plot(self.data[units], self.data["Counts"], ".", **kwargs)
        
        if len(self.fitting) > 0:
            for id in self.fitting.keys():
                colours = sns.color_palette("colorblind", len(self.fitting[id]["fit_components"]))
                if not hasattr(self.fitting[id]["res"].eval(self.fitting[id]["res"].params, x=self.data[units]), "__len__"):
                    y = np.full(len(x), self.fitting[id]["res"].eval(self.fitting[id]["res"].params, x=self.data[units]))
                else:
                    y = self.fitting[id]["res"].eval(self.fitting[id]["res"].params, x=self.data[units])
                ax.plot(x, y, '-', color="red", label=f"{id} best fit")

                for idx, key in enumerate(self.fitting[id]["fit_components"].keys()):
                    if not hasattr(self.fitting[id]["fit_components"][key], "__len__"):
                        y = np.full(len(x), self.fitting[id]["fit_components"][key])
                    else:
                        y = self.fitting[id]["fit_components"][key]
                    ax.plot(x, y, "--", color=colours[idx], markersize=1, label=f"{id} {key}")
        ax.set(xlabel=units, ylabel="Intensity (a.u.)")
        plt.legend()
        return fig, ax
    

class THz:
    """
    Reads THz spectroscopy data from the setup of Sylvain Massabeau and 
    Henri-Yves Jaffres. Allows to interactively plot the spectrum, select and region
    of the spectrum and subsequently view the FFT of the selected area. 
    """
    def __init__(self, fID, data_dir=".", decimal="."):
        """
        This method initializes an instance of the THz class. The method takes 
        two arguments: fID and data_dir. fID is an identifier that is used to 
        search for a file in the data_dir directory. The method looks for a 
        file that has fID in its name and reads it using the pandas read_csv 
        function. The data is then stored in the data attribute of the 
        class instance.
        """
        if isinstance(fID, str):
            self.data = pd.read_csv(
                os.path.join(data_dir, fID), 
                delimiter="\t",
                names = ["time", "Efield", "dE/dt"],
                dtype = float,
                decimal=decimal
            )
        elif isinstance(fID, int):
            for file in os.listdir(data_dir):
                if str(fID) in file and file.endswith(".dat"):
                    self.data = pd.read_csv(
                        os.path.join(data_dir, file), 
                        delimiter="\t",
                        names = ["time", "Efield", "dE/dt"],
                        dtype = float,
                        decimal=decimal
                    )
                    print(file)
                    break
        else:
            raise ImportError("fID not identified!")

    def interactive_fft(self):
        """
        This method generates an interactive plot of the Fourier transform of the 
        data stored in the data attribute. The method creates a figure with two 
        subplots - one for the time-domain signal and one for the frequency-domain 
        signal. The time-domain signal is plotted using the data stored in the 
        data attribute. The frequency-domain signal is initially empty. The method 
        then creates a SpanSelector widget on the time-domain subplot that allows 
        the user to select a region of interest. When a region is selected, the 
        method calculates the Fourier transform of the selected region using the 
        fft function from the numpy module. The frequency-domain signal is then 
        updated with the calculated Fourier transform.
        """
        self.fig, self.ax = plt.subplots(2)


        self.ax[0].plot(
            self.data["time"],
            self.data["Efield"],
        )
        self.ax[0].set(xlabel="Time delay (ps)", ylabel="Intensity (a.u.)")
        self.ax[1].set(xlabel="Frequency (GHz)", ylabel="Intensity (a.u.)")


        self.line, = self.ax[1].plot([], [], color="red")
        self.ax[1].set_yscale("log")
        self.ax[1].set(xlim=(0,4000))
        def onselect(vmin, vmax):
            self.idx1, self.idx2 = np.searchsorted(self.data["time"], (vmin, vmax))
            N = self.idx2 - self.idx1
            
            Ecrop = self.data["Efield"].values[self.idx1:self.idx2]
            Tcrop = self.data["time"].values[self.idx1:self.idx2]
            
            dt = Tcrop[1] - Tcrop[0]

            self.fE = fft(Ecrop)[1:N//2]
            self.fT = fftfreq(N, dt/1e12)[1:N//2]/1e9
            self.line.set_data(self.fT, np.abs(self.fE)/max(np.abs(self.fE)))
            
            # update figure
            self.ax[1].relim()
            self.ax[1].autoscale_view()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        self.span = SpanSelector(
            self.ax[0],
            onselect,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="tab:blue"),
            interactive=True,
            drag_from_anywhere=True
        )


        self.fig.show()

    def fixed_fft(self, bounds=None):

        if bounds == None:
            bounds = (self.data["time"].values[0], self.data["time"].values[-1])
                      
        self.idx1, self.idx2 = np.searchsorted(self.data["time"], (bounds[0], bounds[1]))
        N = self.idx2 - self.idx1
        dt = self.data["time"][1] - self.data["time"][0]
        self.fft = {
            "frequency" : fftfreq(N, dt/1e12)[1:N//2]/1e9,
            "intensity" : fft(self.data["Efield"].values)[1:N//2]
        }
        return (self.fft["frequency"], self.fft["intensity"])


class PolarTHz:
    """
    Class that loads the angle-dependent THz emission of of a sample in positive and 
    negative magnetic field
    """
    def __init__(self, data_dir_posB, data_dir_negB, angle_step, start_angle=0, sampleID=None, ftype=".dat"):
        """
        Loading data here. Requires data for each magnetic field to be in separate folders, 
        and that the angle step for each measurement set is the same.

        Parameters
        ----------
        data_dir_posB : str, os.path
            data directory for the angle-dependent measurements in +B magnetic field
        data_dir_negB : str, os.path
            data directory for the angle-dependent measurements in -B magnetic field
        angle_step : int
            angle step in degrees between each measurement of the angle dependence
        start_angle : int, optional
            Optional input of start angle in case you want to align it with a specific axis
        sampleID : str, optional
            The ID of the sample used in the measurement set
        ftype : str, optional
            filetype used to express THz emission data. Default is .dat
        """

        theta = []
        # load data for +B field
        posB_measure = {}
        for idx, filepath in enumerate(pathlib.Path(data_dir_posB).glob(f"*{ftype}")):
            theta.append(int(start_angle + idx * angle_step + 0.5))
            file = os.path.basename(filepath)
            posB_measure[f"{theta[idx]} deg"] = THz(file, data_dir=data_dir_posB)
        
        self.theta = np.array(theta)
        
        # load data for -B field
        negB_measure = {}
        for idx, filepath in enumerate(pathlib.Path(data_dir_negB).glob(f"*{ftype}")):
            file = os.path.basename(filepath)
            negB_measure[f"{theta[idx]} deg"] = THz(file, data_dir=data_dir_negB)

        self.measurements = pd.DataFrame({"+B" : posB_measure, "-B" : negB_measure})

    def magnetic(self, angle):
        """
        Calculates the magnetic signal of THz emission at a specific angle.

        Parameters
        ----------
        angle : int
            Angle of rotation of the sample w.r.t. the incident pump pulse polarisation
        """
        if angle not in self.theta:
            raise ValueError("Provided angle not in measured angles!")
        else:
            return (self.measurements["+B"][f"{angle} deg"].data["Efield"] - self.measurements["-B"][f"{angle} deg"].data["Efield"])/2

    def nonmagnetic(self, angle):
        """
        Calculates the magnetic signal of THz emission at a specific angle.

        Parameters
        ----------
        angle : int
            Angle of rotation of the sample w.r.t. the incident pump pulse polarisation
        """
        if angle not in self.theta:
            raise ValueError("Provided angle not in measured angles!")
        else:
            return (self.measurements["+B"][f"{angle} deg"].data["Efield"] + self.measurements["-B"][f"{angle} deg"].data["Efield"])/2     

    def polar(self):
        """
        Calculate the peak-to-trough value of an emitted THz pulse as a function of angle 
        """
        self.magnetic_pp = np.zeros(len(self.theta))
        self.nonmagnetic_pp = np.zeros(len(self.theta))

        for idx, a in enumerate(self.theta):
            mag = self.magnetic(a)
            self.magnetic_pp[idx] = max(mag) - min(mag)
            nonmag = self.nonmagnetic(a)
            self.nonmagnetic_pp[idx] = max(nonmag) - min(nonmag)
        
        return self.magnetic_pp, self.nonmagnetic_pp


   
