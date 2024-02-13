import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, re
from matplotlib.patches import Rectangle
from matplotlib.widgets  import RectangleSelector
from matplotlib.backend_bases import MouseButton
from lmfit.models import LorentzianModel, GaussianModel
import tifffile, datetime
from uncertainties import unumpy

class RHEEDPattern:
    """
    A RHEED image analysis tool. Plots integrated line profiles of 
    RHEED images to 
    """
    
    def __init__(self, tif_file, data_dir="."):
        
        self.image = tifffile.imread(os.path.join(data_dir, tif_file))
    
    def get_ROI(self, roi=None):

                
        def select_callback(eclick, erelease):
            """
            Callback for line selection.

            *eclick* and *erelease* are the press and release events.
            """
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
            print(f"The buttons you used were: {eclick.button} {erelease.button}")
            
            self.px1 = int(x1)
            self.py1 = int(y1)
            self.px2 = int(x2)
            self.py2 = int(y2)

            cropped = self.image[self.py1:self.py2, self.px1:self.px2]
            self.hz_sum = np.sum(cropped, axis=0)[:,0]
            self.line.set_data(np.arange(len(self.hz_sum)), self.hz_sum)
            
            # update figure
            self.ax[1].relim()
            self.ax[1].autoscale_view()
            self.fig.canvas.draw() 
            self.fig.canvas.flush_events()
            
        self.fig, self.ax = plt.subplots(1, 2, layout='constrained')

        self.ax[0].imshow(
            self.image,
            cmap="magma"
        )
        self.ax[1].set(xlabel="Pixels", ylabel="Intensity (a.u.)")
        
        if roi is None:
            self.px1 = 0
            self.py1 = 0
            self.px2 = 0
            self.py2 = 0
            self.hz_sum = 0
            self.line, = self.ax[1].plot(self.hz_sum)
        
        self.RS = RectangleSelector(
            self.ax[0], select_callback,
            useblit=True,
            button=[1, 3],  # don't use middle button
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,
        )
        #plt.connect('key_press_event', toggle_selector)

        plt.show()

    def fit_RHEED(self, peak_params, fit_method="least_squares"):
        
        model = GaussianModel(prefix="BG_")
        for idx, key in enumerate(peak_params.keys()):
            model += LorentzianModel(prefix=f"L{idx}_")

        params = model.make_params()
        x = np.arange(len(self.hz_sum))
        self.res = model.fit(self.hz_sum, params, x=x, method=fit_method)
        self.comps = self.res.eval_components(x=x)

class RHEEDOscillator:
    """
    Class to process RHEED patterns
    
    Parameters
    ----------
    file_name : str, os.path
        File to be read, which may contain multiple regions of interest (ROI)
    num_roi : int
        The number of ROIs in the data file
    file_dir : str, os.path
        Directory where the RHEED file is located
    """
    
    def __init__(self, file_name, num_roi, file_dir="."):
        self.roi = []
        self.data_dir = file_dir
        self.file_name = file_name
        self.file = os.path.join(self.data_dir, file_name)
        
        
        try:
            # Assume data is saved from the KsA file
            
            # Read file into dataframe
            data = pd.read_csv(self.file, sep="	", header=None)
            data.columns = ["time", "intensity"]

            # Create dictionary for splitting the data
            d = {}
            time = data["time"].unique()
            d["time"] = time

            # Split data based of unique time values, corresponding to different ROIs
            for i in range(num_roi): 
                d[f"roi_{i}"] = np.array(data["intensity"][int(i*len(time)):int((i+1)*len(time))])

            # Put data into dataframe
            self.d = d
            self.data = pd.DataFrame(d)
        except ValueError:
            # The data is saved from the graph and is in a different format
            
            # Read file into dataframe
            data = pd.read_csv(self.file, sep="	", header=None)
            columns = ["time"]
            for i in range(num_roi):
                columns.append(f"roi_{i}")
            
            data.columns = columns
            
            self.data = data
            
        
    def clip_data(self, end_time, start_time=0, replace=False):
        data = self.data.query('time > @start_time and time < @end_time')
        if replace == True:
            self.data = data
            
            #self.data.time.reset_index(inplace=True, drop=True)
            self.data.time = self.data.time - start_time
            return
        else:
            return data
        
    def plot(self, roi=0, average = None):
        fig, ax = plt.subplots()
        
        if isinstance(roi, list):
            if average:
                for r in roi:
                    ax.plot(self.data.time, self.data[f"roi_{r}"].rolling(average).mean(), label=f"roi{r}")
                ax.set_xlabel("time (s)")
                ax.set_ylabel("Intensity (a.u.)")   
                ax.legend()
            else:
                ax.plot(self.data.time, self.data[f"roi_{roi}"])
                ax.set_xlabel("time (s)")
                ax.set_ylabel("Intensity (a.u.)")
        else:
            if average:
                ax.plot(self.data.time, self.data[f"roi_{roi}"].rolling(average).mean())
            else:
                ax.plot(self.data.time, self.data[f"roi_{roi}"])
            ax.set_xlabel("time (s)")
            ax.set_ylabel("Intensity (a.u.)")
        return fig, ax
    
def pol_correction(up, down, S=0.3, uncertainty=False):

    up = unumpy.uarray(up, np.sqrt(up))
    down = unumpy.uarray(down, np.sqrt(down))


    P = 1/S * (up - down)/(up + down)
    pos_true = (1 + P) * (up + down)/2
    neg_true = (1 - P) * (up + down)/2

    if uncertainty:
        return pos_true, neg_true
    return pos_true.n, neg_true.n


def read_arpes_1D(file, data_dir=None):
    re_info1 = re.compile(r"Info 1")
    
    re_num_regions = re.compile(r"Number of Regions")
    re_region_name = re.compile(r"Region Name")

    re_dim_1_name = re.compile(r"Dimension 1 name")
    re_dim_1_size = re.compile(r"Dimension 1 size")
    re_dim_1_scale = re.compile(r"Dimension 1 scale")

    re_dim_2_name = re.compile(r"Dimension 2 name")
    re_dim_2_size = re.compile(r"Dimension 2 size")
    re_dim_2_scale = re.compile(r"Dimension 2 scale")

    re_pass_energy = re.compile(r"Pass Energy")
    re_num_sweeps = re.compile(r"Number of Sweeps")
    re_excitation_energy = re.compile(r"Excitation Energy")
    re_low_energy = re.compile(r"Low Energy")
    re_high_energy = re.compile(r"High Energy")
    re_energy_step = re.compile(r"Energy Step")
    re_step_time = re.compile(r"Step Time")
    re_date = re.compile(r"Date")
    re_time = re.compile(r"Time")
    re_sample = re.compile(r"Sample")
    re_thetaX = re.compile(r"ThetaX")
    re_thetaY = re.compile(r"ThetaY")

    re_data = re.compile(r"Data 1")
    

    if data_dir:
        fpath = os.path.join(data_dir, file)
    else:
        fpath = file
    with open(fpath, 'r') as f:
        f.readline()
        regions = f.readline()
        #print(regions)
        if re_num_regions.search(regions):
            num_regions = int(regions.split("=")[1])
            for i in range(2):
                f.readline()
            #print(num_regions)
            dicts = []
            for reg in range(num_regions):
                #print(reg)

                dicts.append({})
                f.readline()
                dicts[reg]["region_name"] = f.readline().split("=")[1].strip("\n")
                dicts[reg]["dimension_1_name"] = f.readline().split("=")[1].strip("\n")
                dicts[reg]["dimension_1_size"] = int(f.readline().split("=")[1])
                f.readline()
                f.readline()
                dicts[reg]["signal_name"] = f.readline().split("=")[1].strip("\n")
                dicts[reg]["signal_unit"] = f.readline().split("=")[1].strip("\n")
                for i in range(4):
                    f.readline()

                dicts[reg]["region_name"] = f.readline().split("=")[1].strip("\n")
                for i in range(2):
                    f.readline()

                dicts[reg]["num_passes"] = int(f.readline().split("=")[1].strip("\n"))
                dicts[reg]["excitation_energy"] = float(f.readline().split("=")[1].strip("\n"))
                for i in range(4):
                    f.readline()
                
                dicts[reg]["low_energy"] = f.readline().split("=")[1].strip("\n")
                dicts[reg]["high_energy"] = f.readline().split("=")[1].strip("\n")
                dicts[reg]["energy_step"] = f.readline().split("=")[1].strip("\n")
                dicts[reg]["step_time"] = f.readline().split("=")[1].strip("\n")
                for i in range(2):
                    f.readline()
                dicts[reg]["spectrum_name"] = f.readline().split("=")[1].strip("\n")
                for i in range(3):
                    f.readline()
                dicts[reg]["sample"] = f.readline().split("=")[1].strip("\n")
                f.readline()
                date = f.readline().split("=")[1]
                date = date.strip("\n").split("-")
                dicts[reg]["date"] = datetime.date(int(date[0]), int(date[1]), int(date[2]))

                time = f.readline().split("=")[1].strip("\n").split(":")
                dicts[reg]["time"] = datetime.time(int(time[0]), int(time[1]), int(time[2]))
                for i in range(2):
                    f.readline()
                dicts[reg]["theta_x"] = float(f.readline().split("=")[1])
                dicts[reg]["theta_y"] = float(f.readline().split("=")[1])

                for i in range(5):
                    f.readline()
                
                x = np.zeros(dicts[reg]["dimension_1_size"])
                y = np.zeros(len(x))
                for n in range(dicts[reg]["dimension_1_size"]):
                    ln = f.readline()
                    #print(ln)
                    x[n] = ln.split()[0]
                    y[n] = ln.split()[1]
                
                f.readline()

                dicts[reg]["x"] = x
                dicts[reg]["y"] = y
    return dicts