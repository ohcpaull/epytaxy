import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from matplotlib.widgets  import RectangleSelector
from matplotlib.backend_bases import MouseButton
from lmfit.models import LorentzianModel, GaussianModel

class RHEEDPattern:
    """
    A RHEED image analysis tool. Plots integrated line profiles of 
    RHEED images to 
    """
    
    def __init__(self, tif_file, data_dir="."):
        
        self.image = tifffile.imread(os.path.join(data_dir, tif_file))
    
    def get_ROI(self, roi=None):
        fig, ax = plt.subplots(1,2)
        self.ax = ax
        self.fig = fig
        im = self.ax[0].imshow(
            self.image,
            cmap="magma"
        )
        ax[1].set(xlabel="Pixels", ylabel="Intensity (a.u.)")
        
        if roi == None:
            self.px1 = 0
            self.py1 = 0
            self.px2 = 0
            self.py2 = 0
            self.hz_sum = 0
            self.line, = self.ax[1].plot(self.hz_sum)

        def toggle_selector(event):
            print(' Key pressed.')
            if event.key in ['Q', 'q'] and toggle_selector.RS.active:
                print(' RectangleSelector deactivated.')
                toggle_selector.RS.set_active(False)
            if event.key in ['A', 'a'] and not toggle_selector.RS.active:
                print(' RectangleSelector activated.')
                toggle_selector.RS.set_active(True)
                
        def line_select_callback(eclick, erelease):
            'eclick and erelease are the press and release events'
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata

            self.px1 = int(x1)
            self.py1 = int(y1)
            self.px2 = int(x2)
            self.py2 = int(y2)

            cropped = self.image[self.py1:self.py2, self.px1:self.px2]
            self.hz_sum = np.sum(cropped, axis=0)
            self.line.set_data(np.arange(len(self.hz_sum)), self.hz_sum)
            
            # update figure
            self.ax[1].relim()
            self.ax[1].autoscale_view()
            self.fig.canvas.draw() 
            self.fig.canvas.flush_events()
            
            
            
            
        toggle_selector.RS = RectangleSelector(
            ax[0], line_select_callback,
            button=[1, 3],  # don't use middle button
            minspanx=5, 
            minspany=5,
            spancoords='pixels',
            interactive=True
        )
        
        #plt.connect('key_press_event', toggle_selector)

        plt.show()


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
            