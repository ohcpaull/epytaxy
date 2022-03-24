import os
import os.path
import glob
import argparse
import re
import shutil
from time import gmtime, strftime
import string
import warnings
from contextlib import contextmanager
from operator import attrgetter
from numpy.lib.function_base import average
from scipy.optimize import leastsq, curve_fit
from scipy.stats import t
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py
import warnings
from epytaxy.spm.utils import(
    unit_vector,
    angle_between,
    Gauss2d,
)
from epytaxy.neutron.taipan import average_neighbour_difference
import dateutil


class NIST_TripleAxis:
    """
    Class to represent a datafile from BT4 or SPINS. 
    
    
    """
    
    def __init__(self, file_name, data_dir=None):
        

        if file_name.endswith(".bt4"):
            self.instrument = "BT4"
        elif file_name.endswith(".ng5"):
            self.instrument = "SPINS"
        else:
            raise ValueError("Wrong filetype. Must end with '.bt4' or 'ng5'.")

        if data_dir:
            self.file_dir = os.path.join(file_name, data_dir)
        else:
            self.file_dir = file_name
        
        self.cat = NIST_Catalogue(self.file_dir)

        self.scan_axis = []
        self.q_x = self.cat.qx
        self.q_y = self.cat.qy
        self.q_z = self.cat.qz
        self.cts = self.cat.counts
        self.q_mag = np.sqrt(self.cat.qx**2 + self.cat.qy**2 + self.cat.qz**2) 

        if self.cat.scan_units == "Q":
            self.average_qx = np.average(self.cat.qx)
            self.average_qy = np.average(self.cat.qy)
            self.average_qz = np.average(self.cat.qz)
        elif self.cat.scan_units == "B":
            self.average_a3 = np.average(self.cat.a3)
            self.average_a4 = np.average(self.cat.a4)

        for axis in ["qx", "qy", "qz"]:
            if average_neighbour_difference(getattr(self.cat, axis)) > 1e-4:
                self.scan_axis.append(axis)


    def plot(self, axis=None, fig=None, ax=None, **kwargs):
            if axis is None:
                axis = self.scan_axis[0]

            fig, ax = plt.subplots()

            ax.plot(getattr(self.cat, axis), self.cts/self.cat.bm2_time, **kwargs)
            ax.set(
                xlabel=axis,
                ylabel="Counts/sec"
            )
            return fig, ax


class NIST_Catalogue(object):
    """
    Extract all parts of a BT4 data file
    """

    def __init__(self, file):
        d = {}
        
        with open(file) as f:
            # First line of headers
            ln = f.readline().strip("\n")
            ln = ln.split()
            
            items = []
            for idx, ch in enumerate(ln):
                items.append(ch.strip("'"))
            
            d["date_time"] = dateutil.parser.parse(
                items[2] + " " + 
                items[1] + " " + 
                items[3] + " " + 
                items[4]
            )
            d["scan_units"] = items[5]
            d["monitor"] = items[6]
            d["num_pts"] = items[9]
            d["type"] = items[10]  

            #next two lines are describing the first line and redundant
            f.readline()
            f.readline()

            #Third line describes collimation, mosaic and crystal orientation
            ln = f.readline()
            ln = ln.split()
            d["collimation"] = np.array([float(ln[0]), float(ln[1]), float(ln[2]), float(ln[3])])
            d["mosaic"] = np.array([float(ln[4]), float(ln[5]), float(ln[6])])
            d["oop_dir"] = np.array([int(ln[7]), int(ln[8]), int(ln[9])])
            d["miscut_angle"] = float(ln[10])
            d["ip_dir"] = np.array([int(ln[11]), int(ln[12]), int(ln[13])])
            f.readline() # line describes previous metadata block

            # Lattice parameters and angles
            ln = f.readline()
            ln = ln.split()
            d["a_pc"] = float(ln[0])
            d["b_pc"] = float(ln[1])
            d["c_pc"] = float(ln[2])
            d["alpha"] = float(ln[3])
            d["beta"] = float(ln[4])
            d["gamma"] = float(ln[5])
            f.readline() # line describes previous metadata block

            # Energy transfer metadata
            ln = f.readline()
            ln = ln.split()
            d["E_center"] = ln[0]
            d["delta_E"] = ln[1]
            d["E_i"] = ln[2]
            d["M-dsp"] = ln[3]
            d["A-dsp"] = ln[4]
            d["Tmp start"] = ln[5]
            d["Tmp inc"] = ln[6]
            f.readline() # line describles previous metadata block

            # Scan metadata
            ln = f.readline()
            ln = ln.split()
            d["Q center"] = np.array([float(ln[0]), float(ln[1]), float(ln[2])])
            d["delta Q"] = np.array([float(ln[3]), float(ln[4]), float(ln[5])])
            d["H field"] = ln[6]
            ln = f.readline()

            if d["scan_units"] == "B":
                for idx, mot in enumerate(range(6)):
                    ln = f.readline()
                    ln = ln.split()
                    d[f"a{idx+1}_start"] = ln[1]
                    d[f"a{idx+1}_step"] = ln[2]
                    d[f"a{idx+1}_end"] = ln[3]

                f.readline()

            # Measured data
            ln = f.readline()
            lines = f.readlines()
            q_x = np.empty(len(lines))
            q_y = np.empty(len(lines))
            q_z = np.empty(len(lines))
            E = np.empty(len(lines))
            T_act = np.empty(len(lines))
            mins = np.empty(len(lines))
            counts = np.empty(len(lines))

            for idx, line in enumerate(lines):
                line = line.split()
                if len(line) ==  7:
                    q_x[idx] = float(line[0])
                    q_y[idx] = float(line[1])
                    q_z[idx] = float(line[2])
                    E[idx] = float(line[3])
                    T_act[idx] = float(line[4])
                    mins[idx] = float(line[5])
                    counts[idx] = float(line[6])
                elif len(line) == 6:
                    q_x[idx] = float(line[0])
                    q_y[idx] = float(line[1])
                    q_z[idx] = float(line[2])
                    E[idx] = float(line[3])
                    mins[idx] = float(line[4])
                    counts[idx] = float(line[5])       

            d["qx"] = q_x
            d["qy"] = q_y
            d["qz"] = q_z
            d["E"] = E
            d["T_act"] = T_act
            d["mins"] = mins
            d["counts"] = counts

            self.cat = d

    def __getattr__(self, item):
        return self.cat[item]


class NIST_RSM(object):
    """
    Reciprocal space map measured by BT4 in a series of linescans. 

    Parameters
    ----------
    directory   :   str or os.path-like object
        directory to data files
    range       :   2-tuple
        (datafile_number_min, datafile_number_max)
        Min and max datafile numbers to be loaded for the reciprocal space map in ``directory``.
    verbose     :   bool
        Mainly for debugging and checking things
    """
    def __init__(self, directory, prefix=None, files=[], verbose=False):
        self.scans = []
        self.step_axis = []
        self.prefix = prefix

        for file in files:
            self.scans.append(NIST_TripleAxis(os.path.join(directory, file)))

        motor_averages = {}
        for scan in self.scans:
            for axis in ["qx", "qy", "qz"]:

                if axis not in scan.scan_axis:
                    motor_averages[axis] = getattr(scan, f"average_{axis}")

            
        for motor, diff in motor_averages.items():
            if diff > 1e-4:
                self.step_axis.append(motor)
        print(np.shape(self.scans))
        self.data = np.empty(
            [len(self.scans), len(self.scans[0].cts)],
            dtype=[("qx", "f4"), ("qy", "f4"), ("qz", "f4"), ("cts", "f4")]
        )

        self._load_files()
    
    def __len__(self):
        return len(self.scans)

    def __str__(self):
        qx_min = min(self.data["qx"].flatten())
        qx_max = max(self.data["qx"].flatten())
        qy_min = min(self.data["qy"].flatten())
        qy_max = max(self.data["qy"].flatten())
        qz_min = min(self.data["qz"].flatten())
        qz_max = max(self.data["qz"].flatten())

        return (
            f"{len(self)} TPN files in reciprocal space map.\n"
            f"-------------------------------------\n"
            f"Q ranges:\nq_h \t ({qx_min:.3f}, {qx_max:.3f})\n"
            f"q_k \t ({qy_min:.3f}, {qy_max:.3f})\n"
            f"q_l \t ({qz_min:.3f}, {qz_max:.3f})"
        )

    def sort_by(self, axis):
        """
        Sort files by attribute. Useful if your RSM files aren't measured in
        succession in reciprocal space.

        Parameters
        ----------
        axis    :   {"qh", "qk", "ql", "s1", "s2", "ei", "ef", "filename"}
        """

        if axis in ["qx", "qy", "qz", "a3", "a4"]:
            #rsm.sort(key=lambda x: np.average(getattr(x.cat, axis))
            scans = sorted(self.scans, key=attrgetter(f"average_{axis}"))
            self.scans = scans
        elif axis in "filename":
            scans = sorted([nist_datafile_number(s.cat.filename, prefix=self.prefix, filetype=".bt4") for s in self.scans])
            self.scans = scans

        self._load_files()

    def _load_files(self):
        for idx, scan in enumerate(self.scans):
            #print(idx)
            try:
                self.data["qx"][idx, :] = scan.cat.qx[:]
                self.data["qy"][idx, :] = scan.cat.qy[:]
                self.data["qz"][idx, :] = scan.cat.qz[:]
                self.data["cts"][idx, :] = scan.cts[:]
                #print(max(scan.cts))
            except ValueError:
                warnings.warn(
                    "Incomplete linescan detected. Attempting to fill in gaps by averaging."
                )
                # If one of the scans was aborted early for some reason, 
                # make a reasonable guess for what the rest of the motor positions will be, and fill out the missing arrays.
                normal_length = max([len(getattr(s, "cts")) for s in self.scans])
                num_missing = normal_length - len(scan.cts)
                qx_step = average_neighbour_difference(scan.cat.qx)
                qy_step = average_neighbour_difference(scan.cat.qy)
                qz_step = average_neighbour_difference(scan.cat.qz)
                
                # For missing detector data, get the average of the neighbouring linescans
                # and assign to missing point
                missing_counts = []
                for i in np.arange(len(scan.cat.qx),len(scan.cat.qx)+num_missing):
                    # If missing data is in the middle of good data, take average of neighbouring linescans
                    if (self.scans[idx] is not self.scans[-1]) and (self.scans[idx] is not self.scans[0]):
                        avg = np.average([
                            self.scans[idx-1].cts[i],
                            self.scans[idx+1].cts[i],
                        ])
                        missing_counts.append(avg)
                    # else the missing data is on the end-points of RSM, and we just take it as zero
                    else:
                        missing_counts.append(0)
                    
                counts = np.append(
                    scan.cts,
                    missing_counts
                )
                
                qx = np.append(
                    scan.cat.qx, 
                    np.linspace(
                        scan.cat.qx[-1]+qx_step,
                        scan.cat.qx[-1]+(num_missing)*qx_step,
                        num_missing
                    )
                )
                qy = np.append(
                    scan.cat.qy, 
                    np.linspace(
                        scan.cat.qy[-1]+qy_step,
                        scan.cat.qy[-1]+(num_missing)*qy_step,
                        num_missing
                    )
                )
                qz = np.append(
                    scan.cat.qz, 
                    np.linspace(
                        scan.cat.qz[-1]+qz_step,
                        scan.cat.qz[-1]+(num_missing)*qz_step,
                        num_missing
                    )
                )
                
                self.data["qx"][idx, :] = qx
                self.data["qy"][idx, :] = qy
                self.data["qz"][idx, :] = qz
                self.data["cts"][idx, :] = counts

    def plot(self, axis_1="qy", axis_2="qz", fig=None, ax=None, log=False, **kwargs):
        """
        Plot RSM quickly
        """
        fig, ax = plt.subplots()
        x = self.data[axis_1]
        y = self.data[axis_2]
        z = self.data["cts"]

        if log:
            z = np.log10(z)

        cs = ax.contourf(
            x,
            y,
            z,
            **kwargs
        )
        ax.set(
            xlabel=axis_1,
            ylabel=axis_2,
        )
        fig.colorbar(cs, label="Counts", ax=ax, shrink=0.9)

        return fig, ax

def nist_datafile_number(fname, prefix="mgtrn", filetype=".bt4"):
    """
    From a filename figure out what the run number was

    Parameters
    ----------
    fname : str
        The filename to be processed

    Returns
    -------
    run_number : int
        The run number

    Examples
    --------
    >>> datafile_number('mgtrn043.bt4')
    708

    """
    rstr = ".*" + prefix + "([0-9]{5})" + filetype
    regex = re.compile(rstr)

    _fname = os.path.basename(fname)
    #print(_fname)
    r = regex.search(_fname)
    #print(r)
    if r:
        return int(r.groups()[0])

    return None