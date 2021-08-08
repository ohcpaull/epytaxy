import numpy as np
import refnx.util.general as general
import refnx.util.ErrorProp as EP
import xml.etree.ElementTree as et
from refnx.dataset import ReflectDataset
import re
from itertools import islice

# mm
XRR_BEAMWIDTH_SD = 0.019449

def reduce_xray(f, bkg=None, scale=None, sample_length=None, clip=0):
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
    if f.endswith('.xrdml'):
        spec = parse_xrdml_file(f)
    elif f.endswith('.ras'):
        spec = parse_ras_file(f)

    reflectivity = spec["intensities"][clip:] / spec["count_time"]
    reflectivity = reflectivity
    reflectivity_s = np.sqrt(spec["intensities"][clip:]) / spec["count_time"]
    reflectivity_s = reflectivity_s
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
            reflectivity, reflectivity_s, total_bkgd, total_bkgd_s
        )

    # work out the Q values
    qx, qy, qz = general.q2(
        spec["omega"][clip:],
        spec["twotheta"][clip:],
        np.zeros_like(spec["omega"][clip:]),
        spec["wavelength"],
    )

    # do a footprint correction
    if sample_length is not None:
        footprint_correction = general.beamfrac(
            np.array([XRR_BEAMWIDTH_SD]) * 2.35,
            np.array([sample_length]),
            spec["omega"][clip:],
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
            scale = np.max(reflectivity[qz > 0.0318])

    reflectivity /= scale
    reflectivity_s /= scale

    d = ReflectDataset(data=(qz, reflectivity, reflectivity_s))

    return d


def parse_ras_file(f):
    """
    Parses a RAS file

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

    d["intensities"] = data["int"]*data["att"]
    d["twotheta"] = data["TwoThetaOmega"]
    d["omega"] = [angle/2 for angle in data["TwoThetaOmega"]]
    d["count_time"] = (1/meas_speed) * meas_step
    d["wavelength"] = float(wavelength)

    return d


def parse_xrdml_file(f):
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
        "K-Alpha1": ".//xrdml:kAlpha1",
        "K-Alpha2": ".//xrdml:kAlpha2",
        "ratio": ".//xrdml:ratioKAlpha2KAlpha1",
    }

    res = {key: root.find(value, ns).text for key, value in query.items()}


    kAlpha1 = float(res["K-Alpha1"])
    kAlpha2 = float(res["K-Alpha2"])
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
