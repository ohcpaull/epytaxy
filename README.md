epytaxy
=====

# Overview

epytaxy is a python software package to read and analyse datafiles that is typically seen in epitaxial thin film research. This includes instruments within our lab, other specialised labs we access, as well as national research infrastructure. The package is split into 3 sections:

1. Scanning probe microscopy `epytaxy.spm`
    - Asylum Cypher single-frequency piezoresponse force microscopy (PFM) `epytaxy.spm.AsylumSF`
    - Asylum Cypher Dual-Amplitude Resonance Tracking (DART) PFM `epytaxy.spm.AsylumDART`
    - Bruker multimode AFM and PFM 
2. X-ray scattering `epytaxy.xray`
    - X-ray reflectometry (XRR) `epytaxy.xray.reduce_xray`
    - X-ray diffraction line-scan (XRD) `epytaxy.xray.ras_file`, `epytaxy.xray.xrdml_file`
    - X-ray reciprocal space map (RSM)  `epytaxy.xray.ReciprocalSpaceMap`
3.  Neutron scattering `vlabs.neutron`
    - TAIPAN triple-axis spectrometer `epytaxy.neutron.TaipanNexus`, `epytaxy.neutron.TaipanRSM`

# Installing epytaxy from git
<img align="right" width="300" src="https://firstcontributions.github.io/assets/Readme/fork.png" alt="fork this repository" />

#### If you don't have git on your machine, [install it](https://help.github.com/articles/set-up-git/).

## Fork this repository

Fork this repository by clicking on the fork button on the top of this page.
This will create a copy of this repository in your account.

## Clone the repository

<img align="right" width="300" src="https://firstcontributions.github.io/assets/Readme/clone.png" alt="clone this repository" />

Now clone the forked repository to your machine. Go to your GitHub account, open the forked repository, click on the code button and then click the _copy to clipboard_ icon.

Open a terminal and run the following git command:

```
git clone "url you just copied"
```

where "url you just copied" (without the quotation marks) is the url to this repository (your fork of this project). See the previous steps to obtain the url.

<img align="right" width="300" src="https://firstcontributions.github.io/assets/Readme/copy-to-clipboard.png" alt="copy URL to clipboard" />

For example:

```
git clone https://github.com/this-is-you/valanoorlabs.git
```

where `this-is-you` is your GitHub username. Here you're copying the contents of the first-contributions repository on GitHub to your computer.

## Install cloned repository with `pip`

Open Anaconda Prompt, and navigate to the directory where the cloned valanoorlabs folder is held (e.g. `cd C:\Users\user_name\Documents\code` if valanoorlabs is in the code folder). Then do the following:
```
pip install -e epytaxy
```
