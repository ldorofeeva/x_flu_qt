# `X_flu_qt` 
`X_flu_qt` is a python GUI application for X-ray fluorescence map visualization.

Application supports visualization of NeXuS formatted x-ray fluorescence map files with extensions 
**.hdf5**, **.nxs** or **.nex**.
                        
To allow dynamic visualization, data is interpolated on a regular grid. Grid step and interpolation type is adjustable.

## Requirements
    python>=3.7
    
    pyqt5>=5.13
    numpy>=1.16
	scipy>=1.3
    h5py>=2.9
    matplotlib>=3

## Installation
In your python 3.7 environment:

    $ python setup.py install

## Launch
In your python 3.7 environment:

**On Windows**

    $ python -m x_flu_qt
    
**On Linux**

    $ x_flu

## Usage
* Select an x-ray fluorescence map file;
* Select entry;
* Adjust energy region either using a slider or entering energy range.

On energy region selection, the application searches for X-ray Emission Lines 
in the region of and prints out corresponding periodic table elements.

*Periodic Table and X-ray energies data are taken from BRUKER.*

**Advanced options**

* `Grid step` - size of the interpolation grid cell in millimiters;
* `Interpolation` - interpolation method (either nearest neighbour or linear);
* `Colormap` - allows to select one of four popular color maps.


**Smoothing the image**

The visualization is performed via matplotlib `imshow` that displays an image, 
i.e. data on a 2D regular raster. Depending on interpolation method being used, 
the resulting image will be smoother or coarser.

A coarse/smooth slider below the image can be used to select `imshow` 
interpolation method.