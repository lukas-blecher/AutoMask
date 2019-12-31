# AutoMask - a Blender Add-on for automatic rotoscoping
This repository contains a Blender 2.8x Add-on that can mask an object in the `Movie Clip Editor` with the help of Machine Learning.
AutoMask is the communication between Blender and SiamMask.

## Installation
1. Download the repository
Download the repository as `.zip` file and extract it or clone it to your computer.
2. Python 
To use this Add-on, several 3rd party libraries are required (see `requirements.txt`). You can either install the dependencies to
the Blender python or if python is already installed on the system and the python version is compatible with the Blender python version you can also install the dependencies on your system.
3. Adding the Project to the Python path
Open `automask.py` in your favorite text editor and replace `PYTHON_PATH` with the path to you python site-packages if needed and `PROJECT_DIR` with the path to the directory you downloaded this repository to.
4. Install Dependencies
* PyTorch 
The neural network that does the heavy lifting is written for [PyTorch](https://pytorch.org/).
A Nvidia GPU is recommended but not necessarily required. However, it does speed up the process significantly. If you have a supported GPU install pytorch with GPU support.
* Other Requirements
Most of the required libraries are standard and easy to install. 
```pip install -r requirements.txt```
5. Model
The model weights for SiamMask can be downloaded from [http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth](http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth). Save the file to the subfolder `trackers/SiamMask` under the name `model.pth`. 

## Acknowledgements
[SiamMask](https://github.com/foolwood/SiamMask), [THOR](https://github.com/xl-sr/THOR)
