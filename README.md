# Tikhonov-Regularization-in-Photoacoustic-Imaging

This repository is supposed to provide the Python code used to obtain the results presented in my master's thesis with the same title.

USAGE:

This short repository is structured into the following three relevant files:

  1. utils.py: the script containing all the helper functions, from grid generation to the ones for time reversal/Tikhonov-based reconstruction
  2. main.py: the script contains four functions directly producing the figures shown in the thesis
  3. requirements.txt: the text file which can be used to directly reproduce the virtual environment used

NOTE:

Depending on the machine used, it might be infeasible to store in RAM the arrays used in the computations (as the code is currently structured for simplicity) and errors might be encountered. In that case, it is advisable to use the memmap function from the numpy package for storing/loading of the arrays to/from hard drive. 
