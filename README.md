# discrete_elementGPU
This code simulates the discrete element method for a predifined number of motile discs, which are called cells.
Each cell can respond to an external signalling gradient and the parallel code is implemented over the cells. The code was originally developed to run on an NVIDIA Tesla C1060

DEMCode is a CUDA implementation of the discrete element method that can be used to model collective cell migration.


The code implements a group of cells that migrate through a corridor.

The code can be compiled with nvcc and a choice of c++ compiler with the Makefile specified. Important note: System paths must be changed to make the executable.

To compile for unix based operating systems try:

make Makefile.

The executable will be created. 


# Running the code and files

Once the code is compiled, the model can be simulated by running main:

./main run=1

The executable can take two arguments. run=1 specifies that this is a new instance of the model and data will be returned at the end of the simulation. run=2 specifies that data can be imported and the model will be initialised with the imported data.


Positions of the cells and velocities are written to a file particle_data.dat.

This data can be imported into the open source visualisation software PARAVIEW http://www.paraview.org/.
To convert particle_data.dat to VTK file format for visualisation see Discrete-element-method-visualisation
