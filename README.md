# GCNN
Graph convolutional neural network implemention using PyTorch

# Requirements
Python==3.8.0
PyTorch==1.5.0
scipy==1.4.1
numpy==1.18.1
scikit-learn==0.22.2
Pymatgen==2020.4.29
Matplotlib==3.2.1

# Testing the GCNN
Note that the parameters used in the examples were randomly chosen, and should be tested thoroughly when doing real projects. 
1. Peratom quantities
The first task is to predict the potential energies of particles for a binary Lennard-Jones mixture, the corresponding files are 'dump.dat'. Use 'load_data.py' to extract the data set, then python train.py to train the model.
The molecular dynamics simulations were performed using LAMMPS. The system contains 4,000 particles with composition of A80B20. 

2. Crystal properties
The second task is to predict the formation energies of a bunch of crystals. Just prepare the cif files and Turn off the sitemode in train.py,and it should work.
