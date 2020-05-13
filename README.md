# GCNN
Graph convolutional neural network implemention using PyTorch

# Requirements
Python 3.8.0
PyTorch 1.5.0
scipy 1.4.1
numpy 1.18.1
scikit-learn 0.22.2

# Testing the GCNN
The GCNN was trained to predict the potential energies of particles for a binary Lennard-Jones mixture.
The molecular dynamics simulations were performed using LAMMPS. The system contains 4,000 particles with composition of A80B20. 
5 snapshots obtained from 5 independent MD runs were adopted, where the former 4 snapshots are the training set, the last snapshot is used as testing set. 

