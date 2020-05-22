# GCNN
Graph convolutional neural network implemention using PyTorch

# Requirements
Python==3.8.0
PyTorch==1.5.0
scipy==1.4.1
numpy==1.18.1
scikit-learn==0.22.2

# Testing the GCNN
Note that the parameters used in the examples were randomly chosen, and should be tested thoroughly when doing real projects. 
1. Peratom quantities
The first task is to predict the potential energies of particles for a binary Lennard-Jones mixture, the corresponding files are 'dump.dat','model_peratom.py' and 'test_peratom.py'. Use 'load_data.py' to extract the data set, then python test_peratom.py to train the model. The validation was automatically executed after the training. 
The molecular dynamics simulations were performed using LAMMPS. The system contains 4,000 particles with composition of A80B20. 
5 snapshots obtained from 5 independent MD runs were adopted, where the former 4 snapshots are the training set, the last snapshot is used as testing set. 
This network can also be applied to predict other per-atom quantities such as magnetic moment, propensity etc. 

2. Crystal properties
The second task is to predict the formation energies of a bunch of Lennard-Jones crystals. The corresponding files are 'Extract_data.py', 'LJ_crystals.py', 'model_crystal.py' and 'test_crystal.py'. The LJ crystals are obtained using genetic algorithm implemented in GA package. 2,600 crystals with different compositions were collected, and their formation energies were calculated with respect to A, B FCC crystals. The 2,600 crystals were splited into 26 batches to improve the training efficiency, 22 of which were adopted as training set, and the remaining 4 batches were adopted as testing set. To merge the crystals with different number of atoms into 1 batch, we have to shift the indices of atoms along with the crystal ids, for example crystal-0 has 10 atoms, and the crystal-1 has 8 atoms, then in the batch the two crystals are presented as:
[0,1,2...10,11,12...18], where the former 10 atoms are for crystal-0 and the later 8 atoms are for crystal-1. The corresponding neighboring atom ids were shited too, and a list that maps the shifted atom ids back to crystal_ids is required, with the form [[0,1,2,...10],[11,12...18]]. The above mentioned steps were incoperated in 'Extract_data.py'.
