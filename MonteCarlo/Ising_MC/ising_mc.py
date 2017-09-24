########## ICTP-SAIFR Minicourse on Machine Learning for Many-Body Physics ##########
### Roger Melko, Juan Carrasquilla and Lauren Hayward Sierens
### Tutorial 1: Monte Carlo for the Ising model
#####################################################################################

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import random

### Input parameters: ###
T_list = np.linspace(5.0,0.5,19) #temperature list
L = 4                            #linear size of the lattice
N_spins = L**2                   #total number of spins
J = 1                            #coupling parameter

### Monte Carlo parameters: ###
n_eqSweeps = 10   #number of equilibration sweeps
n_bins = 100     #total number of measurement bins
n_sweepsPerBin=20 #number of sweeps performed in one bin

### Parameters needed to show animation of spin configurations: ###
animate = False
bw_cmap = colors.ListedColormap(['black', 'white'])

### Initially, the spins are in a random state (a high-T phase): ###
spins = np.zeros(N_spins,dtype=np.int)
for i in range(N_spins):
  spins[i] = 2*random.randint(0,1) - 1 #either +1 or -1

### Store each spin's four nearest neighbours in a neighbours array (using periodic boundary conditions): ###
neighbours = np.zeros((N_spins,4),dtype=np.int)
for i in range(N_spins):
  #neighbour to the right:
  neighbours[i,0]=i+1
  if i%L==(L-1):
    neighbours[i,0]=i+1-L
  
  #upwards neighbour:
  neighbours[i,1]=i+L
  if i >= (N_spins-L):
    neighbours[i,1]=i+L-N_spins
  
  #neighbour to the left:
  neighbours[i,2]=i-1
  if i%L==0:
    neighbours[i,2]=i-1+L
  
  #downwards neighbour:
  neighbours[i,3]=i-L
  if i <= (L-1):
    neighbours[i,3]=i-L+N_spins
#end of for loop

### Function to calculate the total energy ###
def getEnergy():
  currEnergy = 0
  for i in range(N_spins):
    currEnergy += -J*( spins[i]*spins[neighbours[i,0]] + spins[i]*spins[neighbours[i,1]] )
  return currEnergy
# end of getEnergy() function

### Function to calculate the total magnetization ###
def getMag():
  return np.sum(spins)
# end of getMag() function

### Function to perform one Monte Carlo sweep ###
def sweep():
  #do one sweep (N_spins local updates):
  for i in range(N_spins):
    #randomly choose which spin to consider flipping:
    site = random.randint(0,N_spins-1)
      
    deltaE = 0
    #calculate the change in energy of the proposed move by considering only the nearest neighbours:
    for j in range(4):
      deltaE += 2*J*spins[site]*spins[neighbours[site,j]]
  
    if (deltaE <= 0) or (random.random() < np.exp(-deltaE/T)):
      #flip the spin:
      spins[site] = -spins[site]
  #end loop over i
# end of sweep() function

#################################################################################
########## Loop over all temperatures and perform Monte Carlo updates: ##########
#################################################################################
for T in T_list:
  print('\nT = %f' %T)
  
  #open a file where observables will be recorded:
  fileName         = 'ising2d_L%d_T%.4f.dat' %(L,T)
  file_observables = open(fileName, 'w')
  
  #equilibration sweeps:
  for i in range(n_eqSweeps):
    sweep()

  #start doing measurements:
  for i in range(n_bins):
    for j in range(n_sweepsPerBin):
      sweep()
    #end loop over j
    
    energy = getEnergy()
    mag    = getMag()
    
    file_observables.write('%d \t %.8f \t %.8f \n' %(i, energy, mag))
  
    if animate:
      #Display the current spin configuration:
      plt.clf()
      plt.imshow( spins.reshape((L,L)), cmap=bw_cmap, norm=colors.BoundaryNorm([-1,0,1], bw_cmap.N) )
      plt.xticks([])
      plt.yticks([])
      plt.title('%d x %d Ising model, T = %.3f' %(L,L,T))
      plt.pause(0.01)
    #end if

    if (i+1)%1000==0:
      print '  %d' %(i+1)
  #end loop over i

  file_observables.close()
#end loop over temperature

if animate:
  plt.show()
