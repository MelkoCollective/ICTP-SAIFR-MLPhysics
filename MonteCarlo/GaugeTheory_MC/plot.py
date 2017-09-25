################# ICTP-SAIFR Minicourse on Machine Learning for Many-Body Physics #################
### Roger Melko, Juan Carrasquilla and Lauren Hayward Sierens
### Tutorial 1: Print observables as a function of temperature for the Ising lattice gauge theory
###################################################################################################

import matplotlib.pyplot as plt
import numpy as np

### Input parameters (these should be the same as in ising_mc.py): ###
T_list = np.linspace(5.0,0.5,19)  #temperature list
L = 4                             #linear size of the lattice
N_spins = 2*L**2                  #total number of spins (one spin on each link)

energy   = []

for T in T_list:
  file = open('gaugeTheory2d_L%d_T%.4f.txt' %(L,T), 'r')
  data = np.loadtxt( file )

  E   = data[:,1]
  energy.append  ( np.mean(E)/(1.0*N_spins) )
#end loop over T

plt.figure()

#Plot the solution for L=4:
file_sol = 'EvsT_gaugeTheory_L4_solution.txt'
data_sol = np.loadtxt( file_sol )
T_sol = data_sol[:,0]
E_sol = data_sol[:,1]
plt.plot(T_sol,E_sol,'-', lw=3)

#Plot the results from the Monte Carlo simulation:
plt.plot(T_list, energy, 'o-')
plt.xlabel('$T$')
plt.ylabel('$<E>$')
plt.xlim([0,5])

plt.suptitle('%d x %d Ising lattice gauge theory' %(L,L))

plt.show()
