# Code to print observables as a function of temperature

import matplotlib.pyplot as plt
import numpy as np

T_list = np.linspace(5.0,0.5,19)
L = 4
N_spins = 2*L**2

energy   = []

for T in T_list:
  file = open('gaugeTheory2d_L%d_T%.4f_init1.txt' %(L,T), 'r')
  data = np.loadtxt( file )

  E   = data[:,1]
  energy.append  ( np.mean(E)/(1.0*N_spins) )
#end loop over T

plt.figure()
#  file_old = 'averages_L4_alpha1.txt'
#  data_old = np.loadtxt( file_old )
#  T_old = data_old[:,0]
#  E_old = data_old[:,1]
#  plt.plot(T_old,E_old,'o-')

plt.plot(T_list, energy, 'o-')
plt.xlabel('T')
plt.ylabel('<E>')
plt.xlim([0,5])

plt.suptitle('%d x %d Ising lattice gauge theory' %(L,L))

plt.show()
