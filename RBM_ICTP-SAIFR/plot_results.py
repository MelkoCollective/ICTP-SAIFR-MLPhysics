########## ICTP-SAIFR Minicourse on Machine Learning for Many-Body Physics ##########
### Roger Melko, Juan Carrasquilla, Lauren Hayward Sierens and Giacomo Torlai
### Tutorial 4: Print sampled observables as a function of temperature
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np

#Input parameters:
L           = 4    #linear size of the system
num_hidden  = 4    #number of hidden nodes

#Temperature list for which there are sampled observables stored in data_ising2d/RBM_observables
T_RBM = [1.0,1.254,1.508,1.762,2.016,2.269,2.524,2.778,3.032,3.286,3.540]

# Read in the observables sampled from the RBM and average over all bins:
E_RBM = []
M_RBM = []
C_RBM = []
S_RBM = []
for T in T_RBM:
  fileName = 'data_ising2d/RBM_observables/bins_nH%d_L%d' %(num_hidden,L)
  fileName += '_T' + str(T) + '.txt'
  file_RBM = open(fileName,'r')
  file_RBM.readline() #read the first line (comment line)
  data_RBM = np.loadtxt(file_RBM)
  E_RBM.append( np.mean(data_RBM[:,0]) )
  M_RBM.append( np.mean(data_RBM[:,1]) )
  C_RBM.append( np.mean(data_RBM[:,2]) )
  S_RBM.append( np.mean(data_RBM[:,3]) )

# Read in observables from MC data file:
file_MC = open('data_ising2d/MC_results_solutions/MC_ising2d_L%d_Observables.txt' %L,'r')
file_MC.readline() #read the first line (comment line)
data_MC = np.loadtxt(file_MC)
T_MC = data_MC[:,0]
E_MC = data_MC[:,1]
M_MC = data_MC[:,2]
C_MC = data_MC[:,3]
S_MC = data_MC[:,4]

# Read in the observables from the RBM_observables_solutions directory:
file_RBM_sol = open('data_ising2d/RBM_observables_solutions/RBM_nH%d_ising2d_L%d_Observables.txt' %(num_hidden,L),'r')
file_RBM_sol.readline() #read the first line (comment line)
data_sol = np.loadtxt(file_RBM_sol)
T_sol = data_sol[:,0]
E_sol = data_sol[:,1]
M_sol = data_sol[:,2]
C_sol = data_sol[:,3]
S_sol = data_sol[:,4]

### Plot the results: ###
fig = plt.figure(figsize=(10,7), facecolor='w', edgecolor='k')

plt.subplot(221)
plt.plot(T_MC,  E_MC,  'o-', label='MC')
plt.plot(T_sol, E_sol, 'o-', label='$n_H = %d$, sol.' %num_hidden)
plt.plot(T_RBM, E_RBM, 'o-', label='$n_H = %d$' %num_hidden)
plt.xlabel('$T$')
plt.ylabel('$<E>/N$')
plt.legend(loc='best')

plt.subplot(222)
plt.plot(T_MC, M_MC, 'o-')
plt.plot(T_sol, M_sol, 'o-')
plt.plot(T_RBM, M_RBM, 'o-')
plt.xlabel('$T$')
plt.ylabel('$<M>/N$')

plt.subplot(223)
plt.plot(T_MC, C_MC, 'o-')
plt.plot(T_sol, C_sol, 'o-')
plt.plot(T_RBM, C_RBM, 'o-')
plt.xlabel('$T$')
plt.ylabel('$C/N$')

plt.subplot(224)
plt.plot(T_MC, S_MC, 'o-')
plt.plot(T_sol, S_sol, 'o-')
plt.plot(T_RBM, S_RBM, 'o-')
plt.xlabel('T')
plt.ylabel('$\chi/N$')

plt.suptitle('%d x %d Ising model' %(L,L))

plt.show()
plt.savefig('RBM_observables.pdf')

