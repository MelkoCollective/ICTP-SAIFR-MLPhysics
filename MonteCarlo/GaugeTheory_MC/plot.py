# Code to print observables as a function of temperature

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

L = 4
N_spins = 2*L**2

### Function for part 1: ###
def plot_versusMCTime():
  T = 1.0000
  
  file_init0 = open("ising2d_L%d_T%.4f_init0.dat" %(L,T), 'r')
  file_init1 = open("ising2d_L%d_T%.4f_init1.dat" %(L,T), 'r')

  data_init0 = np.loadtxt( file_init0 )
  data_init1 = np.loadtxt( file_init1 )

  time_init0    = data_init0[:,0]
  energy_init0  = data_init0[:,1]/(1.0*N_spins)

  time_init1    = data_init1[:,0]
  energy_init1  = data_init1[:,1]/(1.0*N_spins)

  #Plot energy vs. MC time:
  plt.figure()
  plt.plot( time_init0, energy_init0, '-' )
  plt.plot( time_init1, energy_init1, '-' )

  plt.xlabel('Monte Carlo time')
  plt.ylabel('energy')
  plt.title("%d x %d Ising model, T = %.3f" %(L,L,T))

### Function for part 2: ###
def plot_autocorrelation():
  T_list = np.linspace(5.0,1.5,15)
  autocorrTime = []

  for T in T_list:
    file = open("ising2d_L%d_T%.4f_init1.dat" %(L,T), 'r')
    data = np.loadtxt( file )

    time    = data[:,0].astype(int)
    energy  = data[:,1]/(1.0*N_spins)
    
    t_max = time[-1]
    autocorr = []
    t_array = np.array(range(0,100))
    for t in t_array:
      e1 = energy[0:(t_max-t+1)]
      e2 = energy[t:]
      autocorr.append( np.sum(e1*e2)/(1.0*(t_max-t)) - np.sum(e1)*np.sum(e2)/(1.0*(t_max-t)**2) )
    autocorr = np.array(autocorr)/autocorr[0]
    
    exp_func = lambda x,tau: np.exp(-x/tau)
    popt,perr = scipy.optimize.curve_fit(exp_func,t_array,autocorr)
    print popt
    autocorrTime.append(popt[0])
  
    plt.figure()
    plt.plot( t_array, autocorr, '-' )
    plt.plot( t_array, exp_func(t_array,*popt), '-')
    plt.yscale('log')
    plt.xlim([0,100])
    plt.ylim([10**(-2),1])
    plt.title('%d x %d Ising model, T = %.3f' %(L,L,T))
  #end loop over T
  
  plt.figure()
  plt.plot( T_list, autocorrTime, 'o-' )
  plt.xlabel('Temperature')
  plt.ylabel('Energy autocorrelation time')

### Function for part 3: ###
def plot_observables():
  T_list = np.linspace(5.0,0.5,19)
  energy   = []
  mag      = []
  specHeat = []
  susc     = []

  for T in T_list:
    file = open("ising2d_L%d_T%.4f_init1.dat" %(L,T), 'r')
    data = np.loadtxt( file )

    E   = data[:,1]
    #ESq = data[:,2]
    #M   = abs(data[:,3])
    #MSq = data[:,4]

    energy.append  ( np.mean(E)/(1.0*N_spins) )
    #mag.append     ( np.mean(M)/(1.0*N_spins) )
    #specHeat.append( (np.mean(ESq) - np.mean(E)**2)/(1.0*T**2*N_spins) )
    #susc.append    ( (np.mean(MSq) - np.mean(M)**2)/(1.0*T*N_spins) )
  #end loop over T

#plt.figure(figsize=(8,6))
  
  #plt.subplot(221)
  
  file_old = 'averages_L4_alpha1.txt'
  data_old = np.loadtxt( file_old )
  T_old = data_old[:,0]
  E_old = data_old[:,1]
  plt.plot(T_old,E_old,'o-')
  
  plt.plot(T_list, energy, 'o-')
  plt.xlabel('T')
  plt.ylabel('<E>')
#plt.xlim([1,3.54])

#  plt.subplot(222)
#  plt.plot(T_list, mag, 'o-')
#  plt.xlabel('T')
#  plt.ylabel('<M>')
#  plt.xlim([1,3.54])
#
#  plt.subplot(223)
#  plt.plot(T_list, specHeat, 'o-')
#  plt.xlabel('T')
#  plt.ylabel('c')
#  plt.xlim([1,3.54])
#
#  plt.subplot(224)
#  plt.plot(T_list, susc, 'o-')
#  plt.xlabel('T')
#  plt.ylabel('$\chi$')
#  plt.xlim([1,3.54])

  plt.xlim([0,5])

  plt.suptitle('%d x %d Ising model' %(L,L))
  #plt.tight_layout()


#Part 1: Determine the equilibration time:
#plot_versusMCTime()

#Part 2: Study the autocorrelation time:
#plot_autocorrelation()

#Part 3: Observables as a function of temperature
plot_observables()

plt.show()
