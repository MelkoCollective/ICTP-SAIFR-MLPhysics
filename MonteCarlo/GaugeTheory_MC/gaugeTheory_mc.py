# Monte Carlo for the Ising model

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import random

### Input parameters: ###
J = 1            #coupling parameter
L = 4            #linear size of the lattice
N_plaq  = L**2
N_spins = 2*L**2 #total number of spins

### Temperature list: ###
#T_list=[1.0000000000000000,1.0634592657106510,1.1269185314213019,1.1903777971319529,1.2538370628426039,1.3172963285532548,1.3807555942639058,1.4442148599745568,1.5076741256852078,1.5711333913958587,1.6345926571065097,1.6980519228171607,1.7615111885278116,1.8249704542384626,1.8884297199491136,1.9518889856597645,2.0153482513704155,2.0788075170810667,2.1422667827917179,2.2057260485023691,2.3326445799236715,2.3961038456343227,2.4595631113449739,2.5230223770556250,2.5864816427662762,2.6499409084769274,2.7134001741875786,2.7768594398982298,2.8403187056088810,2.9037779713195322,2.9672372370301834,3.0306965027408346,3.0941557684514858,3.1576150341621370,3.2210742998727881,3.2845335655834393,3.3479928312940905,3.4114520970047417,3.4749113627153929,3.5383706284260401]
#T_list = [1.5]
T_list = np.linspace(5.0,0.5,19)


random.seed(1234)
initialState = 1 #0 means start from all up state, 1 means start from random state

### Monte Carlo parameters: ###
n_eqSweeps = 100   #number of equilibration sweeps
n_bins = 1000    #total number of measurement bins
n_sweepsPerBin=1 #number of sweeps performed in one bin

### Parameters needed to show animation of spin configurations: ###
animate = False
bw_cmap = colors.ListedColormap(['black', 'white'])

### Function to calculate the total energy ###
def getEnergy():
  currEnergy = 0
  for i in range(N_plaq):
    currEnergy += -J*getPlaquetteProduct(i)
  return currEnergy

### Function to calculate the total magnetization ###
def getMag():
  return np.sum(spins)

def getPlaquetteProduct(i):
  return spins[2*i]*spins[(2*i)+1]*spins[2*neighbours[i,1]]*spins[(2*neighbours[i,0])+1]

### Function to perform one Monte Carlo sweep ###
def sweep():
  #do one sweep (N_spins local updates):
  for i in range(N_spins):
    #randomly choose which spin to consider flipping:
    spinLoc = random.randint(0,N_spins-1)
    
    #calculate the change in energy of the proposed move by considering the two plaquettes it will affect:
    plaq1 = spinLoc//2
    #get plaq2 based on whether the spin is on a horizontal or vertical link:
    if (spinLoc%2)==0:
      plaq2 = neighbours[plaq1,3]
    else:
      plaq2 = neighbours[plaq1,2]
    
    deltaE = 2*J*( getPlaquetteProduct(plaq1) + getPlaquetteProduct(plaq2) )
  
    if (deltaE <= 0) or (random.random() < np.exp(-deltaE/T)):
      #flip the spin:
      spins[spinLoc] = -spins[spinLoc]
  #end loop over i

### Store each spin's four nearest neighbours in a neighbours array (using periodic boundary conditions): ###
neighbours = np.zeros((N_plaq,4),dtype=np.int)
for i in range(N_plaq):
  #neighbour to the right:
  neighbours[i,0]=i+1
  if i%L==(L-1):
    neighbours[i,0]=i+1-L
  
  #upwards neighbour:
  neighbours[i,1]=i+L
  if i >= (N_plaq-L):
    neighbours[i,1]=i+L-N_plaq

  #neighbour to the left:
  neighbours[i,2]=i-1
  if i%L==0:
    neighbours[i,2]=i-1+L

  #downwards neighbour:
  neighbours[i,3]=i-L
  if i <= (L-1):
    neighbours[i,3]=i-L+N_plaq
#end of for loop

#initially, the spins are all up (a low-T phase) or all random (a high-T phase):
spins = np.ones(N_spins,dtype=np.int)
if initialState != 0:
  for i in range(N_spins):
    spins[i] = 2*random.randint(0,1) - 1 #either +1 or -1

#loop over all temperatures:
for T in T_list:
  print("T = %f"%T)
  
  #open a file where observables will be recorded:
  fileName         = "ising2d_L%d_T%.4f_init%d.dat" %(L,T,initialState)
  file_observables = open(fileName, 'w', 0)
  
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
    
    file_observables.write("%d \t %.8f \t %.8f \n" %(i, energy, mag))
  
    if animate:
      #Display the current spin configuration:
      plt.clf()
      plt.imshow( spins.reshape((L,L)), cmap=bw_cmap, norm=colors.BoundaryNorm([-1,0,1], bw_cmap.N) )
      plt.xticks([])
      plt.yticks([])
      plt.title("%d x %d Ising model, T = %.3f" %(L,L,T))
      plt.pause(0.01)
    #end if

    if (i+1)%1000==0:
      print "  %d" %(i+1)
  #end loop over i

  file_observables.close()
#end loop over temperature

if animate:
  plt.show()
