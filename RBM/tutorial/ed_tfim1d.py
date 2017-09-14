import numpy as np
import argparse

# Define Pauli operators
Id = np.eye(2)
Sx = np.array([[0,1.],[1,0]])
Sz = np.array([[1,0.],[0,-1]])

# Pauli X operator 
def sigmaX(L,i):
    ''' Return the many-body operator
        I x I x .. x Sx x I x .. x I
        with Sx acting on spin i '''
    OpList = []
    for k in range(L):
        if (k == i):
	    OpList.append(Sx)
	else:
	    OpList.append(Id)

    return reduce(np.kron,OpList)

# Pauli Z operator
def sigmaZ(L,i):
    ''' Return the many-body operator
        I x I x .. x Sz x I x .. x I
        with Sz acting on spin i '''
    OpList = []
    for k in range(L):
        if (k == i):
	    OpList.append(Sz)
	else:
	    OpList.append(Id)

    return reduce(np.kron,OpList)

# Magnetic interaction term 
def buildMagneticInteraction(i,B,L):
    return B*sigmaX(L,i)

# Ising interaction term 
def buildIsingInteraction(i,j,J,L):
    ''' Return the Ising interaction term
        I x .. x Sz x Sz x .. x I
        with Sz acting on spins i and j '''
    OpList = []
    for k in range(L):
	if (k == i):
	    OpList.append(Sz)
	elif (k == j):
	    OpList.append(Sz)
	else:
	    OpList.append(Id)

    return J*reduce(np.kron,OpList)

# Build transverse-field Ising model
def build1dIsingModel(L,J,B,OBC):
    
    D = 1<<L    # dimension of Hilbert space
    ''' Return the full Hamiltonian '''
    Ham = np.zeros((D,D))	
    for i in range(L-1):
	Ham = Ham - buildIsingInteraction(i,i+1,J,L)
	Ham = Ham - buildMagneticInteraction(i,B,L)	
    Ham = Ham - buildMagneticInteraction(L-1,B,L)
	
    # Periodic Boundary Conditions
    if OBC is True:
        Ham = Ham - buildIsingInteraction(0,L-1,J,L)
    
    return Ham

# Generate the training dataset
def buildDataset(L,B,psi,Nsamples):

    D = 1<<L
    config = np.zeros((D,L))
    psi2 = np.zeros((D))  
    
    # Build all spin states and psi^2
    for i in range(D):
        state = (bin(i)[2:].zfill(L)).split()
        for j in range(L):
            config[i,j] = int(state[0][j])
        psi2[i] = psi[i]**2 
    
    config_index = range(D)
    
    # Generate the trainset
    index_samples = np.random.choice(config_index,
                                     Nsamples,
                                     p=psi2)
    dataName  = '../data/tfim1d/datasets/tfim1d_L' + str(L)
    dataName += '_B' + str(B) + '_train.txt'
    dataFile = open(dataName,'w')
    for i in range(Nsamples):
        for j in range(L):
            dataFile.write("%d " % config[index_samples[i]][j])
        dataFile.write("\n")
    dataFile.close()
    
    # Generate the testset
    index_samples = np.random.choice(config_index,
                                     Nsamples/10,
                                     p=psi2)
    dataName  = '../data/tfim1d/datasets/tfim1d_L' + str(L)
    dataName += '_B' + str(B) + '_test.txt'
    dataFile = open(dataName,'w')
    for i in range(Nsamples/10):
        for j in range(L):
            dataFile.write("%d " % config[index_samples[i]][j])
        dataFile.write("\n")

# MAIN
def main(pars):
	
    print ('\n--------------------------------\n')
    print ('EXACT DIAGONALIZATION OF THE 1d-TFIM\n')
     
    # Parameters
    L = pars.L;         # number of spins
    B = pars.B          # magnetic field 
    J = pars.J          # ising interaction strength
    Nsamples = 10000    # train dataset size
    
    print ('Number of spins    L = %d' % L)
    print ('Ising interaction  J = %.1f' % J)
    print ('\n\n')

    # Magnetic field range
    Bmin = 0.0
    Bmax = 2.0
    Bsteps = 10
    deltaB = (Bmax-Bmin)/float(Bsteps)
    
    # Observables file
    obs_Name = '../data/tfim1d/observables/ed_tfim1d_L' + str(L) + '_observables.txt'
    obs_File = open(obs_Name,'w')
    obs_File.write('# B               E         ')  # write header
    obs_File.write('<|Sz|>           <Sx>\n')       # write header
    
    # Spin-spin correlation file
    corr_Name = '../data/tfim1d/observables/ed_tfim1d_L' + str(L) + '_correlations.txt'
    corr_File = open(corr_Name,'w')
    
    # Loop over magnetic field values
    for b in range(1,Bsteps+1):
        
        B = Bmin + b*deltaB
        print('Magnetic field B = %.2f' % B) 

        # Wavefunction file
        psiName  = '../data/tfim1d/wavefunctions/wavefunction_tfim1d_L' + str(L)
        psiName += '_B' + str(B) + '.txt'
        
        # Diagonalize the Hamiltonian
        print('diagonalizing...')
        H = build1dIsingModel(L,J,B,True)
        (e,psi) = np.linalg.eigh(H)
        psi0 = np.abs(psi[:,0])
        e0 = e[0]
        
        # Save energy and wavefunction
        obs_File.write('%.1f   %.10f   ' % (B,e0/float(L)))
        np.savetxt(psiName,psi0)
        
        # Magnetic observables
        print('computing observables...')
        # Compute <|Sz|>
        sZ = 0.0
        for i in range(1<<L):
            config = (bin(i)[2:].zfill(L)).split()
            tmp = 0.0
            for j in range(L):
                tmp += 2*float(config[0][j])-1
            sZ += psi0[i]**2*abs(tmp)/float(L)
        # Compute <Sx>
        sX = np.inner(np.conjugate(psi0),sigmaX(L,0).dot(psi0))
        obs_File.write('%.10f   %.10f\n'%(sZ,sX))
        # Compute <SzSz>
        SzSz = np.zeros((L,L))
        for i in range(L):
            for j in range(L):
                for k in range(1<<L):
                    config = (bin(k)[2:].zfill(L)).split()
                    SzSz[i,j] += (psi0[k]**2)*(2*float(config[0][i])-1)*(2*float(config[0][j])-1) 
                corr_File.write('%.10f   ' % SzSz[i,j])
            corr_File.write('\n')
        
        # Generate the training datasets
        print('generating the dataset...')
        #buildDataset(L,B,psi0,Nsamples) 
        print('\n')


#---------------------------------------------------------------


if __name__ == "__main__":
   
    # Read arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-L',type=int,default=10)
    parser.add_argument('-B',type=float)
    parser.add_argument('-J',type=float,default=1.0)
    par = parser.parse_args()

    # RUN
    main(par)

