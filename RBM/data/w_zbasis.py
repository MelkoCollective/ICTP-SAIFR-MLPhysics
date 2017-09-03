import numpy as np
import sys


if(len(sys.argv)!=3):
    print "Insert: Nqubits Nsamples"
    exit()

#number of qubits
N=int(sys.argv[1])

#number of wanted samples
Nsamples=int(sys.argv[2])

vec=np.zeros((Nsamples,N),dtype=int)
for conf in range(Nsamples):
    n=np.random.randint(N)
    vec[conf,n]=1

np.savetxt('train.txt',vec,fmt='%.1i' )  
#wcoeff=np.zeros(2**N)

#setting all non-zero coefficients in the Z basis
#for i in range(N):
#    wcoeff[1<<i]=1./np.sqrt(float(N))

#confs=np.random.choice(2**N,Nsamples, p=wcoeff**2.)
#when printing resampled configurations, the last index is
#1 if Psi>0 and 0 if the Psi is negative

#for conf in confs:
#    s=str(bin(conf)[2:].zfill(N))
#    print s


