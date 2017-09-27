import ising2d
import sampler
import argparse

# Initialize the command line parser
parser = argparse.ArgumentParser()

# Read command line arguments
parser.add_argument('command',type=str,help='command to execute')
parser.add_argument('-L',type=int,help='linear size of the system')                
parser.add_argument('-T',type=float,default=1.0,help='Temperature')              
parser.add_argument('-B',type=float,default=1.0,help='Magnetic field)')              
parser.add_argument('-nH',type=int,help='number of hidden nodes')   
parser.add_argument('-steps',type=int,default=1000000,help='training steps')  
parser.add_argument('-lr',type=float,default=1e-3,help='learning rate')   
parser.add_argument('-bs',type=int,default=100,help='batch size')   
parser.add_argument('-CD',type=int,default=10,help='steps of contrastive divergence') 
parser.add_argument('-nC',type=float,default=10,help='number of chains in PCD')  

# Parse the arguments
args = parser.parse_args()

if args.command == 'train':
  ising2d.train(args)

if args.command == 'sample':
  ising2d.sample(args)
    
if args.command == 'sample_all':
  sampler.ising2d()

