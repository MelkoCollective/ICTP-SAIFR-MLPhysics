# Restricted Boltzmann Machine
This module describes how to employ a [restricted Boltzmann machine][1] (rbm) to learn different physical distributions from raw data, obtained by performing measurements on the systems.
Contents of the folder:
* `rbm.py`: rbm class
* `main.py`: main script
* `data/`: folder containing the data of the 2d classical Ising model and the 1d transverse field quantum Ising model
* `tutorial/`: folder containing the scripts for the tutorial

Usage:
```
$ python main.py [COMMAND] [ARGUMENTS]
```

Command:
* `train`: train the rbm
* `sample`: sample the rbm given a set of trained parameters

Arguments:
* `-nV`: number of visible units
* `-nH`: number of hidden units
* `-lr`: learning rate
* `-CD`: number of Gibbs updates in the contrastive divergence algorithm
* `-bs`: batch size
* `-step`: number of training steps
* `-nC`: number of chains sampled in contrastive divergence

Useful references:
Training of RBMs: https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf

[1]: https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine "rbm"