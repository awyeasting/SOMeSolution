# SOMeSolution
An iteratively developed approach to the problem of fast training of Self Organizing Maps. This is a working implementation of the HPSOM algorithm described by Liu et al. This implementation can be run on:
- Serial architecture (through the `batch-som` branch with `make buildserial`)
- Shared memory architecture (using OpenMP through the `batch-som` branch with the `OMP_NUM_THREADS` environment variable)
- Distributed memory architecture (using OpenMPI through the `mpi` branch)
- Nvidia GPU architecture with shared memory (using CUDA and OpenMP through the `cuda` branch)
- Distributed memory Nvidia GPU architecture with shared memory (using OpenMPI, CUDA, and OpenMP through the `mpicuda` branch)

## C++ install

First clone the repository and checkout the branch of the version you want to use (`batch-som`, `mpi`, `cuda`, `mpicuda`)
```bash
git clone https://github.com/awyeasting/SOMeSolution.git
cd SOMeSolution
git checkout mpicuda
```

Then compile into either a library or an executable. (NOTE: if you installed CUDA in a different location or with a different version than 11.2 you will need to change the install location at the top of the makefile)

To compile the code to a library,
```bash
cd SOMeSolution/src/C++
make
```
The static library will be in `SOMeSolution/src/C++/bin/somesolution.a`

To compile the code to a commandline usable executable,
```bash
cd SOMeSolution/src/C++
make build
```
The executable will be in `SOMeSolution/src/C++/bin`

## Commandline Usage

Through the command line you can add different flags and optional arguments.

Arguments:
```bash
Positional Arguments:
	(int)    SOM width
	(int)    SOM height
	(string) Training data file name
Options:
	(int int)-g --generate       num features, num_dimensions for generating random data
	(string) -o --out            Path of the output file of node weights
	(int)    -e --epochs         Number of epochs used in training
	(int)    -s --seed           Integer value to intialize seed for generating
	         -l --labeled        Indicates the last column is a label
	(int)    -gp --gpus-per-proc The number of gpus each processor should utilize
```

Example:
The following will make a 10 x 10 SOM on 2 processes, generate its own training data (which has 100 examples and 100 dimensions), train the SOM on it, and output the trained map to `trained_map.txt`.
```bash
mpirun -np 2 bin/somwork_mpicuda 10 10 -g 100 100 -o trained_map.txt
```

## Python Visualization

To visualize a SOM weights file produced by the commandline executable, simply run:
```bash
python som.py -i weights.txt -d <display method>
```
(See python som.py -h for supported display methods)

## License 
[3-Clause BSD](https://github.com/awyeasting/SOMeSolution/blob/master/LICENSE)
