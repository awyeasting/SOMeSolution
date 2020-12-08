/*
 * This file is part of SOMeSolution.
 *
 * Developed for Pacific Northwest National Laboratory.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the BSD 3-Clause License as published by
 * the Software Package Data Exchange.
 */

#include "SOM.h"

//----------------------------------------------------
//	CUDA KERNEL FUNCTIONS
//----------------------------------------------------

/*
	CUDA kernel function for performing elementwise multiplication on two matrices</summary>
*/
__global__
void elementMul(double *A, double *B, double *C, int n) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < n)
		C[i] = A[i] * B[i];
}

/*
	CUDA kernel function for filling a matrix with ones
*/
__global__
void fillOnes(double *A, int n) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < n)
		A[i] = 1.0f;
}

/*
	CUDA kernel function calculating the BMUs of nodes as found by distances in the D matrix
*/
__global__
void findBMUsGPU(double *D, int *BMUs, int xdn, int nnodes) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < xdn) {
		// TODO: Optimize this further to utilize more processors
		BMUs[i] = 0;
		for (int j = 1; j < nnodes; j++) {
			// Uses column major order
			if (D[BMUs[i] * xdn + i] > D[j * xdn + i])
				BMUs[i] = j;
		}
	}
}

/*
	CUDA kernel function for calculating the gaussian value as described in the paper by Liu et. al.
*/
__global__
void calcGaussian(double *H, int xdn, int nnodes, double initial_map_radius, double neighborhood_radius, int *BMUs, int height) {
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if ((row < xdn) && (col < nnodes)) {
		int col_y = col % height;
		int col_x = (col - col_y) / height;

		// Get BMU coord
		int row_y = BMUs[row] % height;
		int row_x = (BMUs[row] - row_y) / height;

		H[col*xdn + row] = initial_map_radius * exp(-(double)((row_x - col_x) * (row_x - col_x) + (row_y - col_y) * (row_y - col_y))/(neighborhood_radius * neighborhood_radius));
	}
}

/*
	CUDA kernel function for copying a matrix from row major order to column major order
*/
__global__
void rowToColumnMajor(double *idata, double *odata, int nrows, int ncols, int n) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < n)
		odata[(i%ncols)*nrows + (i/ncols)] = idata[i];
}

//----------------------------------------------------
//	SOM non-member functions
//----------------------------------------------------

void trainOneEpoch(cublasHandle_t &handle, int device, double *train, double *weights, double *numer, double *denom, int map_size, int height, int num_examples, int dimensions, double initial_map_radius, double neighborhood_radius) {

	// Set assigned gpu
	gpuErrchk(cudaSetDevice(device));

	// Find BMUs for every input instance
	// D = X_sq - 2X^TM + M_sq
	// D (xdn * nn)
	
	// Calc m_sq
	// Elementwise multiply M by M
	double *d_msq;
	gpuErrchk(cudaMalloc(&d_msq, map_size * dimensions * sizeof(double)));
	int NUM_THREADS = 256;
	int NUM_BLOCKS = (int) ceil((float)(map_size*dimensions)/NUM_THREADS);
	elementMul<<<NUM_BLOCKS, NUM_THREADS>>>(weights, weights, d_msq, map_size * dimensions);
	// Left multiply elementwise multiplied M by all ones matrix (of dim num examples x dimensions)
	double *d_o;
	gpuErrchk(cudaMalloc(&d_o, num_examples * dimensions * sizeof(double)));
	NUM_BLOCKS = (int) ceil((float)(num_examples * dimensions)/NUM_THREADS);
	fillOnes<<<NUM_BLOCKS,NUM_THREADS>>>(d_o, num_examples * dimensions);
	gpuErrchk(cudaDeviceSynchronize());
	// m_sq = ones x (M * M)^T
	const double alpha0 = 1.0f;
	const double beta0 = 0.0f;
	double *m_sq;
	gpuErrchk(cudaMalloc(&m_sq, num_examples * map_size * sizeof(double)));
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, num_examples, map_size, dimensions, &alpha0, d_o, num_examples, d_msq, map_size, &beta0, m_sq, num_examples);
	
	gpuErrchk(cudaDeviceSynchronize());
	
	gpuErrchk(cudaFree(d_msq));
	gpuErrchk(cudaFree(d_o));

	// Calc x_sq
	// Elementwise multiply X by X
	double *d_xsq;
	gpuErrchk(cudaMalloc(&d_xsq, num_examples * dimensions * sizeof(double)));
	NUM_BLOCKS = (int) ceil((float)(num_examples*dimensions)/NUM_THREADS);
	elementMul<<<NUM_BLOCKS, NUM_THREADS>>>(train, train, d_xsq, num_examples * dimensions);
	gpuErrchk(cudaDeviceSynchronize());
	// Left multiply elementwise multiplied X by all ones matrix (of dim num examples x dimensions)
	gpuErrchk(cudaMalloc(&d_o, dimensions * map_size * sizeof(double)));
	NUM_BLOCKS = (int) ceil((float)(dimensions * map_size)/NUM_THREADS);
	fillOnes<<<NUM_BLOCKS,NUM_THREADS>>>(d_o, dimensions * map_size);
	// x_sq = (X * X) x ones
	double *x_sq;
	gpuErrchk(cudaMalloc(&x_sq, num_examples * map_size * sizeof(double)));
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_examples, map_size, dimensions, &alpha0, d_xsq, num_examples, d_o, dimensions, &beta0, x_sq, num_examples);
	
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaFree(d_xsq));
	gpuErrchk(cudaFree(d_o));

	// Calc D
	// From paper: 
	// D = x_sq - 2 * x^t * m + m_sq

	const double alpha1 = -2.0f;
	const double beta1 = 1.0f;

	gpuErrchk(cudaDeviceSynchronize());

	// m_sq = - 2 * (x^t * m) + (m_sq)
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, num_examples, map_size, dimensions, &alpha1, train, num_examples, weights, map_size, &beta1, m_sq, num_examples);

	gpuErrchk(cudaDeviceSynchronize());

	// D = (x_sq) + (-2 * x^t * m + m_sq)
	double *D;
	gpuErrchk(cudaMalloc(&D, num_examples * map_size * sizeof(double)));
	cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_examples, map_size, &alpha0, x_sq, num_examples, &beta1, m_sq, num_examples, D, num_examples);

	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaFree(m_sq));
	gpuErrchk(cudaFree(x_sq));

	// BMU index of each training instance
	int *BMUs;
	gpuErrchk(cudaMalloc(&BMUs, num_examples * sizeof(int)));
	NUM_BLOCKS = (int) ceil((float)(num_examples)/NUM_THREADS);
	findBMUsGPU<<<NUM_BLOCKS, NUM_THREADS>>>(D, BMUs, num_examples, map_size);

	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaFree(D));

	// Calc gaussian function 
	// (num_examples x num nodes)
	int BLOCK_SIZE = 16;
	int GRID_HEIGHT = (int)ceil((float)num_examples/BLOCK_SIZE);
	int GRID_WIDTH = (int)ceil((float)map_size/BLOCK_SIZE);
	dim3 grid(GRID_WIDTH, GRID_HEIGHT);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	double *H;
	gpuErrchk(cudaMalloc(&H, num_examples * map_size * sizeof(double)));
	calcGaussian<<<grid, threads>>>(H, num_examples, map_size, initial_map_radius, neighborhood_radius, BMUs, height);
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaFree(BMUs));

	// Calc denominators
	// Left multiply H by a num_examples dimensional vector of ones
	gpuErrchk(cudaMalloc(&d_o, num_examples * sizeof(double)));
	NUM_BLOCKS = (int)ceil((float)num_examples/NUM_THREADS);
	
	fillOnes<<<NUM_BLOCKS,NUM_THREADS>>>(d_o, num_examples);
	gpuErrchk(cudaDeviceSynchronize());
	// denom = ones^T (1 x num examples) * H (num examples x map size)
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, map_size, num_examples, &alpha0, d_o, 1, H, num_examples, &beta0, denom, 1);
	
	gpuErrchk(cudaDeviceSynchronize());
	
	gpuErrchk(cudaFree(d_o));
	
	// Calc numerators
	// numer = H^T x X
	cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, map_size, dimensions, num_examples, &alpha0, H, num_examples, train, num_examples, &beta0, numer, map_size);

	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaFree(H));
}

//----------------------------------------------------
//	public SOM functions
//----------------------------------------------------

/* 
	Construct untrained SOM with given lattice width and height
*/
SOM::SOM(unsigned int width, unsigned int height)
{
	MPI_Comm_rank(MPI_COMM_WORLD, &this->_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &this->_numProcs);
	//this->_rank = MPI::COMM_WORLD.Get_rank();
	//this->_numProcs = MPI::COMM_WORLD.Get_size();

	this->_width = width;
	this->_height = height;
}

/*
	Construct SOM from a saved SOM width, height, and set of weights
*/
SOM::SOM(std::istream &in) {
	MPI_Comm_rank(MPI_COMM_WORLD, &this->_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &this->_numProcs);
	//this->_rank = MPI::COMM_WORLD.Get_rank();
	//this->_numProcs = MPI::COMM_WORLD.Get_size();

	this->loadWeights(in);
}

/*
	Generates a random set of training data if there is no input file given
*/
void SOM::gen_train_data(unsigned int num_examples, unsigned int dimensions, unsigned int seedValue)
{
	if (_trainData != NULL) {
		std::cout << "WARNING: train data not initialized" << std::endl;
	}
	this->_dimensions = dimensions;
	// TODO: Switch to compute based examples distribution
	this->_numExamples = num_examples / this->_numProcs;
	this->_trainData = new double [this->_numExamples * this->_dimensions];
	srand(seedValue + this->_rank);
	for (int i = 0; i < this->_numExamples; i++)
	{
		int rowMod = (this->_numExamples - i - 1) * this->_dimensions;
		for (int d = 0; d < this->_dimensions; d++)
		{
			double weight = SOM::randWeight();
			this->_trainData[rowMod + d] = weight;
		}
	}
}

/*
	Load a set of training data from a given filename

	Precondition: File is already open
*/
bool SOM::load_train_data(std::string fileName, bool hasLabelRow, bool hasLabelColumn) {
	unsigned int cols = 0, rows = 0;
	bool okOpen = true;
	if (this->_rank == 0) {
		// Open file for counting number of rows and columns
		std::ifstream infile(fileName, std::ifstream::in);
		if (!infile.is_open()) {
			okOpen = false;
			std::cout << "Invalid training data file '" << fileName << "'" << std::endl;
		} else {
			// Read in first row of data into line
			std::string line;
			if (hasLabelRow) {
				std::getline(infile, line);
				rows++;
			}
			std::getline(infile, line);
			rows++;

			// Count number of values in the first row to determine num columns
			// TODO: make this work with non number labels
			std::stringstream ss(line);
			double temp;
			while (ss >> temp) {
				cols++;
			}
			while(std::getline(infile, line)) {
				// Ignore empty lines to be more forgiving of minor formatting mistakes
				if (line.compare("") != 0)
					rows++;
			}
		}
		infile.close();
	}
	// Check there was no problem with the file
	MPI_Bcast(&okOpen, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
	if (!okOpen) {
		return false;
	}

	// Broadcast rows and columns
	MPI_Bcast(&rows, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	MPI_Bcast(&cols, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	this->_numExamples = rows - ((unsigned int)hasLabelRow);
	this->_dimensions = cols - ((unsigned int)hasLabelColumn);
	
	// Initialize feature maximums and minimums
	this->_featureMaxes = (double *)malloc(sizeof(double) * this->_dimensions);
	this->_featureMins = (double*)malloc(sizeof(double) * this->_dimensions);
	for(int i =0; i < this->_dimensions; i++){
		_featureMaxes[i] = std::numeric_limits<double>::min();
		_featureMins[i] = std::numeric_limits<double>::max();
	}

	// Calculate starting position
	int read_count = this->_numExamples / this->_numProcs;
	int startRow = ((this->_numExamples / this->_numProcs) * this->_rank);
	// Adjust for remainder of examples
	if (this->_rank == this->_numProcs - 1) {
		read_count += (this->_numExamples % this->_numProcs);
	}
	this->_numExamples = read_count;

	// Prepare for reading in assigned chunk
	bool readOk = true;
	std::fstream infile(fileName, std::ifstream::in); // Assume because it opened for rank 0 it will open for all
	std::fstream& procfile = GotoLine(infile, startRow);
	this->_trainData = (double *)malloc(this->_numExamples * this->_dimensions * sizeof(double));

	// Read in assigned portion
	int procSectionLineNum = 0;
	std::string line;
	while(procSectionLineNum < read_count && std::getline(procfile, line)) {
		if (line.compare("") != 0) {
			std::stringstream ss(line);
			double temp;
			int cols_count = 0;
			// Read line into train data
			while (ss >> temp && cols_count < this->_dimensions) {
				this->_trainData[procSectionLineNum * this->_dimensions + cols_count] = temp;
				if (temp > this->_featureMaxes[cols_count]) {
					this->_featureMaxes[cols_count] = temp;
				}
				if (temp < this->_featureMins[cols_count]) {
					this->_featureMins[cols_count] = temp;
				}
				cols_count++;
			}
			// If the line finished reading early then the data is not of a consistent dimension
			if (cols_count != this->_dimensions - 1) {
				readOk = false;
			}
			procSectionLineNum++;
		}
		if (!readOk)
			break;
	}
	// If it didn't read enough lines then the data is not properly formatted
	if (procSectionLineNum != read_count - 1) {
		readOk = false;
		return false;
	}

	// Check that all the threads read their data properly
	bool allReadOk;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(&readOk, &allReadOk, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
	// If any process failed to read the process report it and discharge any allocated memory
	if (!allReadOk) {
		destroy_train_data();
		std::cout << "Error reading input file: Unable to read input file, check to make sure the input is properly formatted" << std::endl;
		return false;
	}

	// Find the true feature maxes and true feature mins
	double *globalMaxes = (double *)malloc(sizeof(double) * this->_dimensions);
	double *globalMins = (double*)malloc(sizeof(double) * this->_dimensions);
	MPI_Allreduce(this->_featureMaxes, globalMaxes, this->_dimensions, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(this->_featureMins, globalMins, this->_dimensions, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
		
	// Free pre reduction maxes and mins
	free(this->_featureMaxes);
	free(this->_featureMins);
	this->_featureMaxes = globalMaxes;
	this->_featureMins = globalMins;

	return true;
}

void SOM::destroy_train_data() {
	free(this->_trainData);
	free(this->_featureMaxes);
	free(this->_featureMins);

	this->_trainData = NULL;
	this->_featureMaxes = NULL;
	this->_featureMins = NULL;
}

/*
	Train the SOM using a set of training data over a given number of epochs with a given learning rate
*/
void SOM::train_data(unsigned int epochs, double initial_learning_rate, unsigned int map_seed)
{
	// Check that the training data has been loaded in
	if (this->_trainData == NULL) {
		std::cout << "Train data not yet initialized in SOM" << std::endl;
		return;
	}

	this->_mapSize = this->_width * this->_height;

	cublasHandle_t* handles;
	double neighborhood_radius, *numer, *denom, **d_train, **d_weights, **d_numer, **d_denom, **gnumer, **gdenom;
	int NUM_GPUS, *GPU_EXAMPLES, *GPU_OFFSET;

	// Establish multi gpu setup on current node
	// TODO: Add num gpus option
	cudaGetDeviceCount(&NUM_GPUS);
	omp_set_dynamic(0); // Disable dynamic teams
	omp_set_num_threads(NUM_GPUS);

	// Allocate memory associated with training on each GPU on each node
	initNumDenom(numer, denom);
	initGPUTrainMemory(NUM_GPUS, handles, d_train, d_weights, d_numer, d_denom, GPU_EXAMPLES, GPU_OFFSET, this->_numExamples);
	initGPUNumDenReducMem(NUM_GPUS, gnumer, gdenom);
	normalizeData(this->_trainData);
	// Split training data onto gpus on each node
	initGPUTrainData(NUM_GPUS, this->_trainData, d_train, GPU_EXAMPLES, GPU_OFFSET);
	
	// TODO: verify that global_numer and denom only need to be allocated on rank 0
	double* global_numer;
	double* global_denom;
	// Init codebook on first node
	if (this->_rank == 0) {
		srand(map_seed);
		if (GPU_BASED_CODEBOOK_INIT)
			initCodebookOnGPU(d_weights);
		else
			initCodebook();

		// Init global numerators and denominators for reduction to node 0
		global_numer = (double*)malloc(_width * _height * _dimensions*sizeof(double));
		global_denom = (double *)malloc(_width * _height * sizeof(double));
	}

	double initial_map_radius = _width < _height ? ((double)_width) / 2.0 : ((double)_height) / 2.0;
	double time_constant = double(epochs) / log(initial_map_radius);
	
	for(int epoch = 0; epoch < epochs; epoch++) {
		// Wait for all other nodes to start the epoch
		MPI_Barrier(MPI_COMM_WORLD);

		// Send out the map on proc 0
		MPI_Bcast(this->_weights, this->_width * this->_height * this->_dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Update gpu copies of the map
		setGPUCodebooks(d_weights);

		// Calculate current neighborhood radius
		neighborhood_radius = initial_map_radius * exp(-((double)(epoch))/time_constant);
		// Train a single epoch on all gpus
		#pragma omp parallel
		{
			int gpu = omp_get_thread_num();
			gpuErrchk(cudaSetDevice(gpu));
			gpuErrchk(cudaDeviceSynchronize());
			trainOneEpoch(handles[gpu], gpu, d_train[gpu], d_weights[gpu], d_numer[gpu], d_denom[gpu], this->_mapSize, this->_height, GPU_EXAMPLES[gpu], this->_dimensions, initial_map_radius, neighborhood_radius);
			gpuErrchk(cudaMemcpy(gnumer[gpu],d_numer[gpu], this->_mapSize * this->_dimensions * sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(gdenom[gpu],d_denom[gpu], this->_mapSize * sizeof(double), cudaMemcpyDeviceToHost));
		}

		// Reduce numerators and denominators across gpus on proc
		// TODO: Implement more complex reduction
		for(int gpu = 0; gpu < NUM_GPUS; gpu++) {
			if (gpu == 0) {
				for (int i = 0; i < this->_mapSize; i++) {
					denom[i] = gdenom[gpu][i];
					for (int d = 0; d < this->_dimensions; d++) {
						numer[d*this->_mapSize + i] = gnumer[gpu][d*this->_mapSize + i];
					}
				}
			} else {
				for (int i = 0; i < this->_mapSize; i++) {
					denom[i] += gdenom[gpu][i];
					for (int d = 0; d < this->_dimensions; d++) {
						numer[d*this->_mapSize + i] += gnumer[gpu][d*this->_mapSize + i];
					}
				}
			}
		}

		// Reduce numerators and denominators across all procs
		MPI_Reduce(numer, global_numer, this->_mapSize * this->_dimensions, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(numer, global_numer, this->_mapSize, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		
		// Update codebook/map
		if (this->_rank == 0) {
			// Recalculate weights with new numerators and denominators
			for (int i = 0; i < this->_mapSize; i++) {
				for (int d = 0; d < this->_dimensions; d++) {
					this->_weights[d*this->_mapSize + i] = numer[d*this->_mapSize + i] / denom[i];
				}
			}
		}
	}

	// Perform column major to row major order on weights matrix
	double *tempWeights = (double *)malloc(this->_mapSize * this->_dimensions * sizeof(double));
	for (int i = 0; i < this->_mapSize; i++) {
		for (int d = 0; d < this->_dimensions; d++) {
			tempWeights[i*this->_dimensions + d] = this->_weights[d*this->_mapSize + i];
		}
	}
	free(this->_weights);
	this->_weights = tempWeights;

	for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
		cudaSetDevice(gpu);
		cublasDestroy(handles[gpu]);
		cudaFree(d_train[gpu]);
		cudaFree(d_weights[gpu]);
		cudaFree(d_numer[gpu]);
		cudaFree(d_denom[gpu]);
		free(gnumer[gpu]);
		free(gdenom[gpu]);
	}

	free(GPU_EXAMPLES);
	free(GPU_OFFSET);
	free(handles);
	free(d_train);
	free(d_weights);
	free(d_numer);
	free(d_denom);
	free(gnumer);
	free(gdenom);
	free(numer);
	free(denom);
	free(global_numer);
	free(global_denom);
}

/*
	Save the width and height of the SOM followed by the weights for each node with a different node's weights on every line
*/
void SOM::save_weights(std::ostream &out)
{
	out << this->_width << " " << this->_height << std::endl;
	for (int i = 0; i < this->_width; i++)
	{
		for (int j = 0; j < this->_height; j++)
		{
			for (int k = 0; k < this->_dimensions; k++) {
				if (k != 0) {
					out << " ";
				}
				out << this->_weights[calcIndex(i,j,k)];
			}
			out << std::endl;
		}
	}
}

std::fstream& SOM::GotoLine(std::fstream& file, unsigned int num){
    file.seekg(std::ios::beg);
    for(int i=0; i < num - 1; ++i){
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    }
    return file;
}

void SOM::printDoubles(double *doubleList, unsigned int numDoubles, unsigned int numLines)
{
	unsigned int numPerLine = numDoubles/numLines;
	unsigned int counter = 0;
	while(counter < numDoubles)
	{
		for (int j = 0; j< numPerLine; j++)
		{
			std::cout << doubleList[counter] << " ";
			counter++;
		}
		std::cout << std::endl;
	}
}

//----------------------------------------------------
//	private SOM functions
//----------------------------------------------------

/*
	Load a trained SOM that was saved using the same algorithm as save_weights from an input stream
*/
void SOM::loadWeights(std::istream &in)
{
	// Load SOM dimensions first
	in >> this->_width >> this->_height;

	// Read first line of matrix to get the dimensionality of weights
	this->_dimensions = 0;
	std::string line;
	std::getline(in, line);
	std::getline(in, line);
	std::stringstream ss(line);
	std::vector<double> line1;
	double temp;
	while (ss >> temp) {
		this->_dimensions++;
		line1.push_back(temp);
	}

	// Put first line of matrix into an array in the 3d weights array
	this->_weights = new double[_width * _height * _dimensions];
	for (int k = 0; k < this->_dimensions; k++) {
		_weights[calcIndex(0,0,_dimensions - k - 1)] = line1.back();
		line1.pop_back();
	}

	// Read the rest of the 3d array in
	for (int i = 0; i < this->_width; i++) {
		for (int j = (i == 0 ? 1 : 0); j < this->_height; j++) {
			for (int k = 0; k < _dimensions; k++) {
				in >> this->_weights[calcIndex(i,j,k)];
			}
		}
	}
}

/*
	Normalizes given data to be between 0 and 1 for each feature
*/
void SOM::normalizeData(double *trainData)
{
	// Find the max and min value for each feature then use it to normalize the feature
	this->_featureMaxes = new double[this->_dimensions];
	this->_featureMins = new double[this->_dimensions];
	for (int d = 0; d < this->_dimensions; d++)
	{
		this->_featureMaxes[d] = -std::numeric_limits<double>::max();
		this->_featureMins[d] = std::numeric_limits<double>::max();
		for (int i = 0; i < this->_numExamples; i++)
		{
			if (trainData[i*this->_dimensions + d] > this->_featureMaxes[d]) {
				this->_featureMaxes[d] = trainData[i*_dimensions + d];
			}
			if (trainData[i*this->_dimensions + d] < this->_featureMins[d]) {
				this->_featureMins[d] = trainData[i*this->_dimensions + d];
			}
		}
		for (int i = 0; i < this->_numExamples; i++) {
			if ((this->_featureMaxes[d] - this->_featureMins[d]) <= std::numeric_limits<double>::min())
			{
				trainData[i*_dimensions + d] = 0;
			}
			else {
				trainData[i*_dimensions + d] = (trainData[i*_dimensions + d] - this->_featureMins[d])/(this->_featureMaxes[d]-this->_featureMins[d]);
			}
		}
	}
}

/*
	Calculate the index of a weight at node (x,y), dimension = d in the weights array
*/
int SOM::calcIndex(int x, int y, int d) {
	return (x*_height + y)*_dimensions + d;
}

void SOM::initMultiGPUSetup(int &ngpus) {
	
}

void SOM::initNumDenom(double *&numer, double *&denom) {
	numer = (double *)malloc(this->_mapSize * this->_dimensions * sizeof(double));
	denom = (double *)malloc(this->_mapSize * sizeof(double));
	for (int i = 0; i < this->_mapSize; i++) {
		denom[i] = 0.0;
		for (int j = 0; j < this->_dimensions; j++) {
			numer[i*this->_dimensions + j] = 0.0;
		}
	}
}

void SOM::initGPUTrainData(const int ngpus, double *trainData, double **d_train, int *GPU_EXAMPLES, int *GPU_OFFSET) {
	#pragma omp parallel
	{
		int NUM_BLOCKS;
		int NUM_THREADS = 256;
		int gpu = omp_get_thread_num();

		NUM_BLOCKS = (int)ceil((float)(GPU_EXAMPLES[gpu] * this->_dimensions)/NUM_THREADS);
		double *temp_d_train;

		gpuErrchk(cudaSetDevice(gpu));
		gpuErrchk(cudaMalloc(&temp_d_train, GPU_EXAMPLES[gpu] * this->_dimensions * sizeof(double)));
		gpuErrchk(cudaMemcpy(temp_d_train, &trainData[GPU_OFFSET[gpu]], GPU_EXAMPLES[gpu] * this->_dimensions * sizeof(double), cudaMemcpyHostToDevice));
		// Convert data from row major order to 
		rowToColumnMajor<<<NUM_BLOCKS, NUM_THREADS>>>(temp_d_train, d_train[gpu], GPU_EXAMPLES[gpu], this->_dimensions, GPU_EXAMPLES[gpu] * this->_dimensions);
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaFree(temp_d_train));
	}
}

void SOM::initGPUTrainMemory(const int ngpus, cublasHandle_t *&handles, double **&d_train, double **&d_weights, double **&d_numer, double **&d_denom, int *&GPU_EXAMPLES, int *&GPU_OFFSET, int num_examples) {
	handles = (cublasHandle_t *)malloc(ngpus * sizeof(cublasHandle_t));
	d_train = (double **)malloc(ngpus * sizeof(double *));
	d_weights = (double **)malloc(ngpus * sizeof(double *));
	d_numer = (double **)malloc(ngpus * sizeof(double *));
	d_denom = (double **)malloc(ngpus * sizeof(double *));
	GPU_EXAMPLES = (int *)malloc(ngpus * sizeof(int));
	GPU_OFFSET = (int *)malloc(ngpus * sizeof(int));
	GPU_OFFSET[0] = 0;

	for (int gpu = 0; gpu < ngpus; gpu++) {
		gpuErrchk(cudaSetDevice(gpu));
		// Create cublas handles associated with each device
		cublasCreate(&handles[gpu]);

		// Set the number of examples allocated to each GPU simply by equal division
		GPU_EXAMPLES[gpu] = num_examples/ngpus;
		if (gpu < ngpus-1)
			GPU_OFFSET[gpu+1] = GPU_OFFSET[gpu] + GPU_EXAMPLES[gpu];
		// Allocate remainder examples to last gpu
		else
			GPU_EXAMPLES[gpu] += num_examples - (GPU_OFFSET[gpu] + GPU_EXAMPLES[gpu]);
		
		// Allocate space for current GPU's share of the examples
		gpuErrchk(cudaMalloc(&d_train[gpu], GPU_EXAMPLES[gpu] * this->_dimensions * sizeof(double)));
		// Allocate space for current GPU's copy of the map
		gpuErrchk(cudaMalloc(&d_weights[gpu], this->_mapSize * this->_dimensions * sizeof(double)));
		// Allocate space for current GPU's copy of numerators and denominators
		gpuErrchk(cudaMalloc(&d_numer[gpu], this->_mapSize * this->_dimensions * sizeof(double)));
		gpuErrchk(cudaMalloc(&d_denom[gpu], this->_mapSize * sizeof(double)));
	}
}

void SOM::initGPUNumDenReducMem(const int ngpus, double **&gnumer, double **&gdenom) {
	gnumer = (double **)malloc(ngpus * sizeof(double *));
	gdenom = (double **)malloc(ngpus * sizeof(double *));
	for (int gpu = 0; gpu < ngpus; gpu++) {
		gnumer[gpu] = (double *)malloc(this->_mapSize * this->_dimensions * sizeof(double));
		gdenom[gpu] = (double *)malloc(this->_mapSize * sizeof(double));
	}
}

void SOM::initCodebook() {
	this->_weights = (double *)malloc(this->_mapSize * this->_dimensions * sizeof(double));
	for (int i = 0; i < this->_mapSize; i++) {
		for (int d = 0; d < this->_dimensions; d++) {
			this->_weights[i * this->_dimensions + d] = this->randWeight();
		}
	}
}

void SOM::initCodebookOnGPU(double **d_weights) {
	const int CODEBOOK_INIT_DEVICE = 0;
	gpuErrchk(cudaSetDevice(CODEBOOK_INIT_DEVICE));
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	cudaDeviceSynchronize();
	// TODO: curandSetPseudoRandomGeneratorSeed(gen, );
	curandGenerateUniformDouble(gen, d_weights[CODEBOOK_INIT_DEVICE], this->_mapSize * this->_dimensions);
	
	// Copy map from gpu to cpu
	this->_weights = (double *)malloc(this->_mapSize * this->_dimensions * sizeof(double));
	gpuErrchk(cudaMemcpy(this->_weights, d_weights[CODEBOOK_INIT_DEVICE], this->_mapSize * this->_dimensions * sizeof(double), cudaMemcpyDeviceToHost));
}

void SOM::setGPUCodebooks(double **d_weights) {
	#pragma omp parallel
	{
		int gpu = omp_get_thread_num();

		gpuErrchk(cudaSetDevice(gpu));
		gpuErrchk(cudaMemcpy(d_weights[gpu], this->_weights, this->_mapSize * this->_dimensions * sizeof(double), cudaMemcpyHostToDevice));
	}
}

double SOM::randWeight()
{
	return (double)rand() / (RAND_MAX);
}