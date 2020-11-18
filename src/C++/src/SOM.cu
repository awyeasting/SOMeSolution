#include "SOM.h"

// Kernel function to perform elementwise multiplication
__global__
void elementMul(double *A, double *B, double *C, int n) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < n)
		C[i] = A[i] * B[i];
}

__global__
void fillOnes(double *A, int n) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < n)
		A[i] = 1.0f;
}

__global__
void findBMUsGPU(double *D, int *BMUs, int xdn, int nnodes) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < xdn) {
		// TODO: Optimize this further to utilize more processors
		BMUs[i] = 0;
		for (int j = 1; j < nnodes; j++) {
			// Uses column major order
			if (BMUs[i] > D[j*xdn + i])
				BMUs[i] = j;
		}
	}
}

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

double h(int j, int i, double initial_radius, double radius, int* BMUs, int height) {
	int i_y = i % height;
	int i_x = (i - i_y) / height;

	// Get BMU coord
	int j_y = BMUs[j] % height;
	int j_x = (BMUs[j] - j_y) / height;

	return initial_radius * exp(-(double)((j_x - i_x) * (j_x - i_x) + (j_y - i_y) * (j_y - i_y))/(radius * radius));
}

void trainOneEpoch(cublasHandle_t &handle, double *train, double *weights, double *numer, double *denom, int map_size, int height, int num_examples, int dimensions, double initial_map_radius, double neighborhood_radius) {

	// Find BMUs for every input instance
	// D = X_sq - 2X^TM + M_sq
	// D (xdn * nn)
	
	// Calc m_sq
	// Elementwise multiply M by M
	double *d_msq;
	cudaMalloc(&d_msq, map_size * dimensions * sizeof(double));
	int NUM_THREADS = 256;
	int NUM_BLOCKS = (int) ceil((float)(map_size*dimensions)/NUM_THREADS);
	elementMul<<<NUM_BLOCKS, NUM_THREADS>>>(weights, weights, d_msq, map_size * dimensions);
	// Left multiply elementwise multiplied M by all ones matrix (of dim num examples x dimensions)
	double *d_o;
	cudaMalloc(&d_o, num_examples * dimensions * sizeof(double));
	NUM_BLOCKS = (int) ceil((float)(num_examples * dimensions)/NUM_THREADS);
	fillOnes<<<NUM_BLOCKS,NUM_THREADS>>>(d_o, num_examples * dimensions);
	// m_sq = ones x (M * M)^T
	const double alpha0 = 1.0f;
	const double beta0 = 1.0f;
	double *m_sq;
	cudaMalloc(&m_sq, num_examples * map_size * sizeof(double));
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, num_examples, map_size, dimensions, &alpha0, d_o, num_examples, d_msq, map_size, &beta0, m_sq, num_examples);
	
	cudaDeviceSynchronize();
	
	cudaFree(d_msq);
	cudaFree(d_o);

	// Calc x_sq
	// Elementwise multiply X by X
	double *d_xsq;
	cudaMalloc(&d_xsq, num_examples * dimensions * sizeof(double));
	NUM_BLOCKS = (int) ceil((float)(num_examples*dimensions)/NUM_THREADS);
	elementMul<<<NUM_BLOCKS, NUM_THREADS>>>(train, train, d_xsq, num_examples * dimensions);
	// Left multiply elementwise multiplied X by all ones matrix (of dim num examples x dimensions)
	cudaMalloc(&d_o, dimensions * map_size * sizeof(double));
	NUM_BLOCKS = (int) ceil((float)(dimensions * map_size)/NUM_THREADS);
	fillOnes<<<NUM_BLOCKS,NUM_THREADS>>>(d_o, dimensions * map_size);
	// x_sq = (X * X) x ones
	double *x_sq;
	cudaMalloc(&x_sq, num_examples * map_size * sizeof(double));
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_examples, map_size, dimensions, &alpha0, d_xsq, num_examples, d_o, dimensions, &beta0, x_sq, num_examples);
	
	cudaDeviceSynchronize();

	cudaFree(d_xsq);
	cudaFree(d_o);

	// Calc D
	// From paper: 
	// D = x_sq - 2 * x^t * m + m_sq

	const double alpha1 = -2.0f;

	cudaDeviceSynchronize();

	// m_sq = - 2 * (x^t * m) + (m_sq)
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, num_examples, map_size, dimensions, &alpha1, train, num_examples, weights, map_size, &beta0, m_sq, num_examples);

	cudaDeviceSynchronize();

	// D = (x_sq) + (-2 * x^t * m + m_sq)
	double *D;
	cudaMalloc(&D, num_examples * map_size * sizeof(double));
	cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_examples, map_size, &alpha0, x_sq, num_examples, &beta0, m_sq, num_examples, D, num_examples);

	cudaDeviceSynchronize();

	cudaFree(m_sq);
	cudaFree(x_sq);

	// BMU index of each training instance
	int *BMUs;
	cudaMalloc(&BMUs, num_examples * sizeof(int));
	NUM_BLOCKS = (int) ceil((float)(num_examples)/NUM_THREADS);
	findBMUsGPU<<<NUM_BLOCKS, NUM_THREADS>>>(D, BMUs, num_examples, map_size);

	cudaDeviceSynchronize();

	cudaFree(D);

	// Calc gaussian function 
	// (num_examples x num nodes)
	int BLOCK_SIZE = 16;
	int GRID_HEIGHT = (int)ceil((float)num_examples/BLOCK_SIZE);
	int GRID_WIDTH = (int)ceil((float)map_size/BLOCK_SIZE);
	dim3 grid(GRID_WIDTH, GRID_HEIGHT);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	double *H;
	cudaMalloc(&H, num_examples * map_size * sizeof(double));
	calcGaussian<<<grid, threads>>>(H, num_examples, map_size, initial_map_radius, neighborhood_radius, BMUs, height);

	cudaDeviceSynchronize();

	cudaFree(BMUs);

	// Calc denominators
	// Left multiply H by a num_examples dimensional vector of ones
	cudaMalloc(&d_o, num_examples * sizeof(double));
	NUM_BLOCKS = (int)ceil((float)num_examples/NUM_THREADS);
	
	fillOnes<<<NUM_BLOCKS,NUM_THREADS>>>(d_o, num_examples);
	// denom = ones^T (1 x num examples) * H (num examples x map size)
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, map_size, num_examples, &alpha0, d_o, 1, H, num_examples, &beta0, denom, 1);
	
	cudaDeviceSynchronize();
	
	cudaFree(d_o);
	
	// Calc numerators
	// numer = H^T x X
	cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, map_size, dimensions, num_examples, &alpha0, H, num_examples, train, num_examples, &beta0, numer, map_size);
	//for (int i = 0; i < map_size; i++) {
	//	for (int d = 0; d < dimensions; d++) {
	//		numer[i * dimensions + d] = 0.0;
	//		for (int j = 0; j < num_examples; j++) {
	//			numer[i*dimensions + d] += H[j*map_size + i] * train[j*dimensions + d];
	//		}
	//	}
	//}

	cudaDeviceSynchronize();
	cudaFree(H);
}

/* 
	Construct untrained SOM with given lattice width and height
*/
SOM::SOM(unsigned int width, unsigned int height)
{
	this->_width = width;
	this->_height = height;
}

/*
	Construct SOM from a saved SOM width, height, and set of weights
*/
SOM::SOM(std::istream &in) {
	this->load_weights(in);
}

/*
	Train the SOM using a set of training data over a given number of epochs with a given learning rate
*/
void SOM::train_data(double *trainData, unsigned int num_examples, unsigned int dimensions, int epochs, double initial_learning_rate)
{
	cublasHandle_t handle;
	cublasCreate(&handle);

	this->_dimensions = dimensions;
	// Normalize data (to be within 0 to 1)
	normalizeData(trainData, num_examples);

	// Randomly initialize codebook
	cudaMallocManaged(&this->_weights, _width * _height * _dimensions * sizeof(double));
	for (int i = 0; i < _width; i++) {
		for (int j = 0; j < _height; j++) {
			for (int d = 0; d < _dimensions; d++) {
				this->_weights[calcIndex(i,j,d)] = randWeight();
			}
		}
	}

	// Calc initial map radius
	double initial_map_radius = _width < _height ? ((double)_width) / 2.0 : ((double)_height) / 2.0;
	double time_constant = double(epochs) / log(initial_map_radius);

	double *numerators, *denominators;
	double *d_train;

	cudaMallocManaged(&numerators, this->_width * this->_height * this->_dimensions * sizeof(double));
	cudaMallocManaged(&denominators, this->_width * this->_height * sizeof(double));

	cudaMalloc(&d_train, num_examples * this->_dimensions * sizeof(double));
	// Fix the data for column major ordering
	cudaMemcpy(trainData, d_train, num_examples * this->_dimensions * sizeof(double), cudaMemcpyHostToDevice);

	double neighborhood_radius;
	for(int epoch = 0; epoch < epochs; epoch++) {
		//learning_rate = initial_learning_rate * exp(-double(epoch)/time_constant);
		neighborhood_radius = initial_map_radius * exp(-double(epoch)/time_constant);

		//trainOneEpoch(train,weights, D, m_sq, x_sq, BMUs, H, numer, denom, map_size, height, int num_examples, int dimensions, double initial_map_radius, double neighborhood_radius)
		trainOneEpoch(handle, d_train, this->_weights, numerators, denominators, this->_width * this->_height, this->_height, num_examples, this->_dimensions, initial_map_radius, neighborhood_radius);

		// Update codebook
		for (int i = 0; i < this->_width * this->_height; i++) {
			for (int d = 0; d < dimensions; d++) {
				this->_weights[i*dimensions + d] = numerators[i*this->_dimensions + d]/denominators[i];
			}
		}
	}

	cudaFree(numerators);
	cudaFree(denominators);

	cublasDestroy(handle);
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

/*
	Load a trained SOM that was saved using the same algorithm as save_weights from an input stream
*/
void SOM::load_weights(std::istream &in)
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
void SOM::normalizeData(double *trainData, int num_examples)
{
	// Find the max and min value for each feature then use it to normalize the feature
	this->_featureMaxes = new double[this->_dimensions];
	this->_featureMins = new double[this->_dimensions];
	for (int d = 0; d < this->_dimensions; d++)
	{
		this->_featureMaxes[d] = -std::numeric_limits<double>::max();
		this->_featureMins[d] = std::numeric_limits<double>::max();
		for (int i = 0; i < num_examples; i++)
		{
			if (trainData[i*_dimensions + d] > this->_featureMaxes[d]) {
				this->_featureMaxes[d] = trainData[i*_dimensions + d];
			}
			if (trainData[i*_dimensions + d] < this->_featureMins[d]) {
				this->_featureMins[d] = trainData[i*_dimensions + d];
			}
		}
		for (int i = 0; i < num_examples; i++) {
			trainData[i*_dimensions + d] = (trainData[i*_dimensions + d] - this->_featureMins[d])/(this->_featureMaxes[d]-this->_featureMins[d]);
		}
	}
}

/*
	Update a node's weights to better match a given example
*/
void SOM::updateNodeWeights(int x, int y, double* example, double learning_rate, double influence) {
	for (int d = 0; d < this->_dimensions; d++)
	{
		this->_weights[calcIndex(x,y,d)] += influence * learning_rate * (example[d] - this->_weights[calcIndex(x,y,d)]);
	}
}

/*
	Generate a vector of size numFeatures
*/
double SOM::randWeight()
{
	return (double)rand() / (RAND_MAX);
}

int SOM::calcIndex(int x, int y, int d) {
	return (x*_height + y)*_dimensions + d;
}

/*
	Calculates the euclidean distance between two vectors
*/
double SOM::EucDist(double* v1, double* v2) {
	double total = 0.0;
	for (int i = 0; i < this->_dimensions; i++) {
		total += (v1[i] - v2[i])*(v1[i] - v2[i]);
	}
	return sqrt(total);
}