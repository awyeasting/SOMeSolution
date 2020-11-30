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

__global__
void rowToColumnMajor(double *idata, double *odata, int nrows, int ncols, int n) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < n)
    odata[(i%ncols)*nrows + (i/ncols)] = idata[i];
}

double h(int j, int i, double initial_radius, double radius, int* BMUs, int height) {
	int i_y = i % height;
	int i_x = (i - i_y) / height;

	// Get BMU coord
	int j_y = BMUs[j] % height;
	int j_x = (BMUs[j] - j_y) / height;

	return initial_radius * exp(-(double)((j_x - i_x) * (j_x - i_x) + (j_y - i_y) * (j_y - i_y))/(radius * radius));
}

void trainOneEpoch(cublasHandle_t &handle, int device, double *train, double *weights, double *numer, double *denom, int map_size, int height, int num_examples, int dimensions, double initial_map_radius, double neighborhood_radius) {

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
	// m_sq = ones x (M * M)
	const double alpha0 = 1.0f;
	const double beta0 = 1.0f;
	double *m_sq;
	gpuErrchk(cudaMalloc(&m_sq, num_examples * map_size * sizeof(double)));
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_examples, map_size, dimensions, &alpha0, d_o, num_examples, d_msq, dimensions, &beta0, m_sq, num_examples);
	
	gpuErrchk(cudaDeviceSynchronize());
	
	gpuErrchk(cudaFree(d_msq));
	gpuErrchk(cudaFree(d_o));

	// Calc x_sq
	// Elementwise multiply X by X
	double *d_xsq;
	gpuErrchk(cudaMalloc(&d_xsq, num_examples * dimensions * sizeof(double)));
	NUM_BLOCKS = (int) ceil((float)(num_examples*dimensions)/NUM_THREADS);
	elementMul<<<NUM_BLOCKS, NUM_THREADS>>>(train, train, d_xsq, num_examples * dimensions);
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

	gpuErrchk(cudaDeviceSynchronize());

	// m_sq = - 2 * (x^t * m) + (m_sq)
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_examples, map_size, dimensions, &alpha1, train, num_examples, weights, dimensions, &beta0, m_sq, num_examples);

	gpuErrchk(cudaDeviceSynchronize());

	// D = (x_sq) + (-2 * x^t * m + m_sq)
	double *D;
	gpuErrchk(cudaMalloc(&D, num_examples * map_size * sizeof(double)));
	cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_examples, map_size, &alpha0, x_sq, num_examples, &beta0, m_sq, num_examples, D, num_examples);

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

	const double beta1 = 0.0f;

	// Calc denominators
	// Left multiply H by a num_examples dimensional vector of ones
	gpuErrchk(cudaMalloc(&d_o, num_examples * sizeof(double)));
	NUM_BLOCKS = (int)ceil((float)num_examples/NUM_THREADS);
	
	fillOnes<<<NUM_BLOCKS,NUM_THREADS>>>(d_o, num_examples);
	// denom = ones^T (1 x num examples) * H (num examples x map size)
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, map_size, num_examples, &alpha0, d_o, 1, H, num_examples, &beta1, denom, 1);
	
	gpuErrchk(cudaDeviceSynchronize());
	
	gpuErrchk(cudaFree(d_o));
	
	// Calc numerators
	// numer = H^T x X
	cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, map_size, dimensions, num_examples, &alpha0, H, num_examples, train, num_examples, &beta1, numer, map_size);

	gpuErrchk(cudaDeviceSynchronize());
	gpuErrchk(cudaFree(H));
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
	this->_dimensions = dimensions;
	const int map_size = this->_width * this->_height;
	// Initialize host numerators and denominators
	double *numer = (double *)malloc(map_size * dimensions * sizeof(double));
	double *denom = (double *)malloc(map_size * sizeof(double));
	for (int i = 0; i < map_size; i++) {
		denom[i] = 0.0;
		for (int j = 0; j < dimensions; j++) {
			numer[i*dimensions + j] = 0.0;
		}
	}

	// Establish multi gpu setup
	int NUM_GPUS;
	cudaGetDeviceCount(&NUM_GPUS);
	omp_set_dynamic(0); // Disable dynamic teams
	omp_set_num_threads(NUM_GPUS);

	// Allocate memory associated with each GPU
	cublasHandle_t* handles = (cublasHandle_t *)malloc(NUM_GPUS * sizeof(cublasHandle_t));
	double **d_train = (double **)malloc(NUM_GPUS * sizeof(double *));
	double **d_weights = (double **)malloc(NUM_GPUS * sizeof(double *));
	double **d_numer = (double **)malloc(NUM_GPUS * sizeof(double *));
	double **d_denom = (double **)malloc(NUM_GPUS * sizeof(double *));
	double **gnumer = (double **)malloc(NUM_GPUS * sizeof(double *));
	double **gdenom = (double **)malloc(NUM_GPUS * sizeof(double *));
	int *GPU_EXAMPLES = (int *)malloc(NUM_GPUS * sizeof(int));
	int *GPU_OFFSET = (int *)malloc(NUM_GPUS * sizeof(int));
	GPU_OFFSET[0] = 0;
	for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
		gpuErrchk(cudaSetDevice(gpu));
		// Create cublas handles associated with each device
		cublasCreate(&handles[gpu]);

		// Set the number of examples allocated to each GPU simply by equal division
		GPU_EXAMPLES[gpu] = num_examples/NUM_GPUS;
		if (gpu < NUM_GPUS-1)
			GPU_OFFSET[gpu+1] = GPU_OFFSET[gpu] + GPU_EXAMPLES[gpu];
		// Allocate remainder examples to last gpu
		else
			GPU_EXAMPLES[gpu] += num_examples - (GPU_OFFSET[gpu] + GPU_EXAMPLES[gpu]);
		
		// Allocate space for current GPU's share of the examples
		gpuErrchk(cudaMalloc(&d_train[gpu], GPU_EXAMPLES[gpu] * dimensions * sizeof(double)));
		// Allocate space for current GPU's copy of the map
		gpuErrchk(cudaMalloc(&d_weights[gpu], map_size * dimensions * sizeof(double)));
		// Allocate space for current GPU's copy of numerators and denominators
		gpuErrchk(cudaMalloc(&d_numer[gpu], map_size * dimensions * sizeof(double)));
		gpuErrchk(cudaMalloc(&d_denom[gpu], map_size * sizeof(double)));
		gnumer[gpu] = (double *)malloc(map_size * dimensions * sizeof(double));
		gdenom[gpu] = (double *)malloc(map_size * sizeof(double));
	}

	// Normalize data (to be within 0 to 1)
	normalizeData(trainData, num_examples);
	// Split training data onto gpus
	// then convert from row major ordering to the column major ordering of cuBLAS
	double **tempd_train = (double **)malloc(NUM_GPUS * sizeof(double *));
	int NUM_THREADS = 256;
	int NUM_BLOCKS;
	for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
		//std::cout << "Preparing train data for gpu " << gpu << std::endl;
		cudaSetDevice(gpu);
		NUM_BLOCKS = (int)ceil((float)(GPU_EXAMPLES[gpu]*dimensions)/NUM_THREADS);
		gpuErrchk(cudaMalloc(&tempd_train[gpu], GPU_EXAMPLES[gpu] * dimensions * sizeof(double)));
		gpuErrchk(cudaMemcpy(tempd_train[gpu], &trainData[GPU_OFFSET[gpu]], GPU_EXAMPLES[gpu] * dimensions * sizeof(double), cudaMemcpyHostToDevice));
		rowToColumnMajor<<<NUM_BLOCKS, NUM_THREADS>>>(tempd_train[gpu], d_train[gpu], GPU_EXAMPLES[gpu], dimensions, GPU_EXAMPLES[gpu] * dimensions);
		gpuErrchk(cudaDeviceSynchronize());
	}

	// Randomly initialize codebook	on first gpu
	const int CODEBOOK_INIT_DEVICE = 0;
	gpuErrchk(cudaSetDevice(CODEBOOK_INIT_DEVICE));
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	cudaDeviceSynchronize();
	// TODO: curandSetPseudoRandomGeneratorSeed(gen, );
	curandGenerateUniformDouble(gen, d_weights[CODEBOOK_INIT_DEVICE], map_size * dimensions);
	this->_weights = (double *)malloc(map_size * dimensions * sizeof(double));
	// Copy map from gpu to cpu
	gpuErrchk(cudaMemcpy(this->_weights, d_weights[CODEBOOK_INIT_DEVICE], map_size * dimensions * sizeof(double), cudaMemcpyDeviceToHost));
	// Copy map from the cpu to gpus
	for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
		gpuErrchk(cudaSetDevice(gpu));
		gpuErrchk(cudaMemcpy(d_weights[gpu], this->_weights, map_size * dimensions * sizeof(double), cudaMemcpyHostToDevice));
	}

	// Calc initial map radius
	double initial_map_radius = _width < _height ? ((double)_width) / 2.0 : ((double)_height) / 2.0;
	double time_constant = double(epochs) / log(initial_map_radius);

	// Synchronize devices and remove any temporary preprocessing memory allocated
	for (int gpu = 0; gpu < NUM_GPUS; gpu++) {
		gpuErrchk(cudaSetDevice(gpu));
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaFree(tempd_train[gpu]));
	}
	free(tempd_train);
	double neighborhood_radius;
	
	for(int epoch = 0; epoch < epochs; epoch++) {
		// Calculate current neighborhood radius
		neighborhood_radius = initial_map_radius * exp(-((double)(epoch))/time_constant);
		// Train a single epoch on all gpus
		#pragma omp parallel
		{
			int gpu = omp_get_thread_num();
			gpuErrchk(cudaSetDevice(gpu));
			gpuErrchk(cudaDeviceSynchronize());
			trainOneEpoch(handles[gpu], gpu, d_train[gpu], d_weights[gpu], d_numer[gpu], d_denom[gpu], map_size, this->_height, GPU_EXAMPLES[gpu], dimensions, initial_map_radius, neighborhood_radius);
			gpuErrchk(cudaMemcpy(gnumer[gpu],d_numer[gpu], map_size * dimensions * sizeof(double), cudaMemcpyDeviceToHost));
			gpuErrchk(cudaMemcpy(gdenom[gpu],d_denom[gpu], map_size * sizeof(double), cudaMemcpyDeviceToHost));
		}

		// Update codebook/map
		// Reduce numerators and denominators
		// TODO: Implement more complex reduction
		for(int gpu = 0; gpu < NUM_GPUS; gpu++) {
			if (gpu == 0) {
				for (int i = 0; i < map_size; i++) {
					denom[i] = gdenom[gpu][i];
					for (int d = 0; d < dimensions; d++) {
						numer[i + d*map_size] = gnumer[gpu][i + d*map_size];
					}
				}
			} else {
				for (int i = 0; i < map_size; i++) {
					denom[i] += gdenom[gpu][i];
					for (int d = 0; d < dimensions; d++) {
						numer[i + d*map_size] += gnumer[gpu][i + d*map_size];
					}
				}
			}
		}
		// Recalculate weights with new numerators and denominators
		for (int i = 0; i < map_size; i++) {
			for (int d = 0; d < dimensions; d++) {
				this->_weights[i*dimensions + d] = numer[i + d*map_size] / denom[i];
			}
		}

		// If not the last epoch update gpu copies of codebook/map
		if (epoch != epochs - 1) { 
			#pragma omp parallel
			{
				int gpu = omp_get_thread_num();
				gpuErrchk(cudaSetDevice(gpu));
				gpuErrchk(cudaMemcpy(d_weights[gpu], this->_weights, map_size * dimensions * sizeof(double), cudaMemcpyHostToDevice));
			}
		}
	}

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