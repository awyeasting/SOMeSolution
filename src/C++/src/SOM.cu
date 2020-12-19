#include "SOM.h"
#include <curand.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

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
//	SOM private member functions
//----------------------------------------------------

void SOM::trainOneEpochOneGPU(int gpu) {

	// Set assigned gpu
	gpuErrchk(cudaSetDevice(this->_gpus[gpu])); // Map internal gpu id to device id
	gpuErrchk(cudaDeviceSynchronize());

	// Establish matrix of ones for multiplication
	int d_o_num = std::max(this->_GPU_EXAMPLES[gpu], this->_mapSize) * this->_dimensions;
	int NUM_THREADS = 256;
	int NUM_BLOCKS = (int) ceil((float)(d_o_num)/NUM_THREADS);
	double* d_o;
	gpuErrchk(cudaMalloc(&d_o, d_o_num * sizeof(double)));
	fillOnes<<<NUM_BLOCKS,NUM_THREADS>>>(d_o, d_o_num);
	gpuErrchk(cudaDeviceSynchronize());

	// Find BMUs for every input instance
	// D = X_sq - 2X^TM + M_sq
	// D (xdn * nn)
	
	// Calc m_sq
	// Elementwise multiply M by M
	double *d_msq;
	gpuErrchk(cudaMalloc(&d_msq, this->_mapSize * this->_dimensions * sizeof(double)));
	NUM_BLOCKS = (int) ceil((float)(this->_mapSize * this->_dimensions)/NUM_THREADS);
	elementMul<<<NUM_BLOCKS, NUM_THREADS>>>(this->_d_weights[gpu], this->_d_weights[gpu], d_msq, this->_mapSize * this->_dimensions);
	gpuErrchk(cudaDeviceSynchronize());
	// Left multiply elementwise multiplied M by all ones matrix (of dim num examples x dimensions)
	// m_sq = ones x (M * M)^T
	const double alpha0 = 1.0f;
	const double beta0 = 0.0f;
	double *m_sq;
	gpuErrchk(cudaMalloc(&m_sq, this->_GPU_EXAMPLES[gpu] * this->_mapSize * sizeof(double)));
	cublasDgemm(this->_handles[gpu], CUBLAS_OP_N, CUBLAS_OP_T, this->_GPU_EXAMPLES[gpu], this->_mapSize, this->_dimensions, &alpha0, d_o, this->_GPU_EXAMPLES[gpu], d_msq, this->_mapSize, &beta0, m_sq, this->_GPU_EXAMPLES[gpu]);
	gpuErrchk(cudaDeviceSynchronize());
	
	gpuErrchk(cudaFree(d_msq));

	// Calc x_sq
	// Elementwise multiply X by X
	double *d_xsq;
	gpuErrchk(cudaMalloc(&d_xsq, this->_GPU_EXAMPLES[gpu] * this->_dimensions * sizeof(double)));
	NUM_BLOCKS = (int) ceil((float)(this->_GPU_EXAMPLES[gpu] * this->_dimensions)/NUM_THREADS);
	elementMul<<<NUM_BLOCKS, NUM_THREADS>>>(this->_d_train[gpu], this->_d_train[gpu], d_xsq, this->_GPU_EXAMPLES[gpu] * this->_dimensions);
	gpuErrchk(cudaDeviceSynchronize());
	// Left multiply elementwise multiplied X by all ones matrix (of dim num examples x dimensions)
	// x_sq = (X * X) x ones
	double *x_sq;
	gpuErrchk(cudaMalloc(&x_sq, this->_GPU_EXAMPLES[gpu] * this->_mapSize * sizeof(double)));
	cublasDgemm(this->_handles[gpu], CUBLAS_OP_N, CUBLAS_OP_N, this->_GPU_EXAMPLES[gpu], this->_mapSize, this->_dimensions, &alpha0, d_xsq, this->_GPU_EXAMPLES[gpu], d_o, this->_dimensions, &beta0, x_sq, this->_GPU_EXAMPLES[gpu]);
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaFree(d_xsq));

	// Calc D
	// From paper: 
	// D = x_sq - 2 * x^t * m + m_sq

	const double alpha1 = -2.0f;
	const double beta1 = 1.0f;

	// m_sq = - 2 * (x^t * m) + (m_sq)
	cublasDgemm(this->_handles[gpu], CUBLAS_OP_N, CUBLAS_OP_T, this->_GPU_EXAMPLES[gpu], this->_mapSize, this->_dimensions, &alpha1, this->_d_train[gpu], this->_GPU_EXAMPLES[gpu], this->_d_weights[gpu], this->_mapSize, &beta1, m_sq, this->_GPU_EXAMPLES[gpu]);
	gpuErrchk(cudaDeviceSynchronize());

	// D = (x_sq) + (-2 * x^t * m + m_sq)
	double *D;
	gpuErrchk(cudaMalloc(&D, this->_GPU_EXAMPLES[gpu] * this->_mapSize * sizeof(double)));
	cublasDgeam(this->_handles[gpu], CUBLAS_OP_N, CUBLAS_OP_N, this->_GPU_EXAMPLES[gpu], this->_mapSize, &alpha0, x_sq, this->_GPU_EXAMPLES[gpu], &beta1, m_sq, this->_GPU_EXAMPLES[gpu], D, this->_GPU_EXAMPLES[gpu]);
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaFree(m_sq));
	gpuErrchk(cudaFree(x_sq));

	// BMU index of each training instance
	int *BMUs;
	gpuErrchk(cudaMalloc(&BMUs, this->_GPU_EXAMPLES[gpu] * sizeof(int)));
	NUM_BLOCKS = (int) ceil((float)(this->_GPU_EXAMPLES[gpu])/NUM_THREADS);
	findBMUsGPU<<<NUM_BLOCKS, NUM_THREADS>>>(D, BMUs, this->_GPU_EXAMPLES[gpu], this->_mapSize);
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaFree(D));

	// Calc gaussian function 
	// (num_examples x num nodes)
	int BLOCK_SIZE = 16;
	int GRID_HEIGHT = (int)ceil((float)this->_GPU_EXAMPLES[gpu]/BLOCK_SIZE);
	int GRID_WIDTH = (int)ceil((float)this->_mapSize/BLOCK_SIZE);
	dim3 grid(GRID_WIDTH, GRID_HEIGHT);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	double *H;
	gpuErrchk(cudaMalloc(&H, this->_GPU_EXAMPLES[gpu] * this->_mapSize * sizeof(double)));
	calcGaussian<<<grid, threads>>>(H, this->_GPU_EXAMPLES[gpu], this->_mapSize, this->_initial_map_radius, this->_neighborhood_radius, BMUs, this->_height);
	gpuErrchk(cudaDeviceSynchronize());

	gpuErrchk(cudaFree(BMUs));

	// Calc denominators
	// Left multiply H by a num_examples dimensional vector of ones
	// denom = ones^T (1 x num examples) * H (num examples x map size)
	cublasDgemm(this->_handles[gpu], CUBLAS_OP_N, CUBLAS_OP_N, 1, this->_mapSize, this->_GPU_EXAMPLES[gpu], &alpha0, d_o, 1, H, this->_GPU_EXAMPLES[gpu], &beta0, this->_d_denom[gpu], 1);
	gpuErrchk(cudaDeviceSynchronize());
	
	gpuErrchk(cudaFree(d_o));
	
	// Calc numerators
	// numer = H^T x X
	cublasDgemm(this->_handles[gpu], CUBLAS_OP_T, CUBLAS_OP_N, this->_mapSize, this->_dimensions, this->_GPU_EXAMPLES[gpu], &alpha0, H, this->_GPU_EXAMPLES[gpu], this->_d_train[gpu], this->_GPU_EXAMPLES[gpu], &beta0, this->_d_numer[gpu], this->_mapSize);
	gpuErrchk(cudaDeviceSynchronize());
	
	gpuErrchk(cudaFree(H));
}

void SOM::trainOneEpochMultiGPU() {
	#pragma omp parallel
	{
		int gpu_id = omp_get_thread_num();
		trainOneEpochOneGPU(gpu_id);
		
		// Copy GPU copy of numerators and denominators back to CPU
		gpuErrchk(cudaMemcpy(this->_gnumer[gpu_id],this->_d_numer[gpu_id], this->_mapSize * this->_dimensions * sizeof(double), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(this->_gdenom[gpu_id],this->_d_denom[gpu_id], this->_mapSize * sizeof(double), cudaMemcpyDeviceToHost));
	}
}

void SOM::chooseGPUs() {
	// If numGPUs was not passed by the user then just use average available in group
	if (this->_numGPUs <= 0) {
		cudaGetDeviceCount(&this->_totalNodeGPUs);
		if (this->_totalNodeGPUs < this->_numGroupProcs) {
			if (this->_groupRank < this->_totalNodeGPUs)
				this->_numGPUs = 1;
			else {
				std::cout << "WARNING: Too many processors allocated, rank " << this->_rank << " was unable to claim one" << std::endl; 
				this->_numGPUs = 0;
			}
		} else {
			// Equally divide gpus among the processes
			this->_numGPUs = this->_totalNodeGPUs/this->_numGroupProcs;
			// Give extra gpus to last node
			this->_numGPUs += (this->_groupRank == this->_numGroupProcs - 1 ? this->_totalNodeGPUs % this->_numGroupProcs : 0);
		}
	}
	// If no gpus were assigned then assign them
	if (this->_gpus == NULL) {
		this->_gpus = (int *)malloc(this->_numGPUs * sizeof(double));
    	for(int i = 0; i < this->_numGPUs; i++) {
        	this->_gpus[i] = i + (this->_groupRank * (this->_totalNodeGPUs / std::min(this->_numGroupProcs, this->_totalNodeGPUs)));
    	}
	}
}

void SOM::initMultiGPUSetup() {
	chooseGPUs();

	omp_set_dynamic(0); // Disable dynamic teams
	omp_set_num_threads(this->_numGPUs);

	// Allocate memory associated with training on each GPU on each node
	allocNumerDenom();
	allocGPUTrainMemory();

	// Split training data onto gpus on each node
	initGPUTrainData();

	initCodebook();
}

void SOM::initGPUTrainData() {
	#pragma omp parallel
	{
		int NUM_BLOCKS;
		int NUM_THREADS = 256;
		int gpu = omp_get_thread_num();

		NUM_BLOCKS = (int)ceil((float)(this->_GPU_EXAMPLES[gpu] * this->_dimensions)/NUM_THREADS);
		double *temp_d_train;

		gpuErrchk(cudaSetDevice(this->_gpus[gpu])); // Map gpu id to device id
		gpuErrchk(cudaMalloc(&temp_d_train, this->_GPU_EXAMPLES[gpu] * this->_dimensions * sizeof(double)));
		gpuErrchk(cudaMemcpy(temp_d_train, &this->_trainData[this->_GPU_OFFSET[gpu] * this->_dimensions], this->_GPU_EXAMPLES[gpu] * this->_dimensions * sizeof(double), cudaMemcpyHostToDevice));
		// Convert data from row major order to 
		rowToColumnMajor<<<NUM_BLOCKS, NUM_THREADS>>>(temp_d_train, this->_d_train[gpu], this->_GPU_EXAMPLES[gpu], this->_dimensions, this->_GPU_EXAMPLES[gpu] * this->_dimensions);
		gpuErrchk(cudaDeviceSynchronize());
		gpuErrchk(cudaFree(temp_d_train));
	}
}

void SOM::allocGPUTrainMemory() {
	this->_handles = (cublasHandle_t *)malloc(this->_numGPUs * sizeof(cublasHandle_t));
	this->_d_train = (double **)malloc(this->_numGPUs * sizeof(double *));
	this->_d_weights = (double **)malloc(this->_numGPUs * sizeof(double *));
	this->_d_numer = (double **)malloc(this->_numGPUs * sizeof(double *));
	this->_d_denom = (double **)malloc(this->_numGPUs * sizeof(double *));
	this->_GPU_EXAMPLES = (unsigned int *)malloc(this->_numGPUs * sizeof(unsigned int));
	this->_GPU_OFFSET = (unsigned int *)malloc(this->_numGPUs * sizeof(unsigned int));
	this->_GPU_OFFSET[0] = 0;

	for (int gpu = 0; gpu < this->_numGPUs; gpu++) {
		gpuErrchk(cudaSetDevice(this->_gpus[gpu])); // Map internal gpu id to device id
		// Create cublas handles associated with each device
		cublasCreate(&this->_handles[gpu]);

		// Set the number of examples allocated to each GPU simply by equal division
		this->_GPU_EXAMPLES[gpu] = this->_numExamples/this->_numGPUs;
		if (gpu < this->_numGPUs-1)
			this->_GPU_OFFSET[gpu+1] = this->_GPU_OFFSET[gpu] + this->_GPU_EXAMPLES[gpu];
		// Allocate remainder examples to last gpu
		else
		this->_GPU_EXAMPLES[gpu] += this->_numExamples - (this->_GPU_OFFSET[gpu] + this->_GPU_EXAMPLES[gpu]);
		
		// Allocate space for current GPU's share of the examples
		gpuErrchk(cudaMalloc(&this->_d_train[gpu], this->_GPU_EXAMPLES[gpu] * this->_dimensions * sizeof(double)));
		// Allocate space for current GPU's copy of the map
		gpuErrchk(cudaMalloc(&this->_d_weights[gpu], this->_mapSize * this->_dimensions * sizeof(double)));
		// Allocate space for current GPU's copy of numerators and denominators
		gpuErrchk(cudaMalloc(&this->_d_numer[gpu], this->_mapSize * this->_dimensions * sizeof(double)));
		gpuErrchk(cudaMalloc(&this->_d_denom[gpu], this->_mapSize * sizeof(double)));
	}
}

void SOM::allocNumerDenom() {
	// Allocate memory for CPU copies of GPU numerators and denominators
	this->_gnumer = (double **)malloc(this->_numGPUs * sizeof(double *));
	this->_gdenom = (double **)malloc(this->_numGPUs * sizeof(double *));
	for (int gpu = 0; gpu < this->_numGPUs; gpu++) {
		this->_gnumer[gpu] = (double *)malloc(this->_mapSize * this->_dimensions * sizeof(double));
		this->_gdenom[gpu] = (double *)malloc(this->_mapSize * sizeof(double));
	}

	// Allocate memory for local numerators and denominators
	this->_numer = (double *)malloc(this->_mapSize * this->_dimensions * sizeof(double));
	this->_denom = (double *)malloc(this->_mapSize * sizeof(double));

	// Allocate memory for global numerators and denominators
	// TODO: verify that global_numer and denom only need to be allocated on rank 0
	if (this->_rank == 0) {
		this->_global_numer = (double *)malloc(this->_mapSize * this->_dimensions * sizeof(double));
		this->_global_denom = (double *)malloc(this->_mapSize * sizeof(double));
	}

}

void SOM::initCodebook() {
	this->_weights = (double *)malloc(this->_mapSize * this->_dimensions * sizeof(double));
	if (this->_rank == 0) {
		srand(this->_mapSeed);
		if (GPU_BASED_CODEBOOK_INIT)
			initCodebookOnGPU();
		else
			initCodebookOnCPU();
	}
}

void SOM::initCodebookOnCPU() {
	for (int i = 0; i < this->_mapSize; i++) {
		for (int d = 0; d < this->_dimensions; d++) {
			this->_weights[i * this->_dimensions + d] = this->randWeight();
		}
	}
}

void SOM::initCodebookOnGPU() {
	const int CODEBOOK_INIT_DEVICE = this->_gpus[0]; // Map internal gpu id to device id
	gpuErrchk(cudaSetDevice(CODEBOOK_INIT_DEVICE));
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	cudaDeviceSynchronize();
	curandSetPseudoRandomGeneratorSeed(gen, this->_mapSeed);
	curandGenerateUniformDouble(gen, this->_d_weights[CODEBOOK_INIT_DEVICE], this->_mapSize * this->_dimensions);
	
	// Copy map from gpu to cpu
	gpuErrchk(cudaMemcpy(this->_weights, this->_d_weights[CODEBOOK_INIT_DEVICE], this->_mapSize * this->_dimensions * sizeof(double), cudaMemcpyDeviceToHost));
}

void SOM::updateGPUCodebooks() {
	//#pragma omp parallel
	for (int gpu = 0; gpu < this->_numGPUs; gpu++)
	{
		//int gpu = omp_get_thread_num();
		updateGPUCodebook(gpu)
	}
}

void SOM::updateGPUCodebook(int gpu) {
	gpuErrchk(cudaSetDevice(this->_gpus[gpu])); // Map internal gpu id to device id
	gpuErrchk(cudaMemcpy(this->_d_weights[gpu], this->_weights, this->_mapSize * this->_dimensions * sizeof(double), cudaMemcpyHostToDevice));
}

void SOM::freeGPUMemory() {
	for (int gpu = 0; gpu < this->_numGPUs; gpu++) {
		cudaSetDevice(this->_gpus[gpu]); // Map internal gpu id to device id
		cublasDestroy(this->_handles[gpu]);
		cudaFree(this->_d_train[gpu]);
		cudaFree(this->_d_weights[gpu]);
		cudaFree(this->_d_numer[gpu]);
		cudaFree(this->_d_denom[gpu]);
		free(this->_gnumer[gpu]);
		free(this->_gdenom[gpu]);
	}
}