#include "SOM.h"
#include <float.h>
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
	Generates a random set of training data if there is no input file given
*/

double* SOM::generateRandomTrainingInputs(unsigned int examples, unsigned int dimensions, int seedValue)
{
	double *returnData = new double [examples * dimensions];
	srand(seedValue);
	for (int i = 0; i < examples; i++)
	{
		int rowMod = (examples - i - 1)*dimensions;
		for (int j = 0; j < dimensions; j++)
		{
			double weight = SOM::randWeight();
			returnData[rowMod+j] = weight;
		}
	}

	for(int i = 0; i < examples; i++){
		int rowMod = (examples-i-1)*dimensions;
		for(int j = 0; j < dimensions; j++){
			std::cout << returnData[rowMod+j] << " ";
		}
		std::cout << std::endl;
	}
	return returnData;
}

/*
	Load a set of training data from a given filename
*/
double* SOM::loadTrainingData(std::fstream& in, unsigned int& rows, unsigned int& cols, int read_count, double* featureMaxes, double* featureMins) {
	// Open file ****DO THIS BEFORE CALLING FUNCTION*****
	
	// std::ifstream in(trainDataFileName, std::ifstream::in);
	// if (!in.is_open()) {
	// 	std::cout << "Invalid training data file '" << trainDataFileName << "'" << std::endl;
	// 	return NULL;
	// }

	// Read the first line to obtain the number of columns (dimensions) in the training data
	std::string line;
	std::getline(in, line);

	std::stringstream ss(line);
	std::vector<double> line1;
	double temp;
	cols = 0;
	while (ss >> temp) {
		if (temp > featureMaxes[cols])
		{
			featureMaxes[cols] = temp;
		}
		if (temp < featureMins[cols])
		{
			featureMins[cols] = temp;
		}
		cols++;
		line1.push_back(temp);
	}
	std::vector<double*> lines;

	// Store first line in dynamic array and put into the vector of rows
	double* tempLine1 = new double[cols];
	for (int j = 0; j < cols; j++) {
		tempLine1[cols - j - 1] = line1.back();
		line1.pop_back();
	}
	lines.push_back(tempLine1);

	// Read all numbers into cols dimensional arrays added to the rows list
	int i = 0;
	double* unpackedLine = NULL;

	for(int linesNum = 1; linesNum < read_count * cols; linesNum++){
		in >> temp;
		if (!unpackedLine) {
			unpackedLine = new double[cols];
		}
		unpackedLine[i] = temp;
		if (temp > featureMaxes[i])
		{
			featureMaxes[i] = temp;
		}
		if (temp < featureMins[i])
		{
			featureMins[i] = temp;
		}
		i++;
		if (i == cols) {
			lines.push_back(unpackedLine);
			i = 0;
			unpackedLine = NULL;
		}
	}
	


	// while (in >> temp) {
	// 	if (!unpackedLine) {
	// 		unpackedLine = new double[cols];
	// 	}
	// 	unpackedLine[i] = temp;
	// 	if (temp > featureMaxes[i])
	// 	{
	// 		featureMaxes[i] = temp;
	// 	}
	// 	if (temp < featureMins[i])
	// 	{
	// 		featureMins[i] = temp;
	// 	}
	// 	i++;
	// 	if (i == cols) {
	// 		lines.push_back(unpackedLine);
	// 		i = 0;
	// 		unpackedLine = NULL;
	// 	}
	// }

	// Convert vector of arrays into 1d array of examples
	rows = lines.size();
	double* res = new double[rows * cols];
	for (i = 0; i < rows; i++) {
		double* temp = lines.back();
		int rowMod = (rows-i-1)*cols;
		for (int j = 0; j < cols; j++) {
			res[rowMod + j] = temp[j];
		}
		lines.pop_back();
		free(temp);
	}

	//print debug
	for(int i = 0; i < rows; i++){
		int rowMod = (rows-i-1)*cols;
		for(int j = 0; j < cols; j++){
			std::cout << res[rowMod+j] << " ";
		}
		std::cout << std::endl;
	}

	return res;
}

/*
	Every process would run this
*/
void SOM::train_one_epoch(double* localMap, double* train_data, double* numerators, double* denominators, int num_examples, double initial_map_radius, int epoch, double time_constant)
{
	double* D = (double *)malloc(num_examples * _width * _height * sizeof(double));
	double* m_sq = (double *)malloc(_width * _height * sizeof(double));
	double* x_sq = (double *)malloc(num_examples * sizeof(double));
	int* BMUs = (int *)malloc(num_examples * sizeof(int));
	double* H = (double *)malloc(num_examples * _width * _height * sizeof(double));
	double neighborhood_radius;
	neighborhood_radius = initial_map_radius * exp(-double(epoch)/time_constant);

	

	//learning_rate = initial_learning_rate * exp(-double(epoch)/time_constant);

	// Find BMUs for every input instance
	// D = X_sq - 2X^TM + M_sq
	// D (xdn * nn)
	// Calc m_sq
	SqDists(localMap, _width * _height, _dimensions, m_sq);
	
	// Calc x_sq
	#pragma omp parallel
	SqDists(train_data, num_examples, _dimensions, x_sq);

	

	//Calculate D matrix
	#pragma omp parallel for
	for (int j = 0; j < num_examples; j++) {
		for (int i = 0; i < _width * _height; i++) {
			// Calc x^Tm
			double xm = 0;
			for (int d = 0; d < _dimensions; d++) {
				xm += train_data[j * _dimensions + d] * localMap[i * _dimensions + d];
			}
			// Combine all
			D[j * _width * _height + i] = x_sq[j] - 2 * xm + m_sq[i];
		}
	}
	
	// BMU index of each training instance
	for (int j = 0; j < num_examples; j++) {
		BMUs[j] = 0;
		for (int i = 1; i < _width * _height; i++) {
			if (D[j * _width * _height + i] < D[j * _width * _height + BMUs[j]]) {
				BMUs[j] = i;
			}
		}
	}
	// Calc gaussian function 
	// (num_examples x num nodes)
	#pragma omp parallel for
	for (int j = 0; j < num_examples; j++) {
		for (int i = 0; i < _width * _height; i++) {
			H[j*_width*_height + i] = h(j, i, initial_map_radius, neighborhood_radius, BMUs);
		}
	}
	// Left multiply H by a num_examples dimensional vector of ones
	for (int i = 0; i < _width * _height; i++) {
		denominators[i] = 0.0;
		for (int j = 0; j < num_examples; j++) {
			denominators[i] += H[j*_width*_height + i];
		}
	}
	
	//Calculate numerators
	#pragma omp parallel for
	for (int i = 0; i < _width * _height; i++) {
		for (int d = 0; d < _dimensions; d++) {
			numerators[i * _dimensions + d] = 0.0;
			for (int j = 0; j < num_examples; j++) {
				numerators[i*_dimensions + d] += H[j*_width*_height + i] * train_data[j*_dimensions + d];
			}
		}
	}
	free(D);
	free (m_sq);
	free (x_sq);
	free(H);
	free(BMUs);
}

std::fstream& SOM::GotoLine(std::fstream& file, unsigned int num){
    file.seekg(std::ios::beg);
    for(int i=0; i < num - 1; ++i){
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    }
    return file;
}

/*
	Train the SOM using a set of training data over a given number of epochs with a given learning rate
*/
void SOM::train_data(std::string fileName,unsigned int current_rank, unsigned int num_procs, unsigned int epochs, unsigned int dimensions, unsigned int rowCount, int rank_seed)
{
	double * train_data;
	int start, shift, read_count;
	double* global_max= (double *)malloc(sizeof(double) * dimensions);
	double* global_min=(double *)malloc(sizeof(double) * dimensions);
	//Where we load in the file.
	start = ((rowCount / num_procs) * current_rank) + 1;
	read_count = rowCount / num_procs;
	if (current_rank >= (num_procs - (rowCount %num_procs)))
	{
		shift = current_rank - (num_procs - (rowCount % num_procs));
		start += shift;
		read_count += 1;
	}

	if (fileName == "")
	{
		int current_rank_seed;
		//Rank 0 create seed value array. Scatter to current_rank_seed.
		train_data = generateRandomTrainingInputs(read_count, dimensions, rank_seed);
	}
	else
	{
		std::fstream in(fileName, std::ios::in | std::ios::out);

		if (!in.is_open()) {
			std::cout << "Invalid training data file '" << fileName << "'" << std::endl;
		}

		_featureMaxes = (double *)malloc(sizeof(double) * dimensions);
		_featureMins = (double*)malloc(sizeof(double) * dimensions);

		for(int i =0; i < dimensions; i++){
			_featureMaxes[i] = -1;
			_featureMins[i] = 10000;
		}

		std::fstream& file = GotoLine(in, start);
		//Need to do reading with localmaxes and localMins.
		train_data = loadTrainingData(file, rowCount, dimensions, read_count, _featureMaxes, _featureMins);

		MPI_Barrier(MPI::COMM_WORLD);

		//RANK 0 Reduces, 
		// Allreduce Maxes
		MPI_Allreduce(_featureMaxes, global_max, dimensions, MPI::DOUBLE , MPI::MAX, MPI::COMM_WORLD);
		// Reduce Mins
		MPI_Allreduce(_featureMins, global_min, dimensions, MPI::DOUBLE , MPI::MIN, MPI::COMM_WORLD);

		if(current_rank == 1){
			for(int i = 0; i < dimensions;i++){
				std::cout<< "_featureMaxes["<<i<<"]=" <<_featureMaxes[i]<<std::endl;
			}
			for(int i = 0; i < dimensions;i++){
				std::cout<< "_featureMins["<<i<<"]=" <<_featureMins[i]<<std::endl;
			}

			std::cout<<"global_max="<<global_max[0]<<std::endl;
			std::cout<<"global_min="<<global_min[0]<<std::endl;
		}
		
		//MPI BARRIER not sure if this is needed, because I think All_reduce is blocking.
		MPI_Barrier(MPI::COMM_WORLD);
		this->_dimensions = dimensions;
		normalizeData(train_data, read_count, global_max, global_min);

		
		// for(int i = 0; i < read_count; i++){
		// 	int rowMod = (read_count-i-1)*dimensions;
		// 	for(int j = 0; j < dimensions; j++){
		// 		std::cout << train_data[rowMod+j] << " ";
		// 	}
		// 	std::cout << std::endl;
		// }
	}

	this->_dimensions = dimensions;
	
	//Rank 0 needs to do the initalization of the map.
	if (current_rank == 0)
	{
		this->_weights = (double *)malloc(_width * _height * _dimensions * sizeof(double));
		for (int i = 0; i < _width; i++) {
			for (int j = 0; j < _height; j++) {
				for (int d = 0; d < _dimensions; d++) {
					this->_weights[calcIndex(i,j,d)] = randWeight();
				}
			}
		}
	}
	

	// Calc initial map radius and time constant.
	double initial_map_radius = _width < _height ? ((double)_width) / 2.0 : ((double)_height) / 2.0;
	double time_constant = double(epochs) / log(initial_map_radius);

	//local_map is the variable used to broadcast, and modify each process.
	//local_numerators  and local_denominators are the variables used to pass to train_one_epoch and then be reduced
	//global_numerators/denominators are the variables used by rank 0 to reduce the local, and then update the map with.
	double* local_map = (double *)malloc(_width * _height * _dimensions * sizeof(double));
	double* local_numerators = (double*)malloc(_width * _height * _dimensions * sizeof(double));
	double* local_denominators = (double*)malloc(_width * _height * sizeof(double));
	double* global_numerators;
	double* global_denominator;
	//Have rank 0 allocate the memory for global num and denom as it will be doing the updating.

	if (current_rank == 0){
		global_numerators = (double*)malloc(_width * _height * _dimensions*sizeof(double));
		global_denominator = (double *)malloc(_width * _height * sizeof(double));
	}

	//Loop for argument passed number of times.
	for(int epoch = 0; epoch < epochs; epoch++) {

		//Filling localMap in rank 0 to broadcast to all processes

		if (current_rank == 0){
			for (int i = 0; i <_width; i++){
				for (int j = 0; j < _height; j++){
					for (int d = 0; d < _dimensions; d++){
						local_map[calcIndex(i,j,d)] = _weights[calcIndex(i,j,d)];
					}
				}
			}
		}
		
		MPI_Barrier(MPI::COMM_WORLD);
		MPI_Bcast(local_map, _width*_height*_dimensions, MPI::DOUBLE, 0, MPI::COMM_WORLD);
		train_one_epoch(local_map, train_data, local_numerators, local_denominators, read_count, initial_map_radius, epoch, time_constant);	
		
		MPI_Barrier(MPI::COMM_WORLD);

		MPI_Reduce(local_numerators, global_numerators, _width *_height * _dimensions, MPI::DOUBLE, MPI::SUM, 0, MPI::COMM_WORLD);
		MPI_Reduce(local_denominators, global_denominator, _width * _height, MPI::DOUBLE, MPI::SUM, 0, MPI::COMM_WORLD);
		if (current_rank == 0)
		{
			// Update codebook
			#pragma omp parallel for
			for (int i = 0; i < _width * _height; i++) {
				for (int d = 0; d < _dimensions; d++) {
					this->_weights[i*_dimensions + d] = global_numerators[i*_dimensions + d]/global_denominator[i];
				}
			}
		}
	}
	if(current_rank == 0)
	{
		free(global_denominator);
		free(global_numerators);
	}
	free(local_map);
	free(local_numerators);
	free(local_denominators);
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
void SOM::normalizeData(double *trainData, int num_examples, double* max, double* min)
{
	std::cout << "In normalizeData()" << std::endl;
	// Find the max and min value for each feature then use it to normalize the feature
	// for (int d = 0; d < this->_dimensions; d++)
	// {
	// 	for (int i = 0; i < num_examples; i++) {
	// 		trainData[i*_dimensions + d] = (trainData[i*_dimensions + d] - this->_featureMins[d])/(this->_featureMaxes[d]-this->_featureMins[d]);
	// 		std::cout << trainData[i*_dimensions + d] << std::endl;
	// 	}
	// }
	for(int i = 0; i < num_examples; i++){
		int rowMod = (num_examples-i-1)*this->_dimensions;
		for(int j = 0; j < this->_dimensions; j++){
			// std::cout << "FeatureMins: " << this->_featureMins[j] << std::endl;
			// std::cout << "FeatureMaxes: " << this->_featureMaxes[j] << std::endl;
			// std::cout << "Data: " << trainData[rowMod+j] << std::endl;
			std::cout << trainData[rowMod+j] <<"=("<<trainData[rowMod+j]<<"-"<<min[j]<<")/("<<max[j]<<"-"<<min[j]<<")"<<std::endl;
			trainData[rowMod+j] = (trainData[rowMod+j] - min[j])/(max[j]-min[j]);
			std::cout << trainData[rowMod+j] << " ";
		}
		std::cout << std::endl;
	}
	
	// for (int d = 0; d < num_examples; d++)
	// {
	// 	int rowMod = (num_examples-d-1)*this->_dimensions;
	// 	for (int i = 0; i < this->_dimensions; i++) {
	// 		//std::cout << "FeatureMins: " << this->_featureMins[d] << std::endl;
	// 		//std::cout << "FeatureMaxes: " << this->_featureMaxes[d] << std::endl;
	// 		trainData[rowMod+i] = (trainData[rowMod+i] - this->_featureMins[d])/(this->_featureMaxes[d]-this->_featureMins[d]);
	// 		std::cout << trainData[rowMod+i] << std::endl;
	// 	}
	// }
	/*
	
		for(int i = 0; i < read_count; i++){
			int rowMod = (read_count-i-1)*dimensions;
			for(int j = 0; j < dimensions; j++){
				std::cout << train_data[rowMod+j] << " ";
			}
			std::cout << std::endl;
		}
	
	*/

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

void SOM::SqDists(double* m, int loop, int dim, double* output) {
	#pragma omp for
	for (int i = 0; i < loop; i++) {
		output[i] = 0;
		for (int d = 0; d < dim; d++) {
			output[i] += m[i * dim + d] * m[i * dim + d]; 
		}
	}
}

double SOM::h(int j, int i, double initial_radius, double radius, int* BMUs) {
	int i_y = i % _height;
	int i_x = (i - i_y) / _height;

	// Get BMU coord
	int j_y = BMUs[j] % _height;
	int j_x = (BMUs[j] - j_y) / _height;

	return initial_radius * exp(-(double)((j_x - i_x) * (j_x - i_x) + (j_y - i_y) * (j_y - i_y))/(radius * radius));
}