#include "SOM.h"

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
	// Normalize data (to be within 0 to 1)
	normalizeData(trainData, num_examples);

	// Randomly initialize codebook
	this->_weights = (double *)malloc(_width * _height * _dimensions * sizeof(double));
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

	double* D = (double *)malloc(num_examples * _width * _height * sizeof(double));
	double* m_sq = (double *)malloc(_width * _height * sizeof(double));
	double* x_sq = (double *)malloc(num_examples * sizeof(double));
	int* BMUs = (int *)malloc(num_examples * sizeof(int));
	double* H = (double *)malloc(num_examples * _width * _height * sizeof(double));
	double* numerators = (double *)malloc(_width * _height * _dimensions * sizeof(double));
	double* denominators = (double *)malloc(_width * _height * sizeof(double));

	double neighborhood_radius;
	for(int epoch = 0; epoch < epochs; epoch++) {
		//learning_rate = initial_learning_rate * exp(-double(epoch)/time_constant);
		neighborhood_radius = initial_map_radius * exp(-double(epoch)/time_constant);

		// Find BMUs for every input instance
		// D = X_sq - 2X^TM + M_sq
		// D (xdn * nn)
		
		// Calc m_sq
		SqDists(this->_weights, _width * _height, _dimensions, m_sq);

		// Calc x_sq
		#pragma omp parallel
		SqDists(trainData, num_examples, _dimensions, x_sq);

		#pragma omp parallel for
		for (int j = 0; j < num_examples; j++) {
			for (int i = 0; i < _width * _height; i++) {
				// Calc x^Tm
				double xm = 0;
				for (int d = 0; d < _dimensions; d++) {
					xm += trainData[j * _dimensions + d] * this->_weights[i * _dimensions + d];
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

		#pragma omp parallel for
		for (int i = 0; i < _width * _height; i++) {
			for (int d = 0; d < _dimensions; d++) {
				numerators[i * _dimensions + d] = 0.0;
				for (int j = 0; j < num_examples; j++) {
					numerators[i*_dimensions + d] += H[j*_width*_height + i] * trainData[j*_dimensions + d];
				}
			}
		}

		// Update codebook
		#pragma omp parallel for
		for (int i = 0; i < _width * _height; i++) {
			for (int d = 0; d < _dimensions; d++) {
				this->_weights[i*_dimensions + d] = numerators[i*_dimensions + d]/denominators[i];
			}
		}
	}

	free(D);
	free (m_sq);
	free (x_sq);
	free(H);
	free(BMUs);
	free(numerators);
	free(denominators);
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