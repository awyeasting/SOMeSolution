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
void SOM::train_data(double *trainData[], unsigned int num_examples, unsigned int dimensions, int epochs, double initial_learning_rate)
{
	normalizeData(trainData, num_examples);

	// Randomly initialize codebook
	this->_dimensions = dimensions;
	this->_weights = (double *)malloc(_width * _height * _dimensions * sizeof(double));
	for (int i = 0; i < _width; i++) {
		for (int j = 0; j < _height; j++) {
			for (int d = 0; d < _dimensions; d++) {
				this->_weights[calcIndex(i,j,d)] = randWeight();
			}
		}
	}

	double initial_map_radius = _width < _height ? ((double)_width) / 2.0 : ((double)_height) / 2.0;
	double time_constant = double(epochs) / log(initial_map_radius);

	// Loop through all epochs
	double learning_rate, neighborhood_radius;
	for (int epoch = 0; epoch < epochs; epoch++) {
		learning_rate = initial_learning_rate * exp(-double(epoch) / time_constant);
		neighborhood_radius = initial_map_radius * exp(-(double(epoch) / time_constant));

		//Loop through all training data points
		for (int example = 0; example < num_examples; example++) {
			//Find the BMU for the current example
			int bmu_x, bmu_y;
			double bmu_dist = std::numeric_limits<double>::max();
			
			for (int i = 0; i < _width; i++)
			{
				for (int j = 0; j < _height; j++)
				{
					double temp_dist = EucDist(_weights + calcIndex(i,j,0), trainData[example]);
					if (temp_dist < bmu_dist)
					{
						bmu_dist = temp_dist;
						bmu_x = i;
						bmu_y = j;
					}
				}
			}

			// Update weights for every node
			#pragma omp parallel for
			for (int i = 0; i < _width; i++)
			{
				for (int j = 0; j < _height; j++)
				{
					// Calculate the euclidean distance of the current node from the BMU
					double euclid_away = (i - bmu_x) *
						(i - bmu_x) +
						(j - bmu_y) *
						(j - bmu_y);

					// Update the node's weight using gaussian curve
					double influence = exp(-(euclid_away) / (2 * neighborhood_radius * neighborhood_radius));
					updateNodeWeights(i, j, trainData[example], learning_rate, influence);
				}
			}
		}
	}
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
void SOM::normalizeData(double **trainData, int num_examples)
{
	// Find the max and min value for each feature then use it to normalize the feature
	this->_featureMaxes = new double[this->_dimensions];
	this->_featureMins = new double[this->_dimensions];
	for (int j = 0; j < this->_dimensions; j++)
	{
		this->_featureMaxes[j] = -std::numeric_limits<double>::max();
		this->_featureMins[j] = std::numeric_limits<double>::max();
		for (int i = 0; i < num_examples; i++)
		{
			if (trainData[i][j] > this->_featureMaxes[j]) {
				this->_featureMaxes[j] = trainData[i][j];
			}
			if (trainData[i][j] < this->_featureMins[j]) {
				this->_featureMins[j] = trainData[i][j];
			}
		}
		for (int i = 0; i < num_examples; i++) {
			trainData[i][j] = (trainData[i][j] - this->_featureMins[j])/this->_featureMaxes[j];
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