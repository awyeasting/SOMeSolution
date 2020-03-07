#include "SOM.h"

/* 
	Construct untrained SOM with given lattice width and height
*/
SOM::SOM(unsigned int width, unsigned int height)
{
	this->_width = width;
	this->_height = height;

	// Initialize map randomly
	this->_weights = new double**[_width];
	for (int i = 0; i < _width; i++)
	{
		this->_weights[i] = new double*[height];
	}
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
	// Randomly initialize weights
	this->_dimensions = dimensions;
	for (int i = 0; i < _width; i++) {
		for (int j = 0; j < _height; j++) {
			double* randWeights = SOM::randWeight(_dimensions);
			this->_weights[i][j] = randWeights;
		}
	}

	double initial_map_radius = _width < _height ? ((double)_width) / 2.0 : ((double)_height) / 2.0;
	double time_constant = double(epochs) / log(initial_map_radius);

	normalizeData(trainData, num_examples);

	// Loop through all epochs
	double learning_rate, neighborhood_radius;
	for (int epoch = 0; epoch < epochs; epoch++) {
		learning_rate = initial_learning_rate * exp(-double(epoch) / time_constant);
		neighborhood_radius = initial_map_radius * exp(-(double(epoch) / time_constant));

		// Loop through all training data points
		for (int example = 0; example < num_examples; example++) {
			// Find the BMU for the current example
			int bmu_x, bmu_y;
			double bmu_dist = DBL_MAX;
			for (int i = 0; i < _width; i++)
			{
				for (int j = 0; j < _height; j++)
				{
					double temp_dist = EucDist(_weights[i][j], trainData[example]);
					if (temp_dist < bmu_dist)
					{
						bmu_dist = temp_dist;
						bmu_x = i;
						bmu_y = j;
					}
				}
			}

			// Update weights for every node
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
				out << this->_weights[i][j][k];
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

	// Generate 3d array for storing the node weights
	this->_weights = new double**[this->_width];
	for (int i = 0; i < _width; i++) {
		_weights[i] = new double*[this->_height];
	}

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
	this->_weights[0][0] = new double[this->_dimensions];
	for (int k = 0; k < this->_dimensions; k++) {
		_weights[0][0][this->_dimensions - k - 1] = line1.back();
		line1.pop_back();
	}

	// Read the rest of the 3d array in
	for (int i = 0; i < this->_width; i++) {
		for (int j = (i == 0 ? 1 : 0); j < this->_height; j++) {
			this->_weights[i][j] = new double[this->_dimensions];
			for (int k = 0; k < _dimensions; k++) {
				in >> this->_weights[i][j][k];
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
		this->_featureMaxes[j] = -DBL_MAX;
		this->_featureMins[j] = DBL_MAX;
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
	for (int k = 0; k < this->_dimensions; k++)
	{
		this->_weights[x][y][k] += influence * learning_rate * (example[k] - this->_weights[x][y][k]);
	}
}

/*
	Generate a vector of size numFeatures
*/
double* SOM::randWeight(int numFeatures)
{
	double* retVector = new double[numFeatures];
	double temp_rand_val;
	for (int i = 0; i < numFeatures; i++)
	{
		temp_rand_val = ((double)rand() / (RAND_MAX));
		retVector[i] = temp_rand_val;
	}
	return retVector;
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