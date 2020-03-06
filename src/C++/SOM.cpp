#include "pch.h"

/* SOM Constructor
Desc: Intitializes a SOM with arguments:
	width - int width of map
	height - int height of map
	hex - bool optional ability to intialize hexagonal map. [HEX OPTION NOT IMPLEMENTED NEED]

*/
SOM::SOM(int width, int height)
{
	_width = width;
	_height = height;

	// Initialize map randomly
	_weights = new double**[_width];
	for (int i = 0; i < _width; i++)
	{
		_weights[i] = new double*[height];
	}
}

/*
	The main function for training. takes in the training data with a pointer to an array of doubles.
	also takes in the arguments of total number of iterations and initial learning rate.
*/
void SOM::train_data(double *trainData[], int num_examples, int dimensions, int iterations, double initial_learning_rate)
{
	// Randomly initialize weights
	_dimensions = dimensions;
	for (int i = 0; i < _width; i++) {
		for (int j = 0; j < _height; j++) {
			double* randWeights = randWeight(_dimensions);
			_weights[i][j] = randWeights;
		}
	}

	int iterations_counter = 0;
	double current_learning = 0.0;
	double neighborhood_radius = 0.0;
	double initial_map_radius = _width < _height ? ((double)_width) / 2.0 : ((double)_height) / 2.0;

	double time_constant = double(iterations) / log(initial_map_radius);

	normalizeData(trainData, num_examples);

	//Main Training Loop
	while (iterations_counter < iterations)
	{
		current_learning = initial_learning_rate * exp(-double(iterations_counter) / time_constant);

		int train_exam = 0;
		//For each example in our training set.
		while (train_exam < num_examples)
		{
			int bmu_x, bmu_y;
			double bmu_dist = DBL_MAX;

			for (int i = 0; i < _width; i++)
			{
				for (int j = 0; j < _height; j++)
				{
					double temp_dist = EucDist(_weights[i][j], trainData[train_exam]);
					if (temp_dist < bmu_dist)
					{
						bmu_dist = temp_dist;
						bmu_x = i;
						bmu_y = j;
					}
				}
			}

			neighborhood_radius = initial_map_radius * exp(-(double(iterations_counter) / time_constant));

			for (int i = 0; i < _width; i++)
			{
				for (int j = 0; j < _height; j++)
				{
					//Loops through every node in the array and calculates the euclidean squared distance away
					double euclid_away = (i - bmu_x) *
						(i - bmu_x) +
						(j - bmu_y) *
						(j - bmu_y);


					double widthSq = neighborhood_radius * neighborhood_radius;

					//Compares the squared euclid distance with the current iteration radius squared. 
					//If the euclidean dist away is less than neighborhood squared, calculate influence and update
					//the nodes weights.
					//if (euclid_away < widthSq)
					//{
					double influence = exp(-(euclid_away) / (2 * widthSq));
					updateNodeWeights(i, j, trainData[train_exam], current_learning, influence);
					//}
				}
			}
			train_exam++;
		}

		iterations_counter++;

	}
}

void SOM::load_weights(std::string path_to_weights_file)
{

}

//Saves the weights with the first line of the file being the width seperated by a space and then the height
//Each subsequent row is a node's weight. Right Now it's hardcoded to save 
void SOM::save_weights(std::string path_to_weights_file)
{
	std::string temp = path_to_weights_file;
	std::ofstream out;
	out.open(path_to_weights_file);
	out << _width << " " << _height << std::endl;
	for (int i = 0; i < _width; i++)
	{
		for (int j = 0; j < _height; j++)
		{
			for (int k = 0; k < _dimensions; k++) {
				if (k != 0) {
					out << " ";
				}
				out << _weights[i][j][k];
			}
			out << std::endl;
		}
	}
	out.close();
}

/*
	Normalizes the data. Right now hardcoded to normalize color data.
	In future will be changed to normalize to normalize by feature.
*/
void SOM::normalizeData(double **trainData, int num_examples)
{
	for (int i = 0; i < num_examples; i++)
	{
		for (int j = 0; j < _dimensions; j++)
		{
			trainData[i][j] = trainData[i][j] / 255;
		}
	}
}

void SOM::updateNodeWeights(int x, int y, double* example, double learning_rate, double influence) {
	for (int k = 0; k < _dimensions; k++)
	{
		_weights[x][y][k] += influence * learning_rate * (example[k] - _weights[x][y][k]);
	}
}

//Returns a vector of size, numFeatures, with values between 0 and 1.
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
	for (int i = 0; i < _dimensions; i++) {
		total += (v1[i] - v2[i])*(v1[i] - v2[i]);
	}
	return sqrt(total);
}