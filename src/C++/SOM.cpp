#include "pch.h"

/*
	The training function to be called with just parameters to file. 
	Under Construction. Not currently reading in inputs correctly.
*/
void SOM::train_function(std::string path_to_data, std::string path_to_param)
{
	std::ifstream instream(path_to_data);
	int num_examples = 0;
	double disc_value = 0.0;
	instream >> num_examples;
	
	double **newData = new double*[num_examples];
	for (int i = 0; i < num_examples; i++)
	{
		newData[i] = new double[3];
	}

	for (int i = 0; i < num_examples; i++)
	{
		for (int features = 0; features < 3; features++)
		{
			instream >> disc_value;
			newData[i][features] = disc_value;
		}
	}
	train_data(newData, num_examples, 10000, .01);
}

/*
	Normalizes the data. Right now hardcoded to normalize color data.
	In future will be changed to normalize to normalize by feature.
*/
void SOM::normalizeData(double *trainData[], int num_examples)
{
	for (int i = 0; i < num_examples; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			trainData[i][j] = trainData[i][j] / 255;
		}
	}
}

/*
	The main function for training. takes in the training data with a pointer to an array of doubles.
	also takes in the arguments of total number of iterations and initial learning rate.
*/
void SOM::train_data(double *trainData[], int num_examples, int iterations, double initial_learning_rate)
{
	int iterations_counter = 0;
	double current_learning = 0.0;
	double neighborhood_radius = 0.0;
	double initial_map_radius;

	if (_width > _height)
		initial_map_radius = _width / 2;
	else
		initial_map_radius = _height / 2;

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
					double temp_dist = two_dimension_map[i][j].calculateDistance(trainData[train_exam]);
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
					two_dimension_map[i][j].update_weights(trainData[train_exam], current_learning, influence, _n_dimensions);
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
	out << _width << " "<< _height << std::endl;
	for (int i = 0; i < _width; i++)
	{
		for (int j = 0; j < _height; j++)
		{
			out << two_dimension_map[i][j]._node_weights[0] << " " 
				<< two_dimension_map[i][j]._node_weights[1] << " "
				<< two_dimension_map[i][j]._node_weights[2] << std::endl;
		}
	}
	out.close();
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

/* SOM Constructor
Desc: Intitializes a SOM with arguments:
	width - int width of map
	height - int height of map
	hex - bool optional ability to intialize hexagonal map. [HEX OPTION NOT IMPLEMENTED NEED]

*/
SOM::SOM(int width, int height, bool hex, int numFeatures)
{
	_width = width;
	_height = height;
	_hex = hex;
	_n_dimensions = 3;
	//Allocates the 2d pointer of nodes with size width x height
	two_dimension_map = new Node* [width];
	for (int i = 0; i < width; i++)
	{
		two_dimension_map[i] = new Node[height];
	}

	//Intitializes the 2d matrix of nodes with their respective coordinates.
	for (int node_row = 0; node_row < width; node_row++)
	{
		for (int col = 0; col < height; col++)
		{
			double* randWeights = randWeight(numFeatures);
			two_dimension_map[node_row][col]._x_coord = col;
			two_dimension_map[node_row][col]._y_coord = node_row;
			two_dimension_map[node_row][col].set_weights(randWeights);
		}
	}
}
