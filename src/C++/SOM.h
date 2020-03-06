#include "Node.h"
#include <limits>

class SOM
{
public:
	int _width;
	int _height;
	bool _hex = false;
	int _n_dimensions;
	Node ** two_dimension_map;
	void train_function(std::string path_to_data, std::string path_to_param);
	void load_weights(std::string path_to_weights_file);
	void save_weights(std::string path_to_weights_file);
	SOM(int width, int height, bool hex, int numFeatures);
	void train_data(double *trainData[], int num_examples, int iterations, double initial_learning_rate);
	double* randWeight(int numFeatures);
	void normalizeData(double *trainData[], int num_examples);
};