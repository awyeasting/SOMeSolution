#include <limits>

class SOM
{
public:
	SOM(int width, int height);

	void train_data(double *trainData[], int num_examples, int dimensions, int iterations, double initial_learning_rate);
	
	void load_weights(std::string path_to_weights_file);
	void save_weights(std::string path_to_weights_file);

private:

	int _width;
	int _height;
	int _dimensions;
	double*** _weights;

	void normalizeData(double **trainData, int num_exampless);
	void updateNodeWeights(int x, int y, double* example, double learning_rate, double influence);
	double* randWeight(int numFeatures);
	double EucDist(double* v1, double* v2);
};