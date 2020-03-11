#ifndef SOM_H
#define SOM_H

#include <limits>
#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>
#include <omp.h>

class SOM
{
public:
	SOM(unsigned int width, unsigned int height);
	SOM(std::istream &in);

	void train_data(double *trainData[], unsigned int num_examples, unsigned int dimensions, int epochs, double initial_learning_rate);
	
	void save_weights(std::ostream &out);

private:

	unsigned int _width;
	unsigned int _height;
	unsigned int _dimensions;
	double*** _weights;
	double* _featureMaxes;
	double* _featureMins;

	void load_weights(std::istream &in);

	void normalizeData(double **trainData, int num_exampless);
	void updateNodeWeights(int x, int y, double* example, double learning_rate, double influence);
	static double* randWeight(int numFeatures);
	double EucDist(double* v1, double* v2);
};

#endif