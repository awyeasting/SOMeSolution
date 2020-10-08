#ifndef SOM_H
#define SOM_H

#include <limits>
#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "cblas.h"
//Include path to blas and lapack libraries
class SOM
{
public:
	SOM(unsigned int width, unsigned int height);
	SOM(std::istream &in);

	void train_data(double *trainData, unsigned int num_examples, unsigned int dimensions, int epochs, double initial_learning_rate);
	static double randWeight();
	void save_weights(std::ostream &out);

private:

	unsigned int _spacing;
	unsigned int _width;
	unsigned int _height;
	unsigned int _dimensions;
	double* _weights;
	double* _featureMaxes;
	double* _featureMins;
	double* _zeroes;	//zeroes and ones will be vectors of dimensions size.
	double* _ones;	   //These vectors will be used for daxpy operations with uneeded vector or addition or subtraction   
	double* _buffer;  //This vector will act as a variable workspace so blas won't alter input vectors 
	void load_weights(std::istream &in);

	void normalizeData(double *trainData, int num_exampless);
	void updateNodeWeights(int x, int y, double* example, double learning_rate, double influence);
	int calcIndex(int x, int y, int d);
	double EucDist(double* v1, double* v2);
	static void SqDists(double* m, int loop, int dim, double* output);
	double h(int i, int j, double initial_radius, double radius, int* D);
};

#endif
