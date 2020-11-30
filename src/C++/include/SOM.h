#ifndef SOM_H
#define SOM_H

#include <limits>
#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>
#include <time.h>
#include <omp.h>

#include <curand.h>
#include "cublas_v2.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void SqDists(double* m, int loop, int dim, double* output);
void trainOneEpoch(cublasHandle_t &handle, int device, double *train, double *weights, double *D, double *m_sq, double *x_sq, int *BMUs, double *H, double *numer, double *denom, int width, int height, int num_examples, int dimensions, double initial_map_radius, double neighborhood_radius);
double h(int j, int i, double initial_radius, double radius, int* BMUs, int height);

class SOM
{
public:
	SOM(unsigned int width, unsigned int height);
	SOM(std::istream &in);

	void train_data(double *trainData, unsigned int num_examples, unsigned int dimensions, int epochs, double initial_learning_rate);
	static double randWeight();
	void save_weights(std::ostream &out);

private:

	unsigned int _width;
	unsigned int _height;
	unsigned int _dimensions;
	double* _weights;
	double* _featureMaxes;
	double* _featureMins;

	void load_weights(std::istream &in);

	void normalizeData(double *trainData, int num_exampless);
	void updateNodeWeights(int x, int y, double* example, double learning_rate, double influence);
	int calcIndex(int x, int y, int d);
	double EucDist(double* v1, double* v2);
};

#endif