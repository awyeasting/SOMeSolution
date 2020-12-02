/*
 * This file is part of SOMeSolution.
 *
 * Developed for Pacific Northwest National Laboratory.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the BSD 3-Clause License as published by
 * the Software Package Data Exchange.
 */

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

#define GPU_BASED_CODEBOOK_INIT true

void trainOneEpoch(cublasHandle_t &handle, int device, double *train, double *weights, double *D, double *m_sq, double *x_sq, int *BMUs, double *H, double *numer, double *denom, int width, int height, int num_examples, int dimensions, double initial_map_radius, double neighborhood_radius);

class SOM
{
public:
	SOM(unsigned int width, unsigned int height);
	SOM(std::istream &in);

	void train_data(double *trainData, unsigned int num_examples, unsigned int dimensions, int epochs, double initial_learning_rate);
	void save_weights(std::ostream &out);

private:

	unsigned int _width;
	unsigned int _height;
	unsigned int _mapSize;
	unsigned int _dimensions;
	double* _weights;
	double* _featureMaxes;
	double* _featureMins;

	void loadWeights(std::istream &in);
	void normalizeData(double *trainData, int num_exampless);
	int calcIndex(int x, int y, int d);

	void initMultiGPUSetup(int &ngpus);
	void initNumDenom(double *&numer, double *&denom);
	void initGPUTrainData(const int ngpus, double *trainData, double **d_train, int *GPU_EXAMPLES, int *GPU_OFFSET);
	void initGPUTrainMemory(const int ngpus, cublasHandle_t *&handles, double **&d_train, double **&d_weights, double **&d_numer, double **&d_denom, int *&GPU_EXAMPLES, int *&GPU_OFFSET, int num_examples);
	void initGPUNumDenReducMem(const int ngpus, double **&gnumer, double **&gdenom);
	void initCodebook();
	void initCodebookOnGPU(double **d_weights);
	void initGPUCodebooks(double **d_weights);

	static double randWeight();
};

#endif