#ifndef SOM_H
#define SOM_H

#include <limits>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <omp.h>
#include <time.h>
#include "mpi.h"

class SOM
{
public:
	SOM(unsigned int width, unsigned int height);
	SOM(std::istream &in);

	void train_data(std::string fileName, unsigned int current_rank, unsigned int num_procs, unsigned int epochs, unsigned int dimensions, unsigned int rowCount, int rank_seed, unsigned int map_seed);
	void train_one_epoch(double* localMap, double* train_data, double* numerators, double* denominators, int num_examples, double initial_map_radius, int epoch, double time_constant);
	static double randWeight();
	void save_weights(std::ostream &out);
	double* generateRandomTrainingInputs(unsigned int examples, unsigned int dimensions, int seedValue);
	double* loadTrainingData(std::fstream& in, unsigned int& rows, unsigned int& cols, int read_count, double* featureMaxes, double* featureMins);
	std::fstream& GotoLine(std::fstream& file, unsigned int num);
private:

	unsigned int _width;
	unsigned int _height;
	unsigned int _dimensions;
	double* _weights;
	double* _featureMaxes;
	double* _featureMins;

	void load_weights(std::istream &in);

	void normalizeData(double *trainData, int num_exampless, double *max, double *min);
	void updateNodeWeights(int x, int y, double* example, double learning_rate, double influence);
	int calcIndex(int x, int y, int d);
	double EucDist(double* v1, double* v2);
	static void SqDists(double* m, int loop, int dim, double* output);
	double h(int i, int j, double initial_radius, double radius, int* D);
};

#endif
