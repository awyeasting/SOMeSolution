/*
 * This file is part of SOMeSolution.
 *
 * Developed for Pacific Northwest National Laboratory.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the BSD 3-Clause License as published by
 * the Software Package Data Exchange.
 */

#include "SOM.h"

#include <mpi.h>

//----------------------------------------------------
//	public SOM functions
//----------------------------------------------------

/* 
	Construct untrained SOM with given lattice width and height
*/
SOM::SOM(unsigned int width, unsigned int height){

    MPI_Group group;
    MPI_Comm_group(MPI_COMM_WORLD, &group);

	MPI_Comm_rank(MPI_COMM_WORLD, &this->_rank);
    MPI_Group_rank(group, &this->_groupRank);
	MPI_Comm_size(MPI_COMM_WORLD, &this->_numProcs);
    MPI_Group_size(group, &this->_numGroupProcs);

	this->_width = width;
	this->_height = height;
}

/*
	Construct SOM from a saved SOM width, height, and set of weights
*/
SOM::SOM(std::istream &in) {

    MPI_Group group;
    MPI_Comm_group(MPI_COMM_WORLD, &group);

	MPI_Comm_rank(MPI_COMM_WORLD, &this->_rank);
    MPI_Group_rank(group, &this->_groupRank);
	MPI_Comm_size(MPI_COMM_WORLD, &this->_numProcs);
    MPI_Group_size(group, &this->_numGroupProcs);

	this->loadWeights(in);
}

/*
	Generates a random set of training data if there is no input file given
*/
void SOM::gen_train_data(unsigned int num_examples, unsigned int dimensions, unsigned int seedValue)
{
	this->_dimensions = dimensions;
	// TODO: Switch to compute based examples distribution
	this->_numExamples = num_examples / this->_numProcs;
	this->_trainData = new float [this->_numExamples * this->_dimensions];
	srand(seedValue + this->_rank);
	for (int i = 0; i < this->_numExamples; i++)
	{
		int rowMod = (this->_numExamples - i - 1) * this->_dimensions;
		for (int d = 0; d < this->_dimensions; d++)
		{
			float weight = SOM::randWeight();
			this->_trainData[rowMod + d] = weight;
		}
	}
}

/*
	Load a set of training data from a given filename

	Precondition: File is already open
*/
bool SOM::load_train_data(std::string &fileName, bool hasLabelRow, bool hasLabelColumn) {
	unsigned int cols = 0, rows = 0;
	bool okOpen = true;
	if (this->_rank == 0) {
		// Open file for counting number of rows and columns
		std::ifstream infile(fileName, std::ifstream::in);
		if (!infile.is_open()) {
			okOpen = false;
        } else {
			// Read in first row of data into line
			std::string line;
			if (hasLabelRow) {
				std::getline(infile, line);
				rows++;
			}
			std::getline(infile, line);
			rows++;

			// Count number of values in the first row to determine num columns
			// TODO: make this work with non number labels
			std::stringstream ss(line);
			float temp;
			while (ss >> temp) {
				cols++;
			}
			while(std::getline(infile, line)) {
				// Ignore empty lines to be more forgiving of minor formatting mistakes
				if (line.compare("") != 0)
					rows++;
			}
		}
		infile.close();
	}
	// Check there was no problem with the file
	MPI_Bcast(&okOpen, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
	if (!okOpen) {
		return false;
	}

	// Broadcast rows and columns
	MPI_Bcast(&rows, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
	MPI_Bcast(&cols, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	this->_numExamples = rows - ((unsigned int)hasLabelRow);
	this->_dimensions = cols - ((unsigned int)hasLabelColumn);
	
	// Initialize feature maximums and minimums
	this->_featureMaxes = (float *)malloc(sizeof(float) * this->_dimensions);
	this->_featureMins = (float*)malloc(sizeof(float) * this->_dimensions);
	for(int i =0; i < this->_dimensions; i++){
		this->_featureMaxes[i] = -std::numeric_limits<float>::max();
		this->_featureMins[i] = std::numeric_limits<float>::max();
	}

	// Calculate starting position
	int read_count = this->_numExamples / this->_numProcs;
	int startRow = ((this->_numExamples / this->_numProcs) * this->_rank);
	// Adjust for remainder of examples
	if (this->_rank == this->_numProcs - 1) {
		read_count += (this->_numExamples % this->_numProcs);
	}
	this->_numExamples = read_count;

	// Prepare for reading in assigned chunk
	bool readOk = true;
	std::fstream infile(fileName, std::ifstream::in); // Assume because it opened for rank 0 it will open for all
    std::fstream& procfile = GotoLine(infile, startRow);
	this->_trainData = (float *)malloc(this->_numExamples * this->_dimensions * sizeof(float));

	// Read in assigned portion
	int procSectionLineNum = 0;
	std::string line;
	while(procSectionLineNum < read_count && std::getline(procfile, line)) {
        if (line.compare("") != 0) {
			std::stringstream ss(line);
			float temp;
			int cols_count = 0;
			// Read line into train data
			while (ss >> temp && cols_count < this->_dimensions) {
				this->_trainData[procSectionLineNum * this->_dimensions + cols_count] = temp;
				if (temp > this->_featureMaxes[cols_count]) {
					this->_featureMaxes[cols_count] = temp;
				}
				if (temp < this->_featureMins[cols_count]) {
					this->_featureMins[cols_count] = temp;
				}
				cols_count++;
			}
			// If the line finished reading early then the data is not of a consistent dimension
			if (cols_count != this->_dimensions) {
				readOk = false;
			}
			procSectionLineNum++;
		}
		if (!readOk)
			break;
	}
	// If it didn't read enough lines then the data is not properly formatted
	if (procSectionLineNum != this->_numExamples) {
		readOk = false;
		return false;
	}

	// Check that all the threads read their data properly
	bool allReadOk;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(&readOk, &allReadOk, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
	// If any process failed to read the process report it and discharge any allocated memory
	if (!allReadOk) {
		destroy_train_data();
		std::cout << "Error reading input file: Unable to read input file, check to make sure the input is properly formatted" << std::endl;
		return false;
	}

	// Find the true feature maxes and true feature mins
	float *globalMaxes = (float *)malloc(sizeof(float) * this->_dimensions);
	float *globalMins = (float*)malloc(sizeof(float) * this->_dimensions);
	MPI_Allreduce(this->_featureMaxes, globalMaxes, this->_dimensions, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
	MPI_Allreduce(this->_featureMins, globalMins, this->_dimensions, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);

	// Free pre reduction maxes and mins
	free(this->_featureMaxes);
	free(this->_featureMins);
	this->_featureMaxes = globalMaxes;
	this->_featureMins = globalMins;

	return true;
}

void SOM::destroy_train_data() {
	free(this->_trainData);
	free(this->_featureMaxes);
	free(this->_featureMins);

	this->_trainData = NULL;
	this->_featureMaxes = NULL;
	this->_featureMins = NULL;
}

void SOM::train_data(unsigned int epochs, unsigned int map_seed, int num_gpus) {
    this->_numEpochs = epochs;
    this->_mapSeed = map_seed;

    this->_numGPUs = num_gpus;
    this->_gpus = NULL;

    this->trainData();
}

void SOM::train_data(unsigned int epochs, unsigned int map_seed, int num_gpus, int gpu_num_offset) {
    this->_numEpochs = epochs;
    this->_mapSeed = map_seed;

    this->_numGPUs = num_gpus;
    this->_gpus = (int *)malloc(this->_numGPUs * sizeof(float));
    for(int i = 0; i < this->_numGPUs; i++) {
        this->_gpus[i] = i + gpu_num_offset;
    }

    this->trainData();
}

void SOM::train_data(unsigned int epochs, unsigned int map_seed, int num_gpus, int* gpus_assigned) {
    this->_numEpochs = epochs;
    this->_mapSeed = map_seed;

    this->_numGPUs = num_gpus;
    this->_gpus = (int *)malloc(this->_numGPUs * sizeof(float));
    for(int i = 0; i < this->_numGPUs; i++) {
        this->_gpus[i] = gpus_assigned[i];
    }

    this->trainData();
}

int SOM::get_num_gpus() {
	return this->_numGPUs;
}

/*
	Train the SOM using a set of training data over a given number of epochs with a given learning rate
*/
void SOM::trainData(){
	// Check that the training data has been loaded in
	if (this->_trainData == NULL) {
		std::cout << "Train data not yet initialized in SOM" << std::endl;
		return;
	}

	this->_mapSize = this->_width * this->_height;

    normalizeData(this->_trainData);
	// Establish multi gpu setup on current node
	// TODO: Add num gpus option
	initMultiGPUSetup();

	this->_initial_map_radius = this->_width < this->_height ? ((float)this->_width) / 2.0 : ((float)this->_height) / 2.0;
	this->_time_constant = float(this->_numEpochs) / log(this->_initial_map_radius);

	this->_currentEpoch = 0;
	#pragma omp parallel 
	{
		int gpu = omp_get_thread_num();
		for(int curEpoch = 0; curEpoch < this->_numEpochs; curEpoch++) {
			if (gpu == 0) {
				// Set the current epoch for the whole proc's SOM on the first thread
				this->_currentEpoch = curEpoch;
				// Calculate current neighborhood radius
				this->_neighborhood_radius = this->_initial_map_radius * exp(-((float)(this->_currentEpoch))/this->_time_constant);

				// Wait for all other nodes to start the epoch
				MPI_Barrier(MPI_COMM_WORLD);

				// Send out the map on proc 0
				MPI_Bcast(this->_weights, this->_mapSize * this->_dimensions, MPI_FLOAT, 0, MPI_COMM_WORLD);
			}
			// Wait for proc allocated to first gpu to have correct map
			#pragma omp barrier

			// Update gpu copies of the map
			updateGPUCodebook(gpu);

			// Train a single epoch on all chosen gpus
			trainOneEpochOneGPU(gpu);

			#pragma omp barrier

			// Reduce numerators and denominators across gpus on proc
			int step = 0;
			while((1 << step) < this->_numGPUs) {
				if (gpu % (1 << (step + 1)) == 0) {
					int otherGPU = gpu + (1 << step);
					if (otherGPU < this->_numGPUs) {
						for (int i = 0; i < this->_mapSize; i++) {
							this->_gdenom[gpu][i] += this->_gdenom[otherGPU][i];
							for (int d = 0; d < this->_dimensions; d++) {
								this->_gnumer[gpu][d*this->_mapSize + i] += this->_gnumer[otherGPU][d*this->_mapSize + i];
							}
						}
					}
				}
				#pragma omp barrier
				step++;
			}
			if (gpu == 0) {
				memcpy(this->_denom, this->_gdenom[0], this->_mapSize * sizeof(float));
				memcpy(this->_numer, this->_gnumer[0], this->_mapSize * this->_dimensions * sizeof(float));

				// Reduce numerators and denominators across all procs
				MPI_Reduce(this->_numer, this->_global_numer, this->_mapSize * this->_dimensions, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
				MPI_Reduce(this->_denom, this->_global_denom, this->_mapSize, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
				
				// Update codebook/map
				if (this->_rank == 0) {
					// Recalculate weights with new numerators and denominators
					for (int i = 0; i < this->_mapSize; i++) {
						for (int d = 0; d < this->_dimensions; d++) {
							this->_weights[d*this->_mapSize + i] = this->_numer[d*this->_mapSize + i] / this->_denom[i];
						}
					}
				}
			}
		}
	}

	// Perform column major to row major order on weights matrix
	float *tempWeights = (float *)malloc(this->_mapSize * this->_dimensions * sizeof(float));
	for (int i = 0; i < this->_mapSize; i++) {
		for (int d = 0; d < this->_dimensions; d++) {
			tempWeights[i*this->_dimensions + d] = this->_weights[d*this->_mapSize + i];
		}
	}
	free(this->_weights);
	this->_weights = tempWeights;

	freeGPUMemory();

	free(this->_GPU_EXAMPLES);
	free(this->_GPU_OFFSET);
	free(this->_handles);
	free(this->_d_train);
	free(this->_d_weights);
	free(this->_d_numer);
	free(this->_d_denom);
	free(this->_gnumer);
	free(this->_gdenom);
	free(this->_numer);
	free(this->_denom);
	free(this->_global_numer);
	free(this->_global_denom);
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
				out << this->_weights[calcIndex(i,j,k)];
			}
			out << std::endl;
		}
	}
}

std::fstream& SOM::GotoLine(std::fstream& file, unsigned int num){
    file.seekg(std::ios::beg);
    for(int i=0; i < num; i++){
        file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    }
    return file;
}

//----------------------------------------------------
//	private SOM functions
//----------------------------------------------------

/*
	Load a trained SOM that was saved using the same algorithm as save_weights from an input stream
*/
void SOM::loadWeights(std::istream &in)
{
	// Load SOM dimensions first
	in >> this->_width >> this->_height;

	// Read first line of matrix to get the dimensionality of weights
	this->_dimensions = 0;
	std::string line;
	std::getline(in, line);
	std::getline(in, line);
	std::stringstream ss(line);
	std::vector<float> line1;
	float temp;
	while (ss >> temp) {
		this->_dimensions++;
		line1.push_back(temp);
	}

	// Put first line of matrix into an array in the 3d weights array
	this->_weights = new float[_width * _height * _dimensions];
	for (int k = 0; k < this->_dimensions; k++) {
		_weights[calcIndex(0,0,_dimensions - k - 1)] = line1.back();
		line1.pop_back();
	}

	// Read the rest of the 3d array in
	for (int i = 0; i < this->_width; i++) {
		for (int j = (i == 0 ? 1 : 0); j < this->_height; j++) {
			for (int k = 0; k < _dimensions; k++) {
				in >> this->_weights[calcIndex(i,j,k)];
			}
		}
	}
}

/*
	Normalizes given data to be between 0 and 1 for each feature
*/
void SOM::normalizeData(float *trainData)
{
	// Find the max and min value for each feature then use it to normalize the feature
	this->_featureMaxes = new float[this->_dimensions];
	this->_featureMins = new float[this->_dimensions];
	for (int d = 0; d < this->_dimensions; d++)
	{
		this->_featureMaxes[d] = -std::numeric_limits<float>::max();
		this->_featureMins[d] = std::numeric_limits<float>::max();
		for (int i = 0; i < this->_numExamples; i++)
		{
			if (trainData[i*this->_dimensions + d] > this->_featureMaxes[d]) {
				this->_featureMaxes[d] = trainData[i*_dimensions + d];
			}
			if (trainData[i*this->_dimensions + d] < this->_featureMins[d]) {
				this->_featureMins[d] = trainData[i*this->_dimensions + d];
			}
		}
		for (int i = 0; i < this->_numExamples; i++) {
			if ((this->_featureMaxes[d] - this->_featureMins[d]) <= std::numeric_limits<float>::min())
			{
				trainData[i*_dimensions + d] = 0;
			}
			else {
				trainData[i*_dimensions + d] = (trainData[i*_dimensions + d] - this->_featureMins[d])/(this->_featureMaxes[d]-this->_featureMins[d]);
			}
		}
	}
}

/*
	Calculate the index of a weight at node (x,y), dimension = d in the weights array
*/
int SOM::calcIndex(int x, int y, int d) {
	return (x*_height + y)*_dimensions + d;
}

float SOM::randWeight()
{
	return (float)rand() / (RAND_MAX);
}