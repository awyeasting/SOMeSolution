/*
 * This file is part of SOMeSolution.
 *
 * Developed for Pacific Northwest National Laboratory.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the BSD 3-Clause License as published by
 * the Software Package Data Exchange.
 */

#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <time.h>
#include "SOM.h"

#include <mpi.h>

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	
	std::string trainingFileName = "";
	std::string outFileName = "weights.txt";
	std::string versionNumber = "0.4.0";
	int epochs = 10;
	unsigned int width = 8, height = 8;
	unsigned int n, d, seed;
	unsigned int map_seed = time(NULL);
	int gpusPerProc = -1;
	bool hasLabelColumn = false;

	// Load program arguments on rank 0
	int posArgPos = 0;
	for(int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			std::cout << "Positional Arguments:" << std::endl
			<< "\t(int)    SOM width" << std::endl
			<< "\t(int)    SOM height" << std::endl
			<< "\t(string) Training data" << std::endl;
			std::cout << "Options:" << std::endl
			<< "\t(int int)-g --generate       num features, num_dimensions for generating random data" << std::endl
			<< "\t(string) -o --out            Path of the output file of node weights" << std::endl
			<< "\t(int)    -e --epochs         Number of epochs used in training" << std::endl
			<< "\t(int)    -s --seed           Integer value to intialize seed for generating" << std::endl
			<< "\t         -l --labeled        Indicates the last column is a label" <<std::endl
			<< "\t(int)    -gp --gpus-per-proc The number of gpus each processor should utilize" << std::endl;
			return 0;
		} else if (strcmp(argv[i], "--version") == 0 || strcmp(argv[i], "-v") == 0) {
			std::cout << "somesolution v" << versionNumber << std::endl;
			return 0;
		} else if (strcmp(argv[i], "--out") == 0 || strcmp(argv[i], "-o") == 0) {
			if (i + 1 < argc) {
				outFileName = std::string(argv[i+1]);
				i++;
			} else {
				std::cout << "If the --out option is used a valid outfile path should be specified." << std::endl;
			}
		} else if (strcmp(argv[i], "--epochs") == 0 || strcmp(argv[i], "-e") == 0) {
			if (i + 1 < argc) {
				try {
					epochs = std::stoi(argv[i+1]);
				} catch (int e) {
					std::cout << "Invalid epochs argument." << std::endl;
				}
				i++;
			} else {
				std::cout << "If the --epochs option is used a valid number of epochs should be specified." << std::endl;
			}
		}
		else if (strcmp(argv[i], "--generate") == 0 || strcmp(argv[i], "-g") == 0) {
			if (i + 2 < argc)
			{
				n = std::stoi(argv[i + 1]);
				d = std::stoi(argv[i + 2]);
				i = i+ 2;
			} else {
				std::cout << "If the --generate option is used, n examples and d dimensions should be specified." << std::endl;
			}
		}
		else if (strcmp(argv[i], "--seed") == 0 || strcmp(argv[i], "-s") == 0){
			if (i + 1 < argc) {
				map_seed = std::stoi(argv[i+1]);
				i++;
			}
			else {
				std::cout << "If the --seed option is used, the following argument should be an integer argument" << std::endl;
			}
		}
		else if (strcmp(argv[i], "--labeled") == 0 || strcmp(argv[i], "-l") == 0)
		{
			hasLabelColumn = true;
		}
		else if (strcmp(argv[i], "--gpus-per-proc") == 0)
		{
			if (i + 1 < argc) {
				gpusPerProc = std::stoi(argv[i+1]);
				i++;
			}
			else {
				std::cout << "If the --gpus-per-proc option is used, the following argument should be an integer argument" << std::endl;
			}
		}
		else {
			// Positional arguments
			// width height trainingdatafile.txt
			switch(posArgPos) {
				case 0:
					try {
						width = std::stoi(argv[i]);
					} catch (int e) {
						std::cout << "Invalid width argument." << std::endl;
					}
					break;
				case 1:
					try {
						height = std::stoi(argv[i]);
					} catch (int e) {
						std::cout << "Invalid height argument." << std::endl;
					}
					break;
				case 2:
					std::cout << (std::string)argv[i] << std::endl;
					trainingFileName = std::string(argv[i]);
					break;
				default:
					std::cout << "Unrecognized positional argument, '" << argv[i] << "'" << std::endl;
			}
			posArgPos++;
		}
	}
	// Create untrained SOM
	SOM newSom = SOM(width, height);

	if(trainingFileName.length() <= 0) {
		newSom.gen_train_data(n, d, map_seed);
	} else {
		newSom.load_train_data(trainingFileName, false, hasLabelColumn);
	}

	// Train SOM and time training
	MPI_Barrier(MPI_COMM_WORLD);
	auto start = std::chrono::high_resolution_clock::now();
	newSom.train_data(epochs, map_seed, gpusPerProc);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
	double trainingTime = duration.count();
	double globalTrainingTime;
	MPI_Reduce(&trainingTime, &globalTrainingTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	// Save the SOM's weights on rank 0
	int rank;
	int numProcs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	int localNumGPUs = newSom.get_num_gpus();
	int globalNumGPUs;
	MPI_Reduce(&localNumGPUs, &globalNumGPUs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == 0)
	{
		std::cout << numProcs << "," << globalNumGPUs << "," << globalTrainingTime << std::endl;

		std::ofstream outFile(outFileName, std::ofstream::out);
		if (outFile.is_open()) {
			newSom.save_weights(outFile);
			outFile.close();
			//std::cout << "SOM saved to " << outFileName << std::endl;
		}
	}
	
	MPI_Finalize();
}