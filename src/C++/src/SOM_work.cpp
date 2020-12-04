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

int main(int argc, char *argv[])
{
	MPI::Init(argc,argv);
	
	char *trainingFileName = new char[100];
	trainingFileName = "";
	std::string outFileName = "weights.txt";
	std::string versionNumber = "0.4.0";
	int epochs = 10;
	unsigned int width = 8, height = 8;
	double learningRate = 0.1;
	unsigned int n, d, seed;
	unsigned int* seedArray = new unsigned int[num_procs];
	unsigned int map_seed = time(NULL);
	bool hasLabelColumn = false;
	
	trainingFileName = (char *)malloc(sizeof(char) * 100);

	// Load program arguments on rank 0
	int posArgPos = 0;
	for(int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			std::cout << "Positional Arguments:" << std::endl
			<< "\t(int)    SOM width" << std::endl
			<< "\t(int)    SOM height" << std::endl
			<< "\t(string) Training data" << std::endl;
			std::cout << "Options:" << std::endl
			<< "\t(int int)-g --generate  num features, num_dimensions for generating random data" << std::endl
			<< "\t(string) -o --out       Path of the output file of node weights" << std::endl
			<< "\t(int)    -e --epochs    Number of epochs used in training" << std::endl
			<< "\t(int)    -s --seed      Integer value to intialize seed for generating" << std::endl
			<< "\t         -l --labeled   Indicates the last column is a label" <<std::endl;
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
					int lengthOf = strlen(argv[i]);
					strcpy(trainingFileName, argv[i]);
					break;
				default:
					std::cout << "Unrecognized positional argument, '" << argv[i] << "'" << std::endl;
			}
			posArgPos++;
		}
	}
	
	// Broadcast the rows_count, dimensions, and epochs that are all handled from the command line. 
	//MPI_Barrier(MPI::COMM_WORLD);
	//MPI_Bcast(&n, 1, MPI::UNSIGNED, 0, MPI::COMM_WORLD);
	//MPI_Bcast(&d, 1, MPI::UNSIGNED, 0, MPI::COMM_WORLD);
	//MPI_Bcast(&epochs, 1, MPI::UNSIGNED, 0, MPI::COMM_WORLD);
	//MPI_Bcast(&width, 1, MPI::UNSIGNED, 0, MPI::COMM_WORLD);
	//MPI_Bcast(&height, 1, MPI::UNSIGNED, 0, MPI::COMM_WORLD);
	//MPI_Bcast(&map_seed, 1, MPI::UNSIGNED, 0, MPI::COMM_WORLD);
	//MPI_Bcast(&column_label, 1, MPI::BOOL, 0, MPI::COMM_WORLD);
	
	//MPI_Bcast(trainingFileName, 100, MPI::CHAR, 0, MPI::COMM_WORLD);
	//MPI_Scatter(seedArray, 1, MPI::UNSIGNED, &seed, 1, MPI::UNSIGNED, 0, MPI::COMM_WORLD);

	//std::cout << "fileName " << (std::string)trainingFileName << std::endl;
	
	// Create untrained SOM
	SOM newSom = SOM(width, height);

	if(fileSize <= 0) {
		newSom.gen_train_data(n, d, map_seed)
	} else {
		std::fstream trainDataFile(fileName, std::ios::in | std::ios::out);

		if (!trainDataFile.is_open()) {
			std::cout << "Invalid training data file '" << fileName << "'" << std::endl;
		}
		newSom.load_train_data(trainDataFile, hasLabelColumn);
	}

	// Train SOM and time training
	auto start = std::chrono::high_resolution_clock::now();
	newSom.train_data(epochs, learningRate, map_seed);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
	std::cout << "Finished training in " << duration.count() << "seconds" << std::endl;

	// Save the SOM's weights on rank 0
	if (rank == 0)
	{
		std::ofstream outFile(outFileName, std::ofstream::out);
		if (outFile.is_open()) {
			newSom.save_weights(outFile);
			outFile.close();
			//std::cout << "SOM saved to " << outFileName << std::endl;
		}
	}
	
	MPI::Finalize();
}