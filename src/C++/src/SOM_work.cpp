#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include "SOM.h"


void countRowsAndCols(std::string fileName, unsigned int&rows, unsigned int&cols, bool label)
{
	std::ifstream in(fileName, std::ifstream::in);
	if (!in.is_open()) {
		std::cout << "Invalid training data file '" << fileName << "'" << std::endl;
		
	}
	std::string line;
	std::getline(in, line);
	std::stringstream ss(line);
	cols = 0;
	double temp;
	while (ss >> temp) {
		cols++;
	}
	if(label){
		cols--;
	}
	rows= 1;
	while (std::getline(in, line)) {
		rows++;
	}
}

int main(int argc, char *argv[])
{
	MPI::Init(argc,argv);
	unsigned int num_procs = MPI::COMM_WORLD.Get_size();
	unsigned int rank = MPI::COMM_WORLD.Get_rank();
	
	//std::string trainingFileName = "";
	char * trainingFileName = new char[100];
	trainingFileName = "";
	std::string outFileName = "weights.txt";
	std::string versionNumber = "0.1.0";
	int epochs = 10;
	unsigned int width = 8, height = 8;
	double learningRate = 0.1;
	unsigned int n, d, seed;
	unsigned int rows_count;
	unsigned int* seedArray = new unsigned int[num_procs];
	unsigned int map_seed = 0;
	bool use_mpi = false;
	bool column_label = false;
	int lengthOf = 0;

	if (rank == 0)
	{
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
				<< "\t(string) --mpi   		  Is compiling with MPI" << std::endl
				<< "\t         -l             Is there a label in the last column" <<std::endl;
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
			else if (strcmp(argv[i], "--mpi") == 0)
			{
				use_mpi = true;
			}	
			else if (strcmp(argv[i], "-l") == 0)
			{
				column_label = true;
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
						lengthOf = strlen(argv[i]);
						trainingFileName = (char *)malloc(sizeof(char) * lengthOf);
						strcpy(trainingFileName, argv[i]);
						//trainingFileName = std::string(argv[i]);
						break;
					default:
						std::cout << "Unrecognized positional argument, '" << argv[i] << "'" << std::endl;
				}
				posArgPos++;
			}
		}
		// Load training data
		if (lengthOf <= 0)
		{
			std::cout << "inside generate path" << std::endl;
			rows_count = n;
		}
		else
		{
			countRowsAndCols((std::string)trainingFileName, n, d, column_label);
		}
		if (n == NULL) {
			return 0;
		}
		
		srand(map_seed);
		for(int i = 0; i < num_procs; i++){
			seedArray[i] = rand() % 7919;
		}
	}
	
	//Broadcast the rows_count, dimensions, and epochs that are all handled from the command line. 
	int fileSize = strlen(trainingFileName);
	MPI_Bcast(&fileSize, 1, MPI::INT, 0, MPI::COMM_WORLD);

	if (rank != 0 && fileSize > 0)
	{
		trainingFileName = (char *)malloc(sizeof(char) * fileSize);
	}

	std::cout << "fileName " << (std::string)trainingFileName << std::endl;
	std::cout << "fileSize " << fileSize << " Rank " << rank << std::endl;
	/*
	char *file_nameString = (char*)malloc(sizeof(char) * fileSize);
	
	if (rank == 0)
	{
		std::cout << std::endl;
		for(int i = 0; i < trainingFileName.size(); i++)
		{
			//std::cout << trainingFileName[i] << std::endl;
			file_nameString[i]=trainingFileName[i];
		}
		std::cout << (std::string)file_nameString << std::endl;
	}

	*/

	MPI_Barrier(MPI::COMM_WORLD);
	MPI_Bcast(&n, 1, MPI::UNSIGNED, 0, MPI::COMM_WORLD);
	MPI_Bcast(&d, 1, MPI::UNSIGNED, 0, MPI::COMM_WORLD);
	MPI_Bcast(&epochs, 1, MPI::UNSIGNED, 0, MPI::COMM_WORLD);
	MPI_Bcast(&width, 1, MPI::UNSIGNED, 0, MPI::COMM_WORLD);
	MPI_Bcast(&height, 1, MPI::UNSIGNED, 0, MPI::COMM_WORLD);
	MPI_Bcast(&map_seed, 1, MPI::UNSIGNED, 0, MPI::COMM_WORLD);
	MPI_Bcast(&column_label, 1, MPI::BOOL, 0, MPI::COMM_WORLD);
	
	MPI_Bcast(trainingFileName, fileSize, MPI::CHAR, 0, MPI::COMM_WORLD);
	MPI_Scatter(seedArray, 1, MPI::UNSIGNED, &seed, 1, MPI::UNSIGNED, 0, MPI::COMM_WORLD);

	std::cout << "fileName " << (std::string)trainingFileName << std::endl;
	//std::cout << "fileSize " << fileSize << " Rank " << rank << std::endl;
	// Create untrained SOM

	
	SOM newSom = SOM(width, height);
	// Train SOM and time training
	auto start = std::chrono::high_resolution_clock::now();
	newSom.train_data((std::string)trainingFileName, fileSize, rank, num_procs, epochs, d, n, seed, map_seed, column_label);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
	std::cout << "Finished training in " << duration.count() << "seconds" << std::endl;

	
	if (rank == 0)
	{
		// Save the SOM's weights
		std::ofstream outFile(outFileName, std::ofstream::out);
		if (outFile.is_open()) {
			newSom.save_weights(outFile);
			outFile.close();
			//std::cout << "SOM saved to " << outFileName << std::endl;
		}
	}
	
	MPI::Finalize();
}