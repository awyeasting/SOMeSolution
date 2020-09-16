#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include "SOM.h"

/*
	Generates a random set of training data if there is no input file given
*/

double *generateRandomTrainingInputs(unsigned int &examples, unsigned int &dimensions, int seedValue)
{
	srand(seedValue);
	double *returnData = new double [examples * dimensions];
	for (int i = 0; i < examples; i++)
	{
		int rowMod = (examples - i - 1)*dimensions;
		for (int j = 0; j < dimensions; j++)
		{
			double weight = SOM::randWeight();
			returnData[rowMod+j] = weight;
		}
	}
	return returnData;
}

/*
	Load a set of training data from a given filename
*/
double* loadTrainingData(std::string trainDataFileName, unsigned int& rows, unsigned int& cols) {
	// Open file
	std::ifstream in(trainDataFileName, std::ifstream::in);
	if (!in.is_open()) {
		std::cout << "Invalid training data file '" << trainDataFileName << "'" << std::endl;
		return NULL;
	}

	// Read the first line to obtain the number of columns (dimensions) in the training data
	std::string line;
	std::getline(in, line);
	std::stringstream ss(line);
	std::vector<double> line1;
	double temp;
	cols = 0;
	while (ss >> temp) {
		cols++;
		line1.push_back(temp);
	}
	std::vector<double*> lines;

	// Store first line in dynamic array and put into the vector of rows
	double* tempLine1 = new double[cols];
	for (int j = 0; j < cols; j++) {
		tempLine1[cols - j - 1] = line1.back();
		line1.pop_back();
	}
	lines.push_back(tempLine1);

	// Read all numbers into cols dimensional arrays added to the rows list
	int i = 0;
	double* unpackedLine = NULL;
	while (in >> temp) {
		if (!unpackedLine) {
			unpackedLine = new double[cols];
		}
		unpackedLine[i] = temp;
		i++;
		if (i == cols) {
			lines.push_back(unpackedLine);
			i = 0;
			unpackedLine = NULL;
		}
	}

	// Convert vector of arrays into 1d array of examples
	rows = lines.size();
	double* res = new double[rows * cols];
	for (i = 0; i < rows; i++) {
		double* temp = lines.back();
		int rowMod = (rows-i-1)*cols;
		for (int j = 0; j < cols; j++) {
			res[rowMod + j] = temp[j];
		}
		lines.pop_back();
		free(temp);
	}
	return res;
}

int main(int argc, char *argv[])
{
	std::string trainingFileName = "";
	std::string outFileName = "weights.txt";
	std::string versionNumber = "0.1.0";
	int epochs = 10;
	unsigned int width = 8, height = 8;
	double learningRate = 0.1;
	unsigned int n, d;
	int posArgPos = 0;
	srand(time(NULL));
	int seedValue = rand();
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
			<< "\t(int)    -s --seed      Integer value to intialize seed for generating" << std::endl;
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
				seedValue = std::stoi(argv[i+1]);
				i++;
			}
			else {
				std::cout << "If the --seed option is used, the following argument should be an integer argument" << std::endl;
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
					trainingFileName = std::string(argv[i]);
					break;
				default:
					std::cout << "Unrecognized positional argument, '" << argv[i] << "'" << std::endl;
			}
			posArgPos++;
		}
	}
	// Load training data
	double *trainData;
	if (trainingFileName == "")
	{
		trainData = generateRandomTrainingInputs(n,d, seedValue);
	}
	else
	{
		trainData = loadTrainingData(trainingFileName, n, d);
	}
	if (trainData == NULL) {
		return 0;
	}
	// Create untrained SOM
	SOM newSom = SOM(width, height);
	// Train SOM and time training
	auto start = std::chrono::high_resolution_clock::now();
	newSom.train_data(trainData, n, d, epochs, learningRate);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
	std::cout << "Finished training in " << duration.count() << "seconds" << std::endl;

	// Save the SOM's weights
	std::ofstream outFile(outFileName, std::ofstream::out);
	if (outFile.is_open()) {
		newSom.save_weights(outFile);
		outFile.close();
		std::cout << "SOM saved to " << outFileName << std::endl;
	}
}