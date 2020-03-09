#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "commandLine.h"
#include "SOM.h"

/*
	Load a set of training data from a given filename
*/
double** loadTrainingData(std::string trainDataFileName, unsigned int& rows, unsigned int& cols) {
	// Open file
	std::ifstream in(trainDataFileName, std::ifstream::in);

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

	// Convert vector of arrays into 2d array
	rows = lines.size();
	double** res = new double*[rows];
	for (i = 0; i < rows; i++) {
		res[rows - i - 1] = lines.back();
		lines.pop_back();
	}
	return res;
}



int main(int argc, char *argv[])
{
	std::unordered_map<std::string, std::string> initiators;

	std::string trainingIterations;
	std::string inputFile;
	std::string source = "train.txt";
	std::string outputFile;
	std::string squareNeurons;
	std::string display;
	int epochs = 1000;
	unsigned int width = 15, height = 15;

	if(argc > 4)
	{
		setHashMap(initiators, argv, argc);
		setInitialValues(initiators, &trainingIterations, &outputFile, &inputFile, &squareNeurons, &display);
		if(trainingIterations != "")
			epochs = std::stoi(trainingIterations);
	}
	else if(argc < 4)
	{
		if(strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
		{
			help();

			return 0;
		}
		else if(strcmp(argv[1], "-v") == 0 || strcmp(argv[1], "--version") == 0)
		{
			showVersion();

			return 0;
		}
	}

	source = argv[argc - 1];
	height = atoi(argv[argc - 2]);
	width = atoi(argv[argc - 3]);

	double learningRate = 0.1;
	std::string trainingInFileName(source);
	std::string weightsOutFileName(outputFile);

	// Load training data
	unsigned int n, d;
	double **trainData = loadTrainingData(trainingInFileName, n, d);

	// Create untrained SOM
	SOM newSom = SOM(width, height);

	// Train SOM and time training
	auto start = std::chrono::high_resolution_clock::now();
	newSom.train_data(trainData, n, d, epochs, learningRate);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
	std::cout << "Finished training in " << duration.count() << "seconds" << std::endl;

	// Save the SOM's weights
	std::ofstream outFile(weightsOutFileName, std::ofstream::out);
	newSom.save_weights(outFile);
	outFile.close();

	std::cout << "SOM saved to " << weightsOutFileName << std::endl;
}