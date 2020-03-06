#pragma once
#include <vector>
#include <math.h>
class Node
{
public:
	int _x_coord;
	int _y_coord;

	Node();

	void set_weights(double *input_weights);
	double calculateDistance(double train_example_weights[]);
	void update_weights(double input_weights[], double learing_rate, double influence, int dimensions);
	double* _node_weights;
private:
	
	
};