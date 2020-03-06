#include "pch.h"

/* Node
The default constuctor.
*/
Node::Node()
{
}

/* set_weights
Desc: The function to set the weights of the function to arg:
	input_weights
*/
void Node::set_weights(double* input_weights)
{
	this->_node_weights = input_weights;
}

/* calculateDistance
Desc: The Node public member function. Returns the calcualted distance
	  between the values of the node's weight and the vector pass as arg:
	  'train_example_weights'
*/
double Node::calculateDistance(double train_example_weights[])
{
	float total = 0.0;
	int i = 0;
	while (train_example_weights[i])
	{
		total = total + pow(train_example_weights[i] - _node_weights[i], 2);
		i++;
	}
	return sqrt(total);
}

void Node::update_weights(double input_weights[], double learing_rate, double influence, int dimensions)
{
	for (int i = 0; i < dimensions; i++)
	{
		_node_weights[i] = _node_weights[i] + influence * learing_rate * (input_weights[i] - _node_weights[i]);
	}
}
