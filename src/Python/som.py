import argparse
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import functools

l_0 = 0.1

def euc(x):
	s = 0
	for i in x:
		s += i ** 2
	return s

class SOM:
	def __init__(self, width, height, squareneurons=False):
		# 'Generate' node lattice
		self._width = width
		self._height = height
		self._hex = not squareneurons
	
	def train(self, train_data, iterations):
		# Normalize data to be between 0 and 1
		train_data = train_data / train_data.max(axis=0)

		# Randomly initialize weights
		self._weights = np.random.rand(self._width, self._height, train_data.shape[1])

		# Initialize NxNx2 matrix where positions[i,j] = [i,j] for the node distance calculations
		positions = np.array(np.meshgrid(np.arange(self._width),np.arange(self._height))).T.reshape(self._width,self._height,2)

		# Initialize map radius
		map_radius = float(max(self._width,self._height) / 2)

		# Initialize time constant
		tc = float(iterations)/math.log(map_radius)

		# Repeat N times
		for ts in range(iterations):
			# Current learning rate
			l_t = l_0 * math.exp(-float(ts)/tc)

			for example in train_data:
				# Calculate the euclidean distance between the weights and the example
				weight_sq_dists = np.apply_along_axis(euc, 2, np.subtract(self._weights, example))
				# Select the BMU
				bmu_index = np.unravel_index(np.argmin(weight_sq_dists, axis=None), weight_sq_dists.shape)

				# Calculate radius of the neighborhood radius of the BMU (start large, diminish each time step) (any nodes within the raidus are inside the BMU's neighborhood)
				neighborhood_radius = map_radius * math.exp(-(float(ts))/tc)

				# Adjust each node's weights (to make more like the chosen vector) (adjust by factor of closeness to BMU)
				# Vectorized function to calculate the theta that partially determines the amount each weight is modified to match the BMU (based on the squared distance to bmu)
				theta_lambda_v = np.vectorize(lambda x: math.exp(-(x)/(2*(neighborhood_radius ** 2))))
				
				# Calculate the distance of each position to the BMU
				dists = np.apply_along_axis((lambda x: x[0] ** 2 + x[1] ** 2), 2, np.subtract(positions, np.array(bmu_index)))
				# Perform theta calculation for each weight
				thetas = theta_lambda_v(dists)

				# Update weights
				self._weights = self._weights + l_t * thetas[:,:,np.newaxis] * (example - self._weights)

	def saveWeights(self, filename):
		np.savetxt(filename, self._weights.reshape(-1,self._weights.shape[2]), header=str(self._width) + ' ' + str(self._height), comments='')

	def loadWeights(self, filename):
		dim = np.loadtxt(filename, max_rows=1)
		self._width = int(dim[0])
		self._height = int(dim[1])

		temp = np.loadtxt(filename, skiprows=1)
		self._weights = np.reshape(temp, (self._width,self._height,temp.shape[1]))

	def displayTopology(self):
		plt.figure()
		plt.imshow(np.average(self._weights, axis=2), cmap=cm.get_cmap(name="RdBu"), interpolation='bicubic')
		plt.title("Topology")

	def displayInputPlanes(self):
		for k in range(self._weights.shape[2]):
			plt.figure()
			plt.imshow(self._weights[:,:,k], cmap=cm.get_cmap(name="RdBu"), interpolation='bicubic')
			plt.title("Input plane " + str(k+1))
	
	def displayColor(self):
		plt.figure()
		plt.imshow(self._weights, interpolation='bicubic')
	
	def displayUMatrix(self):
		# Step 1: Calculate the distance between each node and its neighbors (for square node maps)
		umatrix = np.empty(self._weights.shape[:-1])
		for row in range(self._weights.shape[0]):
			for col in range(self._weights.shape[1]):
				dist = 0
				n_neighbors = 0
				# Up
				if row != self._weights.shape[0] - 1:
					dist += math.sqrt(euc(np.subtract(self._weights[row][col], self._weights[row+1][col])))
					n_neighbors += 1
				# Right
				if col != self._weights.shape[1] - 1:
					dist += math.sqrt(euc(np.subtract(self._weights[row][col], self._weights[row][col+1])))
					n_neighbors += 1
				# Down
				if row != 0:
					dist += math.sqrt(euc(np.subtract(self._weights[row][col], self._weights[row-1][col])))
					n_neighbors += 1
				# Left
				if col != 0:
					dist += math.sqrt(euc(np.subtract(self._weights[row][col], self._weights[row][col-1])))
					n_neighbors += 1
				dist /= n_neighbors
				umatrix[row][col] = dist
		plt.figure()
		plt.imshow(umatrix, cmap=cm.get_cmap(name="RdBu"), interpolation='bicubic')

def getArguments():
	parser = argparse.ArgumentParser(description="Generates a SOM from a given file source or loads a pretrained SOM from a file. Can also display SOMs using a variety of display methods.")

	# SOM dimensions
	parser.add_argument('width', type=int, nargs='?', default=0, help='The width of the SOM.')
	parser.add_argument('height', type=int, nargs='?', default=0, help='The height of the SOM.')
	# Training data (determines SOM dimensionality)
	parser.add_argument('source', nargs='?',type=argparse.FileType('r'), help='The source data for the SOM to be trained on.')
	
	# Number of training iterations
	parser.add_argument('-N', '--trainingiterations', type=int, default=100, help='The number of iterations to do in training.')
	# Output file for weights
	parser.add_argument('-o', '--out', type=argparse.FileType('w'), help='The output file for the SOM\'s node weights.')
	# Input file for weights
	parser.add_argument('-i', '--input', type=argparse.FileType('r'), help='The input file for the SOM\'s node weights (does not retrain loaded map).')

	# Square/Hex neurons
	parser.add_argument('-s', '--squareneurons', type=bool, nargs='?', const=True, default=False, help='Whether the neurons in the map should be square (otherwise hexagonal).')
	
	# Display methods
	parser.add_argument('-d', '--display', action='append', nargs='?', type=str, const='topology', default=[], help='The display method to be used on the SOM. Current usable display methods: topology, input-planes, color, u-matrix. If no argument after flag then defaults to topology') 

	args = parser.parse_args()
	for dm in args.display:
		if dm not in ['topology','input-planes','color','u-matrix']:
			print('Invalid display method \'' + dm + '\'')
			return None

	return args

if __name__ == '__main__':
	args = getArguments()

	if args != None:
		if args.input != None:
			# Load trained SOM
			s = SOM(0,0)
			s.loadWeights(args.input.name)
		else:
			# Pull data from file
			data = np.genfromtxt(args.source.name)
			s = SOM(args.width, args.height, squareneurons=args.squareneurons)

			t0 = time.time()
			# Train SOM
			s.train(data, iterations=args.trainingiterations)
			t1 = time.time()

			print('Total training time:', t1-t0, flush=True)

		if args.out != None:
			s.saveWeights(args.out.name)

		if len(args.display) > 0:
			for dm in args.display:
				if dm == 'topology':
					s.displayTopology()
				elif dm == 'input-planes':
					s.displayInputPlanes()
				elif dm == 'color':
					if s._weights.shape[2] != 3:
						print('Display method color invalid for dimensions not equal to 3')
					else:
						s.displayColor()
				elif dm == 'u-matrix':
					s.displayUMatrix()
			plt.show()