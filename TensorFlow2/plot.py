import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from time import time

from model import AttentionModel
from data import generate_data, data_from_txt
from baseline import load_model
from config import test_parser

import tensorflow as tf

def get_clean_path(arr):
	"""Returns extra zeros from path.
	   Dynamical model generates duplicated zeros for several graphs when obtaining partial solutions.
	"""
	p1, p2 = 0, 1
	output = []
	while p2 < len(arr):
		if arr[p1] != arr[p2]:
			output.append(arr[p1])
			if p2 == len(arr) - 1:
				output.append(arr[p2])
		p1 += 1
		p2 += 1

	if output[0] != 0:
		output.insert(0, 0)# insert 0 in 0th of the array
	if output[-1] != 0:
		output.append(0)# insert 0 at the end of the array
	return output

def plot_route(data, pi, costs, title, idx_in_batch = 0):
	"""Plots journey of agent
	Args:
		data: dataset of graphs
		pi: (batch, decode_step) # tour
		idx_in_batch: index of graph in data to be plotted
	"""

	cost = costs[idx_in_batch].numpy()
	# Remove extra zeros
	pi_ = get_clean_path(pi[idx_in_batch].numpy())

	depot_xy = data[0][idx_in_batch].numpy()
	customer_xy = data[1][idx_in_batch].numpy()
	demands = data[2][idx_in_batch].numpy()
	demands = np.insert(demands, 0, 0.0, axis=0)
	customer_labels = ['(' + str(i) + ', ' + str(demand) + ')' for i, demand in enumerate(demands.round(2), 1)]
	# customer_labels = ['(' + str(demand) + ')' for demand in demands.round(2)]
	
	xy = np.concatenate([depot_xy.reshape(1, 2), customer_xy], axis = 0)

	# Get list with agent loops in path
	list_of_paths, cur_path = [], []
	for idx, node in enumerate(pi_):

		cur_path.append(node)

		if idx != 0 and node == 0:
			if cur_path[0] != 0:
				cur_path.insert(0, 0)
			list_of_paths.append(cur_path)
			cur_path = []

	# print('path: ', pi_)
	# print("xy:", xy)
	# print("demands:", demands)
	# print("list_of_path:", list_of_paths)
	return xy, demands, list_of_paths


def main():
	args = test_parser()
	t1 = time()
	pretrained = load_model(args.path, embed_dim = 128, n_customer = args.n_customer, n_encode_layers = 3)
	# model = AttentionModel()
	# print(f'model loading time:{time()-t1}s')
	if args.txt is not None:
		dataset = data_from_txt(args.txt)
	else:
		dataset = generate_data(n_samples = 1, n_customer = args.n_customer, seed = args.seed) 	
	# print(f'data generate time:{time()-t1}s')
		
	# dataset = generate_data(n_samples = 128, n_customer = 100, seed = 29) 
	# for i, data in enumerate(dataset.batch(128)):
	for i, data in enumerate(dataset.repeat().batch(args.batch)):
		# print("data[0]:", data[0])
		# print("data[1]:", data[1])
		# print("data[2]:", data[2])
		costs, _, pi = pretrained(data, return_pi = True, decode_type = args.decode_type)
		idx_in_batch = tf.argmin(costs, axis = 0)
		# print('costs:', costs)
		# print(f'decode type:{args.decode_type}\nminimum cost: {costs[idx_in_batch]:.3f} and idx: {idx_in_batch} out of {args.batch} solutions')
		# print(f'{pi[idx_in_batch]}\ninference time: {time()-t1}s')
		locations, demands, list_of_paths = plot_route(data, pi, costs, 'Pretrained', idx_in_batch)
		# print("locations.tolist():", locations.tolist())
		# print("demands.tolist()", demands.tolist())
		return locations.tolist(), demands.tolist(), list_of_paths