import tensorflow as tf
import numpy as np
from numpy.random import seed, random, randint
from collections import namedtuple

class AliasTable():    
	'''
	This class implement the alias table data structure. 
	It is able to draw samples from a fixed distribution at O(1) complexity.
	'''

	def __init__(self, weights, keys):
		'''
		Initiate the alias table according to the targeted probability distribution and labels.

		Input: weights --- the target probability distribution.
		       labels  --- the labels corresponding to the probability distribution.

		Return: None
		'''
		self.keys = keys
		self.keyLen = len(keys)
		weights = weights * self.keyLen / weights.sum()

		inx = -np.ones(self.keyLen, dtype=int)
		shortPool = np.where(weights < 1)[0].tolist()
		longPool = np.where(weights > 1)[0].tolist()
		while shortPool and longPool:
			j = shortPool.pop()
			k = longPool[-1]
			inx[j] = k
			weights[k] -= (1 - weights[j])
			if weights[k] < 1:
				shortPool.append( k )
				longPool.pop()

		self.prob = weights
		self.inx = inx

	def draw(self, count=None):
		'''
		Draw several samples from the target probability distribution at O(1) complexity.

		Input: count --- the number of samples.

		Return: self.keys[k] --- the drawn samples.
		'''
		u = random(count)
		j = randint(self.keyLen, size=count)
		k = np.where(u <= self.prob[j], j, self.inx[j])
		return self.keys[k] 




def variable_summaries(var):
	"""
	Track sevearl statistics of the target Tensor variable (for TensorBoard visualization).

	Input: var --- The target Tensor variable. 

	Return: None
	"""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)



'''
The data structure that specifies the parameter of a GCN layer.

input_dim --- the dimension of input node feature.
source_node_type --- type of nodes that this layer sample from.
dest_node_type --- type of nodes that this layer compute embedding for.
sample_size --- number of neighbours sampled for each node.
output_dim --- dimension of generated node embeddings.
head_num --- number of attention head in the aggregator.
nonlinear --- specify the nonlinear activation function.
layer_name --- specify the name of this layer.
'''
layerInfo = namedtuple("layerInfo",
	['input_dim',
	'source_node_type',
	 'dest_node_type',
	 'sample_size',
	 'output_dim',
	 'head_num',
	 'nonlinear',
	 'layer_name']) 


'''
The data structure that specifies the parameter of a Metapath.
It consists of a list of conseuctive GCN layer.

path_length --- the length of a Metapath.
path_name --- specify the name of the Metapath
layer_info_list --- specify the list of GCN layer it consists of.
'''
pathInfo = namedtuple("pathInfo",
	['path_length',
	 'path_name',
	 'layer_info_list'])
