import numpy as np
from numpy.random import seed, random, randint
import tensorflow as tf
import time
from utils import *


class RecoGCN():
	'''
	The implementation of RecoGCN model.
	'''
	def __init__(self, userCnt, agentCnt, itemCnt, pathInfos, adj, numNegative, featEmbedDim, idenEmbedDim, userFeatureMap, agentFeatureMap, itemFeatureMap, dim, batchUser, batchAgent, batchItem, batchNeg, inDrop=0.0, lr = 0.005, l2_coef = 0.0005):
		'''
		Initiate the parameter of the RecoGCN model.

		Input: userCnt, agentCnt, itemCnt --- the number of nodes that corresponds to user, agent and item. 
		       pathInfos --- a dictionary of pathInfo data structure that specify the designed Metapath for user, agent and item.
		       adj --- adjcency matrices of the hetergenous network. 0 -- user, 1 -- agent, 2 -- item in social setting, 3 -- item in app. 
		       adj[0][1] denotes the adjcency matrix from user to agent.
		       numNegative --- number of negative items that are drawn to train the model.
		       featEmbedDim, idenEmbedDim --- embedding dimension for node feature and node identity.
		       userFeatureMap, agentFeatureMap, itemFeatureMap --- the feature map of user, agent and item.
		       dim --- the dimension of the final embedding of user, agent and item.
		       batchUser, batchAgent, batchItem, batchNeg --- batch samples of user, agent, item and negative items.
		       inDrop --- the drop out probability.
		       lr --- the learning rate.
		       l2_coef --- the weight of l2-norm regularization term. 
		
		Return: None
		'''
		self.userPath = pathInfos['user_path']
		self.itemPath = pathInfos['item_path']
		self.agentPath = pathInfos['agent_path']
		self.featEmbedDim = featEmbedDim
		self.idenEmbedDim = idenEmbedDim

		self.adj = {0: {1: tf.constant(adj[0][1], dtype=tf.int32), 2: tf.constant(adj[0][2], dtype=tf.int32), 3: tf.constant(adj[0][3], dtype=tf.int32)}, 
					1: {0: tf.constant(adj[1][0], dtype=tf.int32)}, 2: {0: tf.constant(adj[2][0], dtype=tf.int32)}, 3: {0: tf.constant(adj[3][0], dtype=tf.int32)}}
		self.numNegative = numNegative
		self.margin = 0.1

		userFeatureMap = tf.Variable(tf.constant(userFeatureMap, dtype=tf.int8), trainable=False)
		agentFeatureMap = tf.Variable(tf.constant(agentFeatureMap, dtype=tf.int8), trainable=False)
		itemFeatureMap = tf.Variable(tf.constant(itemFeatureMap, dtype=tf.int8), trainable=False)
		featureMap = [userFeatureMap, agentFeatureMap, itemFeatureMap]

		user_feat_embed = tf.get_variable('user_feat_embed', shape=(userFeatureMap.get_shape()[1], featEmbedDim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		agent_feat_embed = tf.get_variable('agent_feat_embed', shape=(agentFeatureMap.get_shape()[1], featEmbedDim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		item_feat_embed = tf.get_variable('item_feat_embed', shape=(itemFeatureMap.get_shape()[1], featEmbedDim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		self.feat_embed = [user_feat_embed, agent_feat_embed, item_feat_embed]

		user_iden_embed = tf.get_variable('user_iden_embed', shape=(userCnt, idenEmbedDim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		agent_iden_embed = tf.get_variable('agent_iden_embed', shape=(agentCnt, idenEmbedDim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		item_iden_embed = tf.get_variable('item_iden_embed', shape=(itemCnt, idenEmbedDim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
		self.iden_embed = [user_iden_embed, agent_iden_embed, item_iden_embed]	

		self.build(featureMap, batchUser, batchAgent, batchItem, batchNeg, inDrop, lr, l2_coef, dim)

	def build(self, featureMap, batchUser, batchAgent, batchItem, batchNeg, inDrop, lr, l2_coef, dim):
		'''
		Build the embedding inferring modules of RecoGCN model. Specify the loss computing, accuracy evaluating and training module.

		Input: featureMap --- the list of feature map for user, agent and item.
		       batchUser, batchAgent, batchItem, batchNeg --- batch samples of user, agent, item and negative items.
		       inDrop --- the drop out probability.
		       lr --- the learning rate.
		       l2_coef --- the weight of l2-norm regularization term. 
		       dim --- the dimension of the final embedding of user, agent and item.
		
		Return: None
		'''
		with tf.name_scope('User_ReceptiveField_Sampling'):
			receptiveFieldUser, userNodeTypeList = self.sample(batchUser, self.userPath)
		with tf.name_scope('User_Embedding'):
			outputUser = self.forward(receptiveFieldUser, userNodeTypeList, featureMap, self.userPath, inDrop)

		with tf.name_scope('Agent_ReceptiveField_Sampling'):
			receptiveFieldAgent, agentNodeTypeList = self.sample(batchAgent, self.agentPath)
		with tf.name_scope('Agent_Embedding'):
			outputAgent = self.forward(receptiveFieldAgent, agentNodeTypeList, featureMap, self.agentPath, inDrop)

		with tf.name_scope('Item_ReceptiveField_Sampling'):			
			receptiveFieldItem, itemNodeTypeList = self.sample(batchItem, self.itemPath)
			receptiveFieldNeg, negNodeTypeList = self.sample(batchNeg, self.itemPath)
		with tf.name_scope('Item_Embedding'):
			outputItem = self.forward(receptiveFieldItem, itemNodeTypeList, featureMap, self.itemPath, inDrop)		
		with tf.name_scope('Negative_Item_Embedding'):
			outputNeg = self.forward(receptiveFieldNeg, negNodeTypeList, featureMap, self.itemPath, inDrop)
		
		with tf.name_scope('Coattention_Embedding'):
			user_vec_pos, agent_vec_pos, self.item_vec_pos = self.coAttendEmbedding(outputUser, len(self.userPath), outputAgent, len(self.agentPath), outputItem, len(self.itemPath), dim, negative_flag=False)
			user_vec_neg, agent_vec_neg, self.item_vec_neg = self.coAttendEmbedding(outputUser, len(self.userPath), outputAgent, len(self.agentPath), outputNeg, len(self.itemPath), dim, negative_flag=True)
		
		with tf.name_scope('User_Agent_Embedding'):
			self.user_agent_pos = tf.concat(axis=1, values=[user_vec_pos, agent_vec_pos])
			self.user_agent_neg = tf.concat(axis=1, values=[user_vec_neg, agent_vec_neg])
			self.bilinear_weights = tf.get_variable('bilinear_weights', shape=(self.user_agent_pos.get_shape()[1], self.item_vec_pos.get_shape()[1]),
								   dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

		self.loss = self.loss()
		self.mrr, self.ndcg, self.hr1, self.hr3, self.hr5, self.hr10 = self.accuracy()
		self.train_op = self.training(self.loss, lr, l2_coef)

	def attAggregator(self, layerInfo, inputFeature, contextFeature, inDrop=0.0, layerName='attentionAgg'):
		'''
		The implementation of a graph attention layer.

		Input: layerInfo --- the layerInfo data structure that specifies the parameter of this layer.
		       inputFeature --- the embedding of target nodes.
		       contextFeature --- the embedding of the sampled neighbours of the target nodes.
		       inDrop --- the drop out probability.
		       layerName --- specify the variable_scope of this layer with a layer name(important for sharing layer relation-wise). 
		
		Return: output --- updated embeddings of the target nodes.
		'''
		with tf.variable_scope(layerName, reuse=tf.AUTO_REUSE):
			atten_dim = 32

			self_embed = [tf.get_variable('self_embed_'+str(headInd), shape=(self.featEmbedDim+self.idenEmbedDim, layerInfo.output_dim/2),
						 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) for headInd in range(layerInfo.head_num)]
			context_embed = [tf.get_variable('context_embed_'+str(headInd), shape=(layerInfo.input_dim, layerInfo.output_dim/2),
						 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) for headInd in range(layerInfo.head_num)]
			
			self_attend = [tf.get_variable('self_attend_'+str(headInd), shape=(self.featEmbedDim+self.idenEmbedDim, atten_dim),
						 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) for headInd in range(layerInfo.head_num)]
			context_attend = [tf.get_variable('context_attend_'+str(headInd), shape=(layerInfo.input_dim, atten_dim),
						 dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) for headInd in range(layerInfo.head_num)]
			
			inputSize = tf.cast(inputFeature.get_shape().as_list()[0], tf.int32)
			sampleSize = tf.cast(contextFeature.get_shape().as_list()[0]/inputFeature.get_shape().as_list()[0], tf.int32)

			if inDrop != 0.0:
				inputFeature = tf.nn.dropout(inputFeature, 1.0 - inDrop)
				contextFeature = tf.nn.dropout(contextFeature, 1.0 - inDrop)

			input_emb = [tf.matmul(inputFeature, embedTemp) for embedTemp in self_embed]
			context_emb = [tf.matmul(contextFeature, embedTemp) for embedTemp in context_embed] 

			input_att = [tf.matmul(inputFeature, attTemp) for attTemp in self_attend]
			context_att = [tf.reshape(tf.matmul(contextFeature, attTemp),shape=[inputSize,sampleSize,-1]) for attTemp in context_attend] 

			# map node embbedings to two attention values(self & context) through two parralel linear transforms
			attCoef = [tf.nn.softmax(tf.reduce_sum(tf.expand_dims(input_att[headInd], 1)* context_att[headInd], 2)) for headInd in range(layerInfo.head_num)]

			# aggregate embbeding of neighbours weighted by attentin coef
			context_emb = [tf.reshape(item, shape=(inputSize, sampleSize, -1)) for item in context_emb]
			attCoef = [tf.reshape(item, shape=(inputSize, sampleSize, 1)) for item in attCoef]
			vals = [tf.concat([input_emb[headInd], tf.reduce_sum(attCoef[headInd] * context_emb[headInd], 1)], axis=1) for headInd in range(layerInfo.head_num)]

			ret = [tf.contrib.layers.bias_add(vals[headInd]) for headInd in range(layerInfo.head_num)]

			output = tf.add_n(ret)/layerInfo.head_num

			output = layerInfo.nonlinear(tf.concat(output, axis=-1))			
			return output

	def sample(self, seedNodes, pathInfos):
		'''
		Sample the receptive field of the target nodes hop by hop iteratively according to the parameter specified in pathInfos.

		Input: seedNodes --- the target nodes we want to infer embeddings for.
		       pathInfos --- the data structure specifies a Metapath and how to sample receptive field according to it.
		
		Return: output_samples --- the list of sampled recpetive field of target nodes.
		        node_type_lists --- the list of node type in each hop.
		'''
		output_samples = []
		node_type_lists = []
		for pathInfo in pathInfos:
			samples = [tf.reshape(seedNodes,[-1])]
			node_type_list = [min(pathInfo.layer_info_list[-1].source_node_type, 2)]
			for k in range(pathInfo.path_length):
				t = pathInfo.path_length - k - 1
				source_type = pathInfo.layer_info_list[t].source_node_type
				dest_type = pathInfo.layer_info_list[t].dest_node_type
				sample_size = pathInfo.layer_info_list[t].sample_size
				neighbor_list = tf.nn.embedding_lookup(self.adj[source_type][dest_type], samples[k]) 
				neighbor_list = tf.transpose(tf.random_shuffle(tf.transpose(neighbor_list)))
				neighbor_list = tf.slice(neighbor_list, [0,0], [-1, sample_size])

				samples.append(tf.reshape(neighbor_list, [-1]))
				node_type_list.append(min(pathInfo.layer_info_list[t].dest_node_type,2))
			output_samples.append(samples)
			node_type_lists.append(node_type_list)
		return output_samples, node_type_lists

	def forward(self, receptiveField, nodeTypelist, featureMap, pathInfos, inDrop):    
		'''
		The complete embedding inferring process.

		Input: receptiveField --- the sampled receptive field of target nodes, each element is the neighbours for a certain hop.
		       nodeTypelist --- the list of node type in each hop.
		       featureMap --- the list of feature map for user, agent and item.
		       pathInfos --- a dictionary of pathInfo data structure that specify the designed Metapath for user, agent and item.
		       inDrop --- the drop out probability.
		
		Return: output --- lists of embeddings for the target nodes, each element is the embedding for one Metapath.
		'''
		output_embeds = []
		for pathInd in range(len(pathInfos)):
			with tf.name_scope(pathInfos[pathInd].path_name):
				pathLen = pathInfos[pathInd].path_length
				featureEmbedList = [tf.matmul(tf.cast(tf.nn.embedding_lookup(featureMap[nodeTypelist[pathInd][hopid]], receptiveField[pathInd][hopid]), tf.float32), 
					                   self.feat_embed[nodeTypelist[pathInd][hopid]]) for hopid in range(pathLen+1)]
				idenEmbedList = [tf.nn.embedding_lookup(self.iden_embed[nodeTypelist[pathInd][hopid]], receptiveField[pathInd][hopid]) for hopid in range(pathLen+1)]				
				featureMapList = [tf.concat([featureEmbedList[hopid], idenEmbedList[hopid]], axis=1) for hopid in range(pathLen+1)]

				layerInd = 0
				hiddenEmb = featureMapList[-1]
				while(layerInd<pathLen):	
					hiddenEmb = self.attAggregator(pathInfos[pathInd].layer_info_list[layerInd], featureMapList[pathLen-layerInd-1], hiddenEmb, inDrop=inDrop, layerName=pathInfos[pathInd].layer_info_list[layerInd].layer_name) 
					layerInd += 1
				output_embeds.append(tf.expand_dims(hiddenEmb, axis=1))

		output_embeds = tf.concat(output_embeds, axis=1)

		return output_embeds

	def coAttendEmbedding(self, user_vec, user_path_num, agent_vec, agent_path_num, item_vec, item_path_num, dim, negative_flag):
		'''
		The co-attention module that fuses the embeddings from different Metapath together by attending to other elements in the transaction.

		Input: user_vec, agent_vec, item_vec --- the lists of embeddings for user, agent and item that learned from different Metapaths.
		       user_path_num, agent_path_num, item_path_num --- the number of different Metapath for user, agent and item.
		       dim --- the dimension of output embeddings.
		       negative_flag --- wheter the item embeddings corresponding to negtaive items. 
		
		Return: user_emb, agent_emb, item_emb --- the fused embeddings for user, agent and item.
		'''
		with tf.variable_scope("coAttention_embedding_layer", reuse=tf.AUTO_REUSE):
			inputDim = user_vec.get_shape().as_list()[2]
			attnDim = 64
			valueDim = dim
			user_value_embed = tf.get_variable('user_value_embed', shape=(inputDim, valueDim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 
			user_query_embed = tf.get_variable('user_query_embed', shape=(inputDim, attnDim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 
			user_query_comb = tf.get_variable('user_query_comb', shape=(1, user_path_num, 1), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 

			agent_value_embed = tf.get_variable('agent_value_embed', shape=(inputDim, valueDim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 
			agent_query_embed = tf.get_variable('agent_query_embed', shape=(inputDim, attnDim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 
			agent_query_comb = tf.get_variable('agent_query_comb', shape=(1, agent_path_num, 1), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 

			item_value_embed = tf.get_variable('item_value_embed', shape=(inputDim, valueDim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 
			item_query_embed = tf.get_variable('item_query_embed', shape=(inputDim, attnDim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 
			item_query_comb = tf.get_variable('item_query_comb', shape=(1, item_path_num, 1), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 

			ua_query_embed = tf.get_variable('ua_query_embed', shape=(attnDim*2, valueDim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 
			ui_query_embed = tf.get_variable('ui_query_embed', shape=(attnDim*2, valueDim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 
			ai_query_embed = tf.get_variable('ai_query_embed', shape=(attnDim*2, valueDim), dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 
			
			user_vec_flat = tf.reshape(user_vec, shape=[-1, inputDim])
			agent_vec_flat = tf.reshape(agent_vec, shape=[-1, inputDim])
			item_vec_flat = tf.reshape(item_vec, shape=[-1, inputDim])
			
			user_value = tf.nn.relu(tf.matmul(user_vec_flat, user_value_embed))
			user_query = tf.nn.relu(tf.matmul(user_vec_flat, user_query_embed))
			user_query = tf.nn.relu(tf.reduce_sum(tf.reshape(user_query, shape=[-1, user_path_num, attnDim]) * user_query_comb, axis=1))

			agent_value = tf.nn.relu(tf.matmul(agent_vec_flat, agent_value_embed))
			agent_query = tf.nn.relu(tf.matmul(agent_vec_flat, agent_query_embed))
			agent_query = tf.nn.relu(tf.reduce_sum(tf.reshape(agent_query, shape=[-1, agent_path_num, attnDim]) * agent_query_comb, axis=1))

			item_value = tf.nn.relu(tf.matmul(item_vec_flat, item_value_embed))
			item_query = tf.nn.relu(tf.matmul(item_vec_flat, item_query_embed))
			item_query = tf.nn.relu(tf.reduce_sum(tf.reshape(item_query, shape=[-1, item_path_num, attnDim]) * item_query_comb, axis=1))

			if negative_flag:
				ua_query = tf.expand_dims(tf.expand_dims(tf.matmul(tf.concat([user_query, agent_query], axis=1), ua_query_embed), axis=1), axis=1)

				ui_user_query = tf.expand_dims(tf.matmul(user_query, ui_query_embed[:attnDim,:]), axis=1)
				ui_item_query = tf.expand_dims(tf.matmul(item_query, ui_query_embed[attnDim:,:]), axis=0)

				ui_query = ui_user_query + ui_item_query

				ai_agent_query = tf.expand_dims(tf.matmul(agent_query, ai_query_embed[:attnDim,:]), axis=1)
				ai_item_query = tf.expand_dims(tf.matmul(item_query, ai_query_embed[attnDim:,:]), axis=0)
				ai_query = ai_agent_query + ai_item_query

				user_value = tf.reshape(user_value, shape=[-1, 1, user_path_num, valueDim])
				user_coef = tf.nn.softmax(tf.reduce_sum(tf.expand_dims(ai_query, axis=2) * user_value, axis=3))
				user_emb = tf.reduce_sum(user_value * tf.expand_dims(user_coef, axis=3), axis=2)

				agent_value = tf.reshape(agent_value, shape=[-1, 1, agent_path_num, valueDim])
				agent_coef = tf.nn.softmax(tf.reduce_sum(tf.expand_dims(ui_query, axis=2) * agent_value, axis=3))
				agent_emb = tf.reduce_sum(agent_value * tf.expand_dims(agent_coef, axis=3), axis=2)

				item_value = tf.reshape(item_value, shape=[-1, item_path_num, valueDim])
				item_coef = tf.nn.softmax(tf.reduce_sum(ua_query * tf.expand_dims(item_value, axis=0), axis=3))
				item_emb = tf.reduce_sum(item_value * tf.expand_dims(item_coef, axis=3), axis=2)

				user_emb = tf.reshape(user_emb, shape=[-1, valueDim])
				item_emb = tf.reshape(item_emb, shape=[-1, valueDim])
				agent_emb = tf.reshape(agent_emb, shape=[-1, valueDim])

			else:
				ua_query = tf.expand_dims(tf.matmul(tf.concat([user_query, agent_query], axis=1), ua_query_embed), axis=1)
				ui_query = tf.expand_dims(tf.matmul(tf.concat([user_query, item_query], axis=1), ui_query_embed), axis=1)
				ai_query = tf.expand_dims(tf.matmul(tf.concat([agent_query, item_query], axis=1), ai_query_embed), axis=1)

				item_value = tf.reshape(item_value, shape=[-1, item_path_num, valueDim])
				user_value = tf.reshape(user_value, shape=[-1, user_path_num, valueDim])
				agent_value = tf.reshape(agent_value, shape=[-1, agent_path_num, valueDim])
				item_coef = tf.nn.softmax(tf.reduce_sum(ua_query * item_value, axis=2))
				user_coef = tf.nn.softmax(tf.reduce_sum(ai_query * user_value, axis=2))
				agent_coef = tf.nn.softmax(tf.reduce_sum(ui_query * agent_value, axis=2))
				
				self.agent_coef = agent_coef
				self.user_coef = user_coef
				self.item_coef = item_coef

				user_emb = tf.reduce_sum(tf.expand_dims(user_coef, axis=2) * user_value, axis=1)
				item_emb = tf.reduce_sum(tf.expand_dims(item_coef, axis=2) * item_value, axis=1)
				agent_emb = tf.reduce_sum(tf.expand_dims(agent_coef, axis=2) * agent_value, axis=1)

			return user_emb, agent_emb, item_emb

	def loss(self):
		'''
		Compute the hinge loss of model prediction.

		Input: None.
		
		Return: loss --- the loss of model prediction.
		'''
		user_agent_emb_pos = tf.matmul(self.user_agent_pos, self.bilinear_weights)
		user_agent_emb_neg = tf.matmul(self.user_agent_neg, self.bilinear_weights)
		with tf.name_scope('item_logits'):
			self.item_logits = tf.sigmoid(tf.reduce_sum(user_agent_emb_pos * self.item_vec_pos, axis=1))
			variable_summaries(self.item_logits)
		with tf.name_scope('neg_logits'):
			self.neg_logits = tf.reshape(tf.sigmoid(tf.reduce_sum(user_agent_emb_neg * self.item_vec_neg, axis=1)),shape=[-1, self.numNegative])
			variable_summaries(self.neg_logits)		
		with tf.name_scope('loss'):	
			diff = tf.nn.relu(tf.subtract(self.neg_logits, tf.expand_dims(self.item_logits, axis=1) - self.margin), name='diff')
			variable_summaries(diff)
			loss = tf.reduce_sum(diff)
		return loss

	def accuracy(self):
		'''
		Evaluate and return the accuracy of model prediction.

		Input: None 
		
		Return: mrr_avg --- the average mrr score.
		        ndcg_avg --- the averge ndcg score.
		        hr1, hr3, hr5, hr10 --- hit ratio at the top 1,3,5,10 rankings.
		'''
		with tf.name_scope('accuracy'):
			all_logits = tf.concat(axis=1, values=[self.neg_logits, tf.expand_dims(self.item_logits, axis=1)])
			candidate_len = all_logits.get_shape()[1]
			_, indices_of_ranks = tf.nn.top_k(all_logits, k=candidate_len)
			_, ranks = tf.nn.top_k(-indices_of_ranks, k=candidate_len)
			rank = tf.cast(ranks[:, -1], tf.float32)

			mrr = tf.div(1.0, rank + 1)
			ndcg = tf.div(tf.log(2.0), tf.log(rank+2))

			variable_summaries(mrr)
			variable_summaries(ndcg)
			mrr_avg = tf.reduce_mean(mrr)
			ndcg_avg = tf.reduce_mean(ndcg)

			hr1 = tf.reduce_mean(tf.cast(tf.greater(1.0, rank), tf.float32))
			hr3 = tf.reduce_mean(tf.cast(tf.greater(3.0, rank), tf.float32))
			hr5 = tf.reduce_mean(tf.cast(tf.greater(5.0, rank), tf.float32))
			hr10 = tf.reduce_mean(tf.cast(tf.greater(10.0, rank), tf.float32))
		return mrr_avg, ndcg_avg, hr1, hr3, hr5, hr10

	def training(self, loss, lr, l2_coef):
		'''
		Derive and return the training operation.

		Input: loss --- the loss of model prediction.
		       lr --- learning rate.
		       l2_coef --- the weight coefficient of l2-norm regularization.
		
		Return: train_op --- training operation.
		'''
		# weight decay
		vars = tf.trainable_variables()
		lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * l2_coef
		# optimizer
		opt = tf.train.AdamOptimizer(learning_rate=lr)
		# training op
		train_op = opt.minimize(loss+lossL2)		
		return train_op


