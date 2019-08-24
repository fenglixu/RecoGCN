import sys
import os
import networkx
import numpy as np
from numpy.random import seed, random, randint
import tensorflow as tf
import time
from collections import namedtuple
import h5py
from utils import *
from models import *

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

def genPathInfo(nonlinearity, featEmbedDim, idenEmbedDim, pathNum):
	'''
	The function that specify the designed Meta-path. Feel free to try other Meta-paths.

	Input: nonlinearity --- the nonlinear activation function.
	       featMebedDim, idenEmbedDim --- the embedding dimension node feature and identity.
	       pathNum --- the number returned Metapaths. To evaluate the effectiveness of adding new paths.
	
	Return: pathInfos --- a dictionary of pathInfo data structure that specify the designed Metapath for user, agent and item.
	'''
	hiddenDim = featEmbedDim + idenEmbedDim
	outputDim = featEmbedDim + idenEmbedDim
	user_path = [	pathInfo(3, 'U_I_U_I', [layerInfo(featEmbedDim+idenEmbedDim, 0, 2, 5, hiddenDim, 1, nonlinearity, 'U_I'), layerInfo(hiddenDim, 2, 0, 5, hiddenDim, 1, nonlinearity, 'I_U'), layerInfo(hiddenDim, 0, 2, 5, outputDim, 1, nonlinearity, 'U_I')]),
					pathInfo(1, 'U_I', [layerInfo(featEmbedDim+idenEmbedDim, 0, 2, 20, outputDim, 1, nonlinearity, 'U_I')]),
					pathInfo(1, 'U_AI', [layerInfo(featEmbedDim+idenEmbedDim, 0, 3, 20, outputDim, 1, nonlinearity, 'U_AI')]), 				
					pathInfo(3, 'U_A_U_I', [layerInfo(featEmbedDim+idenEmbedDim, 0, 2, 5, hiddenDim, 1, nonlinearity, 'U_I'), layerInfo(hiddenDim, 1, 0, 5, hiddenDim, 1, nonlinearity, 'A_U'), layerInfo(hiddenDim, 0, 1, 5, outputDim, 1, nonlinearity, 'U_A')]),
					pathInfo(3, 'U_AI_U_I', [layerInfo(featEmbedDim+idenEmbedDim, 0, 2, 5, hiddenDim, 1, nonlinearity, 'U_I'), layerInfo(hiddenDim, 3, 0, 5, hiddenDim, 1, nonlinearity, 'AI_U'), layerInfo(hiddenDim, 0, 3, 5, outputDim, 1, nonlinearity, 'U_AI')]),
					#pathInfo(3, 'U_A_U_AI', [layerInfo(featEmbedDim+idenEmbedDim, 0, 3, 5, hiddenDim, 1, nonlinearity, 'U_AI'), layerInfo(hiddenDim, 1, 0, 5, hiddenDim, 1, nonlinearity, 'A_U'), layerInfo(hiddenDim, 0, 1, 5, outputDim, 1, nonlinearity, 'U_A')]),
					#pathInfo(3, 'U_I_U_AI', [layerInfo(featEmbedDim+idenEmbedDim, 0, 3, 5, hiddenDim, 1, nonlinearity, 'U_AI'), layerInfo(hiddenDim, 2, 0, 5, hiddenDim, 1, nonlinearity, 'I_U'), layerInfo(hiddenDim, 0, 2, 5, outputDim, 1, nonlinearity, 'U_I')])
					]


	# for item: I-U, I-U-I-U, I-U-A-U
	item_path = [	pathInfo(3, 'I_U_I_U', [layerInfo(featEmbedDim+idenEmbedDim, 2, 0, 5, hiddenDim, 1, nonlinearity, 'I_U'), layerInfo(hiddenDim, 0, 2, 5, hiddenDim, 1, nonlinearity, 'U_I'), layerInfo(hiddenDim, 2, 0, 5, outputDim, 1, nonlinearity, 'I_U')]),
					pathInfo(3, 'I_U_A_U', [layerInfo(featEmbedDim+idenEmbedDim, 1, 0, 5, hiddenDim, 1, nonlinearity, 'A_U'), layerInfo(hiddenDim, 0, 1, 5, hiddenDim, 1, nonlinearity, 'U_A'), layerInfo(hiddenDim, 2, 0, 5, outputDim, 1, nonlinearity, 'I_U')]),					
					pathInfo(3, 'I_U_AI_U', [layerInfo(featEmbedDim+idenEmbedDim, 3, 0, 5, hiddenDim, 1, nonlinearity, 'AI_U'), layerInfo(hiddenDim, 0, 3, 5, hiddenDim, 1, nonlinearity, 'U_AI'), layerInfo(hiddenDim, 2, 0, 5, outputDim, 1, nonlinearity, 'I_U')]),
					pathInfo(1, 'I_U', [layerInfo(featEmbedDim+idenEmbedDim, 2, 0, 20, outputDim, 1, nonlinearity, 'I_U')]),
					#pathInfo(2, 'I_U_I', [layerInfo(featEmbedDim+idenEmbedDim, 2, 0, 5, hiddenDim, 1, nonlinearity, 'I_U'), layerInfo(hiddenDim, 0, 2, 5, outputDim, 1, nonlinearity, 'U_I')]),
					#pathInfo(2, 'I_U_AI', [layerInfo(featEmbedDim+idenEmbedDim, 0, 3, 5, hiddenDim, 1, nonlinearity, 'U_AI'), layerInfo(hiddenDim, 2, 0, 5, outputDim, 1, nonlinearity, 'I_U')])
					]

	
	# for agent: A-U, A-U-I
	agent_path = [pathInfo(1, 'A_U', [layerInfo(featEmbedDim+idenEmbedDim, 1, 0, 20, outputDim, 1, nonlinearity, 'A_U')]), 
				pathInfo(2, 'A_U_I', [layerInfo(featEmbedDim+idenEmbedDim, 0, 2, 5, hiddenDim, 1, nonlinearity, 'U_I'), layerInfo(hiddenDim, 1, 0, 5, outputDim, 1, nonlinearity, 'A_U')]),
				pathInfo(2, 'A_U_AI', [layerInfo(featEmbedDim+idenEmbedDim, 0, 3, 5, hiddenDim, 1, nonlinearity, 'U_AI'), layerInfo(hiddenDim, 1, 0, 5, outputDim, 1, nonlinearity, 'A_U')]),
				#pathInfo(2, 'A_U_I_U', [layerInfo(featEmbedDim+idenEmbedDim, 2, 0, 5, hiddenDim, 1, nonlinearity, 'I_U'),layerInfo(hiddenDim, 0, 2, 5, hiddenDim, 1, nonlinearity, 'U_I'), layerInfo(hiddenDim, 1, 0, 5, outputDim, 1, nonlinearity, 'A_U')]),
				#pathInfo(2, 'A_U_AI_U', [layerInfo(featEmbedDim+idenEmbedDim, 3, 0, 5, hiddenDim, 1, nonlinearity, 'AI_U'),layerInfo(hiddenDim, 0, 3, 5, hiddenDim, 1, nonlinearity, 'U_AI'), layerInfo(hiddenDim, 1, 0, 5, outputDim, 1, nonlinearity, 'A_U')]),
				]

	user_path_len = min(pathNum, 5)
	user_path = user_path[:user_path_len]

	item_path_len = min(pathNum, 4)
	item_path = item_path[:item_path_len]

	agent_path_len = min(pathNum, 3)
	agent_path = agent_path[:agent_path_len]
	pathInfos = {'user_path': user_path, 'item_path': item_path, 'agent_path':agent_path}
	return pathInfos


def evaluate_model(lr, lamb, adj, batch_size, userCnt, agentCnt, itemCnt, numNegative, featEmbedDim, idenEmbedDim, userFeature, agentFeature, itemFeature, train_sample, val_sample, test_sample, aliasTable, log_dir, test_num, dim, pathNum):
	'''
	Evaluate the model performance with a specified settings, and log the result in a predefined directory.

	Input: lr --- learning rate.
		   lamb --- the weight of regularization terms.
		   adj --- adjcency matrices of the hetergenous network. 0 -- user, 1 -- agent, 2 -- item in social setting, 3 -- item in app. 
		   adj[0][1] denotes the adjcency matrix from user to agent.
		   batch_size --- the number of samples in a batch.		
	       userCnt, agentCnt, itemCnt --- the number of nodes that corresponds to user, agent and item. 
	       numNegative --- the number of negative samples.
	       featMebedDim, idenEmbedDim --- the embedding dimension node feature and identity.
	       userFeature, agentFeature, itemFeature --- the feature map for user, agent and item.
	       train_sample, val_sample, test_sample --- the feature map for user, agent and item.
	       aliasTable --- the alias table data structure to efficiently draw negative items.
	       log_dir --- the file directory to log the results.
	       test_num --- the number of evaluation per parameter settings.
	       dim --- the dimension of final node embeddings. 
	       pathNum --- the number of Metapaths for each type of node.
	
	Return: None
	'''

	print('----- Opt. hyperparams -----')
	print('lr: ' + str(lr))
	print('lamb: ' + str(lamb))
	print('batch_size: ' + str(batch_size))
	print('numNegative: ' + str(numNegative))
	print('featEmbedDim: ' + str(featEmbedDim))
	print('idenEmbedDim: ' + str(idenEmbedDim))
	print('Dim: ' + str(dim))
	print('pathNum: ' + str(pathNum))

	nonlinearity = tf.nn.relu
	pathInfos = genPathInfo(nonlinearity, featEmbedDim, idenEmbedDim, pathNum)
	nb_epochs = 30
	patience = 99
	warmup_step = 10.0 
	checkpt_file = log_dir+test_num+'/model.ckpt'
	logfile = open(log_dir+'RecoGCN','a')
	logfile.write(test_num+'\t'+'lr={},lamb={},batch_size={},numNegative={},featEmbedDim={},idenEmbedDim={},outputDim={},pathNum={}\t'.format(lr,lamb,batch_size,numNegative,featEmbedDim,idenEmbedDim,dim,pathNum))

	with tf.Graph().as_default():
		with tf.name_scope('input'):
			user_in = tf.placeholder(dtype=tf.int32, shape=(batch_size))
			agent_in = tf.placeholder(dtype=tf.int32, shape=(batch_size))
			item_in = tf.placeholder(dtype=tf.int32, shape=(batch_size))
			neg_item_in = tf.placeholder(dtype=tf.int32, shape=(numNegative))
			ffd_drop = tf.placeholder(dtype=tf.float32, shape=())   
			lr_in = tf.placeholder(dtype=tf.float32, shape=())  

		print('Start building model.')

		model = RecoGCN(userCnt, agentCnt, itemCnt, pathInfos, adj, numNegative, featEmbedDim, idenEmbedDim, userFeature, agentFeature, itemFeature, dim,
			batchUser=user_in, batchAgent=agent_in, batchItem=item_in, batchNeg=neg_item_in, inDrop=ffd_drop, lr=lr_in, l2_coef=lamb)
		print('Finish building model.')
		
		ndcg = model.ndcg
		mrr = model.mrr
		hr1 = model.hr1
		hr3 = model.hr3
		hr5 = model.hr5
		hr10 = model.hr10
		loss = model.loss
		train_op = model.train_op 

		saver = tf.train.Saver()

		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

		vlss_mn = np.inf
		vacc_mx = 0.0
		curr_step = 0

		with tf.Session(config=config) as sess:
			merged = tf.summary.merge_all()
			#train_writer = tf.summary.FileWriter(log_dir+test_num+'/train', sess.graph)
			train_writer = tf.summary.FileWriter(log_dir+test_num+'/train')
			test_writer = tf.summary.FileWriter(log_dir+test_num+'/test')

			sess.run(init_op)
			global_step_train = 0
			global_step_val = 0

			for epoch in range(nb_epochs):
				train_loss = []
				train_mrr = []
				train_ndcg = []
				train_hr10 = []
				train_hr5 = []
				train_hr3 = []
				train_hr1 = []

				tr_step = 0
				tr_size = len(train_sample)
				lr_actual = lr*(epoch/warmup_step)*5 if epoch<=warmup_step else lr*5*pow(epoch,-0.5)
				start = time.time()
				
				while tr_step * batch_size < tr_size:
					if (tr_step+1) * batch_size >= tr_size:
						batchUser = [item[0] for item in train_sample[-batch_size:]]
						batchAgent = [item[1] for item in train_sample[-batch_size:]]
						batchItem = [item[2] for item in train_sample[-batch_size:]]
						batchNeg = aliasTable.draw(numNegative)
					else:
						batchUser = [item[0] for item in train_sample[tr_step*batch_size:(tr_step+1)*batch_size]]
						batchAgent = [item[1] for item in train_sample[tr_step*batch_size:(tr_step+1)*batch_size]]
						batchItem = [item[2] for item in train_sample[tr_step*batch_size:(tr_step+1)*batch_size]]
						batchNeg = aliasTable.draw(numNegative)

					summary, _, loss_value_tr, mrr_tr, ndcg_tr, hr10_tr, hr5_tr, hr3_tr, hr1_tr = sess.run([merged, train_op, loss, mrr, ndcg, hr10, hr5, hr3, hr1],
					feed_dict={user_in: batchUser, agent_in: batchAgent, item_in:batchItem, neg_item_in:batchNeg, ffd_drop: 0.0, lr_in:lr_actual})
					
					if global_step_train%20==0:
						train_writer.add_summary(summary, global_step_train)

					train_loss.append(loss_value_tr)
					train_mrr.append(mrr_tr)
					train_ndcg.append(ndcg_tr)
					train_hr10.append(hr10_tr)
					train_hr5.append(hr5_tr)
					train_hr3.append(hr3_tr)
					train_hr1.append(hr1_tr)
					tr_step += 1
					global_step_train += 1

				train_loss_avg = np.mean(train_loss)
				train_mrr_avg = np.mean(train_mrr)
				train_ndcg_avg = np.mean(train_ndcg)
				train_hr10_avg = np.mean(train_hr10)
				train_hr5_avg = np.mean(train_hr5)
				train_hr3_avg = np.mean(train_hr3)
				train_hr1_avg = np.mean(train_hr1)

				val_loss = []
				val_mrr = []
				val_ndcg = []
				val_hr10 = []
				val_hr5 = []
				val_hr3 = []
				val_hr1 = []
				user_coef_list = []
				agent_coef_list = []
				item_coef_list = []

				vl_step = 0
				vl_size = len(val_sample)
				while vl_step * batch_size < vl_size:
					if (vl_step+1) * batch_size >= vl_size:
						batchUser = [item[0] for item in val_sample[-batch_size:]]
						batchAgent = [item[1] for item in val_sample[-batch_size:]]
						batchItem = [item[2] for item in val_sample[-batch_size:]]
						batchNeg = aliasTable.draw(numNegative)
					else:
						batchUser = [item[0] for item in val_sample[vl_step*batch_size:(vl_step+1)*batch_size]]
						batchAgent = [item[1] for item in val_sample[vl_step*batch_size:(vl_step+1)*batch_size]]
						batchItem = [item[2] for item in val_sample[vl_step*batch_size:(vl_step+1)*batch_size]]
						batchNeg = aliasTable.draw(numNegative)

					summary, loss_value_vl, mrr_vl, ndcg_vl, hr10_vl, hr5_vl, hr3_vl, hr1_vl, user_coef, agent_coef, item_coef = sess.run([merged, loss, mrr, ndcg, hr10, hr5, hr3, hr1, model.user_coef, model.agent_coef, model.item_coef],
					feed_dict={user_in: batchUser, agent_in: batchAgent, item_in:batchItem, neg_item_in:batchNeg, ffd_drop: 0.0})

					test_writer.add_summary(summary, global_step_val)
					
					val_loss.append(loss_value_vl)
					val_mrr.append(mrr_vl)
					val_ndcg.append(ndcg_vl)
					val_hr10.append(hr10_vl)
					val_hr5.append(hr5_vl)
					val_hr3.append(hr3_vl)
					val_hr1.append(hr1_vl)
					user_coef_list.append(user_coef)
					agent_coef_list.append(agent_coef)
					item_coef_list.append(item_coef)

					vl_step += 1
					global_step_val += 1

				val_loss_avg = np.mean(val_loss)
				val_mrr_avg = np.mean(val_mrr)
				val_ndcg_avg = np.mean(val_ndcg)
				val_hr10_avg = np.mean(val_hr10)
				val_hr5_avg = np.mean(val_hr5)
				val_hr3_avg = np.mean(val_hr3)
				val_hr1_avg = np.mean(val_hr1)

				elapsed = (time.time() - start)				
				print('Time elapsed = %.2f mins, Training: loss = %.5f, mrr = %.5f, ndcg = %.5f, hr1 = %.5f, hr3 = %.5f, hr5 = %.5f, hr10 = %.5f | Val:loss = %.5f, mrr = %.5f, ndcg = %.5f, hr1 = %.5f, hr3 = %.5f, hr5 = %.5f, hr10 = %.5f' %
						(elapsed/60, train_loss_avg, train_mrr_avg, train_ndcg_avg, train_hr1_avg, train_hr3_avg, train_hr5_avg, train_hr10_avg,
						val_loss_avg, val_mrr_avg, val_ndcg_avg, val_hr1_avg, val_hr3_avg, val_hr5_avg, val_hr10_avg))

				if val_mrr_avg >= vacc_mx:
					vacc_early_model = val_mrr_avg
					vlss_early_model = val_loss_avg
					saver.save(sess, checkpt_file)
					vacc_mx = np.max((val_mrr_avg, vacc_mx))
					vlss_mn = np.min((val_loss_avg, vlss_mn))
					curr_step = 0
				else:
					curr_step += 1
					if curr_step == patience:
						print('Early stop! Min loss: ', vlss_mn, ', Max mrr: ', vacc_mx)
						print('Early stop model validation loss: ', vlss_early_model, ', mrr: ', vacc_early_model)
						logfile.write('Min val loss:'+str(vlss_mn)+'; Max Val mrr:'+str(vacc_mx)+'\t')
						break
						

			saver.restore(sess, checkpt_file)

			ts_size = len(test_sample)
			ts_step = 0

			ts_loss = []
			ts_mrr = []
			ts_ndcg = []
			ts_hr5 = []
			ts_hr10 = []
			ts_hr3 = []
			ts_hr1 = []
			user_coef_list = []
			agent_coef_list = []
			item_coef_list = []
			while ts_step * batch_size < ts_size:
				if (ts_step+1) * batch_size >= ts_size:
					batchUser = [item[0] for item in test_sample[-batch_size:]]
					batchAgent = [item[1] for item in test_sample[-batch_size:]]
					batchItem = [item[2] for item in test_sample[-batch_size:]]
					batchNeg = aliasTable.draw(numNegative)
				else:
					batchUser = [item[0] for item in test_sample[ts_step*batch_size:(ts_step+1)*batch_size]]
					batchAgent = [item[1] for item in test_sample[ts_step*batch_size:(ts_step+1)*batch_size]]
					batchItem = [item[2] for item in test_sample[ts_step*batch_size:(ts_step+1)*batch_size]]
					batchNeg = aliasTable.draw(numNegative)

				loss_value_ts, mrr_ts, ndcg_ts, hr10_ts, hr5_ts, hr3_ts, hr1_ts, user_coef, agent_coef, item_coef = sess.run([loss, mrr, ndcg, hr10, hr5, hr3, hr1, model.user_coef, model.agent_coef, model.item_coef],
					feed_dict={user_in: batchUser, agent_in: batchAgent, item_in:batchItem, neg_item_in:batchNeg, ffd_drop: 0.0})

				user_coef_list.append(user_coef)
				agent_coef_list.append(agent_coef)
				item_coef_list.append(item_coef)

				ts_loss.append(loss_value_ts)
				ts_mrr.append(mrr_ts)
				ts_ndcg.append(ndcg_ts)
				ts_hr5.append(hr5_ts)
				ts_hr10.append(hr10_ts)
				ts_hr3.append(hr3_ts)
				ts_hr1.append(hr1_ts)
				ts_step += 1

			with h5py.File(log_dir + test_num + str(lr)+str(lamb) + '_attn_test.hdf5', 'w') as f:
			 	f.create_dataset("user_coef", data=np.concatenate(user_coef_list))
			 	f.create_dataset("agent_coef", data=np.concatenate(agent_coef_list))
			 	f.create_dataset("item_coef", data=np.concatenate(item_coef_list))
				

			print('Test loss:', np.mean(ts_loss), '; Test mrr:', np.mean(ts_mrr), '; Test ndcg:', np.mean(ts_ndcg), '; Test hr1:', np.mean(ts_hr1), '; Test hr3:',np.mean(ts_hr3),'; Test hr5:', np.mean(ts_hr5), '; Test hr10:', np.mean(ts_hr10))
			logfile.write('Test loss:'+ str(np.mean(ts_loss))+ '; Test mrr:'+ str(np.mean(ts_mrr))+ '; Test ndcg:'+ str(np.mean(ts_ndcg))+ '; Test hr1:'+ str(np.mean(ts_hr1))+ '; Test hr3:'+ str(np.mean(ts_hr3))+ '; Test hr5:'+ str(np.mean(ts_hr5))+ '; Test hr10:'+ str(np.mean(ts_hr10))+'\n')

			logfile.close()
			sess.close()

def main(log_dir):
	dataset = './Dataset'
	print('Dataset: ' + dataset)

	adj = {0:{}, 1:{}, 2:{}, 3:{}}
	with h5py.File(dataset, 'r') as f:
		adj[0][1] = f['adj01'][:]
		adj[1][0] = f['adj10'][:]
		adj[0][2] = f['adj02'][:]
		adj[2][0] = f['adj20'][:]
		adj[0][3] = f['adj03'][:]
		adj[3][0] = f['adj30'][:]

		train_sample = f['train_sample'][:]
		val_sample = f['val_sample'][:]
		test_sample = f['test_sample'][:]
		
		item_freq = f['item_freq'][:]
		user_feature = f['user_feature'][:]
		agent_feature = f['agent_feature'][:]
		item_feature = f['item_feature'][:]

		userCnt = f['userCnt'][()]
		agentCnt = f['agentCnt'][()]
		itemCnt = f['itemCnt'][()]

	nodeid = np.array([int(item[0]) for item in item_freq])
	freq = np.array([item[1] for item in item_freq])
	aliasTable = AliasTable(weights=freq, keys=nodeid)


	batch_size = 400
	numNegative = 100
	featEmbedDim = 64
	idenEmbedDim = 64
	lr = 0.0001
	dim = 128
	lamb = 0.5
	testrun = 3
	pathNums = 7

	# evalute the model several time to report the stable results
	for count in range(testrun):
		evaluate_model(lr, lamb, adj, batch_size, userCnt, agentCnt, itemCnt, numNegative, featEmbedDim, idenEmbedDim, user_feature, agent_feature, item_feature, train_sample, val_sample, test_sample, aliasTable, log_dir, str(count), dim, pathNums)


if __name__=='__main__':
	main('./result/')