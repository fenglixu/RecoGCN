## This repo contains a tensorflow implementation of RecoGCN and the experiment dataset


## Running the RecoGCN model
```
python train.py 
```

## Example training output
```
Time elapsed = 6.89 mins, Training: loss = 389.51047, mrr = 0.63130, ndcg = 0.71369, hr1 = 0.50939, hr3 = 0.69945, hr5 = 0.78027, hr10 = 0.87522 | Val:loss = 2172.41870, mrr = 0.25467, ndcg = 0.40172, hr1 = 0.15110, hr3 = 0.25807, hr5 = 0.33136, hr10 = 0.45893
```

## Example evaluation result
```
0	lr=0.0001,lamb=0.55,batch_size=400,numNegative=100,featEmbedDim=64,idenEmbedDim=64,outputDim=128,pathNum=7	Test loss:2033.5934; Test mrr:0.25339168; Test ndcg:0.3976466; Test hr1:0.14939758; Test hr3:0.2633283; Test hr5:0.34176204; Test hr10:0.46430722
```

These variant models below had been supported: 

- ReGCN
- ReGCN_{MP}
- RecoGCN

## Dependencies (other versions may also work):
- python == 3.6
- tensorflow == 1.13.1
- numpy == 1.16.3
- h5py == 2.9.0
- GPUtil ==1.4.0
- setproctitle == 1.1.10

## Dataset
You can download the experiment data from [Here](https://drive.google.com/file/d/1ZwlB3_NsbOjVM4tVKIwchQwkZbmYJUJx/view?usp=sharing). An example loading code is provided as follow.
```
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
```

The data structure is explained as follow.

```adj[x][y]``` denotes the adjancy relationship from x to y. Here, 0 stands for user, 1 is selling agent, 2 and 3 are two kinds of items. The shape of ```adj[x][y]``` is ```[Num_of_node_x ,maximum_link]```. Each line stores the node ids of type y who are linked with node x. Note that maximum_link should be the same for each of these relations. 

```train_sample, val_sample, test_sample``` are triplet of ```[user, selling_agent, item]``` pairs. Each type of node is encoded from 0. 

```item_freq``` is ```[item_id, item_frequency]``` matrix denotes the occur frequency of each item in train set.

```user_feature, agent_feature, item_feature``` are three featrue matrix of shape ```[node_num, feature_num]```. Here features for each node are multi-hot encoded, and different type of node can have different feature numbers. 

## Citation
If you use our code or dataset in your research, please cite:
```
@inproceedings{xu2019relation,
  title={Relation-aware graph convolutional networks for agent-initiated social e-commerce recommendation},
  author={Xu, Fengli and Lian, Jianxun and Han, Zhenyu and Li, Yong and Xu, Yujian and Xie, Xing},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={529--538},
  year={2019}
}
```
