## This is the code of submission 485 in CIKM2019


## evaluate model
```
python train.py 
```

## example training output
```
Time elapsed = 6.89 mins, Training: loss = 389.51047, mrr = 0.63130, ndcg = 0.71369, hr1 = 0.50939, hr3 = 0.69945, hr5 = 0.78027, hr10 = 0.87522 | Val:loss = 2172.41870, mrr = 0.25467, ndcg = 0.40172, hr1 = 0.15110, hr3 = 0.25807, hr5 = 0.33136, hr10 = 0.45893
```

## example evaluation result
```
0	lr=0.0001,lamb=0.55,batch_size=400,numNegative=100,featEmbedDim=64,idenEmbedDim=64,outputDim=128,pathNum=7	Test loss:2033.5934; Test mrr:0.25339168; Test ndcg:0.3976466; Test hr1:0.14939758; Test hr3:0.2633283; Test hr5:0.34176204; Test hr10:0.46430722
```

These variant models below had been supported: 

- ReGCN
- ReGCN_{MP}
- RecoGCN

## dependencies (other versions may also work):
- python == 3.6
- tensorflow == 1.13.1
- numpy == 1.16.3
- h5py == 2.9.0
- GPUtil ==1.4.0
- setproctitle == 1.1.10
