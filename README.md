## This is the code of submission 485 in CIKM2019


## evaluate model
```
python train.py 
```

## example output
```
Time elapsed = 6.89 mins, Training: loss = 389.51047, mrr = 0.63130, ndcg = 0.71369, hr1 = 0.50939, hr3 = 0.69945, hr5 = 0.78027, hr10 = 0.87522 | Val:loss = 2172.41870, mrr = 0.25467, ndcg = 0.40172, hr1 = 0.15110, hr3 = 0.25807, hr5 = 0.33136, hr10 = 0.45893
```

The variant models below had been supported: 

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
