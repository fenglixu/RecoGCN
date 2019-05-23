## This is the code of submission 485 in CIKM2019

## prepare data
You can get the data from amazon website and process it using the script
```
sh prepare_data.sh
```
When you see the files below, you can do the next work. 
- cat_voc_large.pkl 
- mid_voc_large.pkl 
- uid_voc_large.pkl 
- local_train 
- local_test
- reviews-info
- item-info

## train model
```
python script/train.py train [model name] 
```

## test model
```
python script/train.py test [model name] 
```

The variant models below had been supported: 

- ReGCN
- ReGCN_{MP}
- RecoGCN

## dependencies (other versions may also work):
- python==2.7
- tensorflow==1.4.1
- keras==2.1.5
- numpy==1.15.4
