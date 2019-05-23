## This is the code of paper 1329 in IJCAI2019

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

The model below had been supported: 

Baselines:
- ASVD
- DIN
- LSTM
- LSTMPP
- NARM
- CARNN
- Time1LSTM
- Time2LSTM
- Time3LSTM
- DIEN

Our models:
- A2SVD
- T_SeqRec
- TC_SeqRec_I
- TC_SeqRec_G
- TC_SeqRec
- SLi_Rec_Fixed
- SLi_Rec_Adaptive

## dependencies (other versions may also work):
- python==2.7
- tensorflow==1.4.1
- keras==2.1.5
- numpy==1.15.4
