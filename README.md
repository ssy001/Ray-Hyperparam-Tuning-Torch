# Hyper-parameter Tuning of LSTM with Ray

This project explores hyper-parameter tuning of LSTM using Ray hyper-parameter optimization library. Tuning is performed on 
a) locally, b) non-Databricks compute cluster, and c) Databricks compute cluster

## LSTM Specifications

LSTM is trained to predict multi-label topic classification of short words

Code allows unidirectional or bidirectional LSTM configurations to be trained, attached to a fully-connected torch nn.Linear head

LSTM is trained using Adam optimizer and binary cross-entropy loss 

## Ray Training Specifications

Ray driver uses AsyncHyperBandScheduler to optimally schedule the training processes in the CPU or GPU cores. 

All training experiment results are stored in sub-epoch increments with all metadata. This allows stopping and resuming the training by running tune.Tuner.restore("save_path"). 

After each training epoch, the validation losses are evaluated, with an evaluation grace-period of 2 epochs, and the top-third are kept and proceeded to the next training round. 

Hyper-parameter search is performed over the learning rate, embedding dimension, hidden dimension, number of layers, activation function, bidirectionality, and batch size. 

## Results Evaluation

Best results can be retrieved by calling get_results() on the tuner instance, and then calling get_best_result() on the resulting instance. 

For multi-label classification, evaluations are on multi-label accuracy, hamming loss, ROC-AUC score per class, and multi-label confusion matrix. 

## Scripts and usage

1. To train LSTM with Ray locally or in a non-Databricks compute cluster, run the python script scripts/lstm_train_ray_hp_tune.py 

2. To train LSTM with Ray locally or in a Databricks compute cluster, run the Jupyter notebook notebooks/lstm_train_ray_hp_tune_dtb.ipynb

