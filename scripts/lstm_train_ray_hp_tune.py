import os
import random
import sys
import argparse
import csv
import math
from functools import partial
import multiprocessing as mp
import tempfile as tmp
from filelock import FileLock
from tqdm import tqdm

import numpy as np
import pandas as pd

from azureml.core import Run

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim

import ray
from ray import air, tune
from ray.air import session
from ray.air.config import ScalingConfig
from ray.train.torch import TorchCheckpoint
from ray.tune.schedulers import ASHAScheduler, AsyncHyperBandScheduler

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, multilabel_confusion_matrix, roc_auc_score, hamming_loss, roc_curve, auc
import seaborn as sns

random.seed(1234)
np.random.seed(5678)
torch.manual_seed(0)


class TermsDataset(Dataset):
    def __init__(self, file_path, transform=None, target_transform=None):
        self.terms_n_labels = pd.read_csv(file_path)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.terms_n_labels)

    def __getitem__(self, idx):
        terms = np.array(self.terms_n_labels.iloc[idx, 0:6])
        zero_idx = np.where(terms == 0)[0]
        if len(zero_idx):
            terms_len = zero_idx[0].item()
        else:
            terms_len = 6
        label = np.array(self.terms_n_labels.iloc[idx, 6:])
        if self.transform:
            terms = self.transform(terms)
        if self.target_transform:
            label = self.target_transform(label)
        return terms, terms_len, label


def get_train_valid_data_loaders(data_path, batch_size=64, transform=None, target_transform=None):
    data_splits = ['train', 'valid', 'test']

    # We add FileLock here because multiple workers will want to download data,
    # and this may cause overwrites since DataLoader is not threadsafe.
    # with FileLock(os.path.expanduser(os.path.join(data_path, "data.lock"))):

    train_dataset = TermsDataset(os.path.join(data_path, data_splits[0] + '.csv'), transform=transform,
                                       target_transform=target_transform)
    valid_dataset = TermsDataset(os.path.join(data_path, data_splits[1] + '.csv'), transform=transform,
                                       target_transform=target_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader


class term_topic_LSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, activation=F.relu, emb_dim=300, hidden_dim=128, num_layers=1, bidir=False):
        super(term_topic_LSTM, self).__init__()

        self.bidir = bidir
        mult = 2 if bidir else 1
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidir)
#         self.act = nn.GELU()
#         self.act = nn.ReLU()
        self.act = activation
#         self.drop = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm1d(num_features=mult * hidden_dim)
        self.fc1 = nn.Linear(mult * hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, terms, terms_len):
        embs = self.word_embeddings(terms)

        packed_input = pack_padded_sequence(embs, terms_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (ph_n, pc_n) = self.lstm(packed_input)
        output, sizes = pad_packed_sequence(packed_output, batch_first=True)

        if self.bidir:
            out_forward = output[range(len(output)), terms_len - 1, :self.hidden_dim]  # output[:, last_hs, :h_dim] => for the forward LSTM, take the last hidden state, i.e., last row of output tensor
            out_reverse = output[:, 0, self.hidden_dim:]  # output[:, 1st_hs, h_dim:] => for the backward LSTM, take the first hidden state, i.e., first row, 2nd half of output tensor
            out_reduced = torch.cat((out_forward, out_reverse), 1)
        else:
            out_reduced = output[range(len(output)), terms_len - 1, :self.hidden_dim]  # output[:, [4,3], :] => for the forward LSTM, take the last hidden state, i.e., last row of output tensor

        act1_res = self.act(out_reduced)
        bn1_res = self.bn1(act1_res)
        fc1_res = self.fc1(bn1_res)

        act2_res = self.act(fc1_res)
        bn2_res = self.bn2(act2_res)
        logits = self.fc2(bn2_res)
        logits = torch.squeeze(logits, 1)
        prob = torch.sigmoid(logits)
        return prob


# Save and Load Functions
def save_checkpoint(save_path, model, optimizer, valid_loss):
    if save_path is None:
        return
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


# def load_checkpoint(load_path, model, optimizer, device):
#     if load_path is None:
#         return
#     state_dict = torch.load(load_path, map_location=device)
#     print(f'Model loaded from <== {load_path}')
#     model.load_state_dict(state_dict['model_state_dict'])
#     optimizer.load_state_dict(state_dict['optimizer_state_dict'])
#     return state_dict['valid_loss']


def load_checkpoint(checkpoint, model, optimizer=None):
    state_dict = checkpoint.to_dict()
    print(f'Model loaded from <== {checkpoint.uri}')
    res = model.load_state_dict(state_dict['model'])
    print(f'model load result: {res}')
    if optimizer:
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    return model, optimizer


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path is None:
        return
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    torch.save(state_dict, save_path)
    print(f'Metrics saved to ==> {save_path}')


def load_metrics(load_path, device):
    if load_path is None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f'Metrics loaded from <== {load_path}')
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def train(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = 0.0
    for batch_idx, (termsb, terms_lenb, labelsb) in enumerate(train_loader):
        labelsb = labelsb.to(device, dtype=torch.float)
        termsb = termsb.to(device)
        terms_lenb = terms_lenb.to(device)

        optimizer.zero_grad()
        output = model(termsb, terms_lenb)
        loss = criterion(output, labelsb)
        loss.backward()
        optimizer.step()

        # update running values
        train_loss += loss.item()
    return train_loss / len(train_loader)


def validate(model, criterion, valid_loader, device):
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch_idx, (termsb, terms_lenb, labelsb) in enumerate(valid_loader):
            labelsb = labelsb.to(device, dtype=torch.float)
            termsb = termsb.to(device)
            terms_lenb = terms_lenb.to(device)
            output = model(termsb, terms_lenb)

            loss = criterion(output, labelsb)
            valid_loss += loss.item()
        return valid_loss / len(valid_loader)


# Trainer Function
def lstm_trainer(config,
                 word_to_idx, class_to_idx,
                 data_path, transform=None, target_transform=None
                 ):

    with FileLock('/dev/shm/data.lock'):
    # with lock:
        train_loader, valid_loader = get_train_valid_data_loaders(data_path, config["batch_size"], transform,
                                                                  target_transform)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = term_topic_LSTM(
        vocab_size=len(word_to_idx) + 1, num_classes=len(class_to_idx),
        activation=config["activation"],
        emb_dim=config["emb_dim"], hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"], bidir=config["bidir"]
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.BCELoss()

    # initialize running values
    # train_losses = []
    # valid_losses = []
    best_valid_loss = float("Inf")
    best_epoch = 0

    for epoch in range(config["max_epochs"]):
        train_loss = train(model, optimizer, criterion, train_loader, device)
        # train_losses.append(train_loss)

        valid_loss = validate(model, criterion, valid_loader, device)
        # valid_losses.append(valid_loss)

        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
            #             "train_losses": train_losses,
            #             "valid_losses": valid_losses,
        }
        checkpoint = TorchCheckpoint.from_state_dict(model.state_dict())
        session.report(metrics, checkpoint=checkpoint)  # Report metrics (and a checkpoint) to Tune


def driver(args, data_path, class_to_idx, word_to_idx, result_dir='~/ray_results', exp_name='exp'):
    n_epochs = 3 if args.smoke_test else 10
    grace_period = 2 if args.smoke_test else 3
    # num_samples = 1 if args.smoke_test else 20
    num_samples = 1

    # schedulers for early stopping
    scheduler = AsyncHyperBandScheduler(
        # metric="test_accuracy",
        # mode='max',
        max_t=n_epochs,
        grace_period=grace_period,
        reduction_factor=2,
    )

    checkpoint_cfg = air.CheckpointConfig(
        checkpoint_score_attribute="valid_loss",
        checkpoint_score_order='min',
        num_to_keep=2,
    )
    failure_cfg = air.FailureConfig(max_failures=2)
    sync_cfg = tune.SyncConfig(
        upload_dir=result_dir
        # syncer=None
    )

    # NOTE: can use ScalingConfig to limit total cpu's and cpu's per trial
    # NOTE: exp hangs (trials do not run) when num_workers is given
    # working configs:
    # 1) trainer_resources={"CPU": 8}, resources_per_worker={"CPU": 1} - takes 800s, 1005s ASHA
    # 2) trainer_resources={"CPU": 8}, resources_per_worker={"CPU": 2} - takes 964s 1020s ASHA
    # 3) trainer_resources={"CPU": 8}, resources_per_worker={"CPU": 4} - takes 885s, 1082s ASHA
    scaling_config = ScalingConfig(
        trainer_resources={"CPU": 8},
        # num_workers=8,
        # use_gpu=False,
        resources_per_worker={"CPU": 2},
    )

    resources_per_trial = {"cpu": 1, "gpu": int(args.use_gpu) if torch.cuda.is_available() else 0}  # set this for GPUs - takes 989s ASHA

    reporter = tune.CLIReporter()

    lstm_trainer_p = partial(lstm_trainer,
                             word_to_idx=word_to_idx, class_to_idx=class_to_idx,
                             data_path=data_path, transform=torch.from_numpy, target_transform=torch.from_numpy
                             )
    tuner = tune.Tuner(
        trainable=tune.with_resources(lstm_trainer_p, resources=resources_per_trial),
        #         trainable=tune.with_resources(lstm_trainer_p, resources=scaling_config),
        tune_config=tune.TuneConfig(
            metric="valid_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=air.RunConfig(
            name=exp_name,
            local_dir=result_dir,
            checkpoint_config=checkpoint_cfg,
            failure_config=failure_cfg,
            sync_config=sync_cfg,
            stop={
                # "valid_loss": 1e-5,
                "training_iteration": n_epochs,
            },
            progress_reporter=reporter,
            verbose=3,
            log_to_file=True,
        ),
        # param_space={
        #     "lr": tune.loguniform(1e-4, 1e-2),
        #     "emb_dim": tune.grid_search([64, 96, 128, 192]),
        #     "hidden_dim": tune.grid_search([64, 96, 128]),
        #     "activation": tune.grid_search([F.relu, F.gelu]),
        #     "num_layers": tune.grid_search([1, 2, 3]),
        #     "bidir": tune.grid_search([False, True]),
        #     "batch_size": tune.grid_search([64, 128, 256, 512]),
        #     "max_epochs": n_epochs,
        # },
        param_space={
            "lr": 0.0007,
            "emb_dim": 192,
            "hidden_dim": 128,
            "activation": F.gelu,
            "num_layers": 3,
            "bidir": True,
            "batch_size": 64,
            "max_epochs": n_epochs,
        },
    )
    results = tuner.fit()

    print('Best result is:', results.get_best_result())
    print("Best config is:", results.get_best_result().config)

    return tuner, results


def plot_results(result):
    df = result.metrics_dataframe[["training_iteration", "train_loss", "valid_loss"]]
    # ax = None
    cfg = result.config
    title = f"lr={cfg['lr']:.3f}, emb_dim={cfg['emb_dim']}, hid_dim={cfg['hidden_dim']}, num_layers={cfg['num_layers']}, bidir={cfg['bidir']}, batch_size={cfg['batch_size']}"
    plt.plot(df["training_iteration"], df["train_loss"], label='Train Loss')
    plt.plot(df["training_iteration"], df["valid_loss"], label='Valid Loss')
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()


def euc_dist(p1=(0.0, 0.0), p2=(1.0, 0.0)):
    # print(p1, p2)
    sq_dist = math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2)
    # print(sq_dist)
    return math.sqrt(sq_dist)


def get_max_dist(x, y):
    assert len(x) == len(y)
    dists = []
    for pxy in zip(x, y):
        dists.append(euc_dist(pxy))
    return dists, np.argmax(dists)


# Evaluation Function
def evaluate(model, class_dict, test_loader, device, threshold=0.5, title_max_chars=25):
    target_names = list(class_dict.keys())
    labels = list(class_dict.values())

    y_proba = []
    y_pred = []
    y_true = []

    print(f'Evaluating model with test data with threshold={threshold}')
    model.eval()
    with torch.no_grad():
        for termsb, terms_lenb, labelsb in test_loader:
            labelsb = labelsb.to(device, dtype=torch.float)
            termsb = termsb.to(device)
            terms_lenb = terms_lenb.to(device)
            output = model(termsb, terms_lenb)
            y_proba.extend(output.cpu().detach().numpy().tolist())

            output = (output > threshold).int()
            y_pred.extend(output.cpu().tolist())
            y_true.extend(labelsb.cpu().tolist())

    # In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    print('Accuracy (% of exactly matched samples):', accuracy_score(y_true, y_pred))

    # hamming loss is the fraction of labels that are incorrectly predicted
    print('Hamming Loss: (% of incorrect labels in all samples)', hamming_loss(y_true, y_pred))

    # roc_auc score per class
    print('ROC-AUC Score: (per class)\n', roc_auc_score(y_true, y_proba, average=None))

    print(f'Classification Report (threshold = {threshold}):')
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=4))
    cr = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, digits=4)

    #     cm = multilabel_confusion_matrix(y_true, y_pred, labels=[1,0])
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    mcm = np.flip(mcm, axis=(1, 2))  # reverse orders of rows and columns to have TP in (0,0), FN in (0,1), FP in (1,0), TN in (1,1)
    print('mcm shape: ', mcm.shape)
    print(mcm)

    nrows, ncols = 4 * 2, 6
    eff_cols = ncols // 2  # = 3
    fig_buf = 1
    sp_size = 4
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                           figsize=(sp_size * ncols + fig_buf, sp_size * nrows))
    fig.suptitle(f'Confusion Matrices per Topic (threshold = {threshold})', fontsize=22)
    plt.subplots_adjust(left=0.1,
                        bottom=0.05,
                        right=0.9,
                        top=0.95,
                        wspace=0.3,
                        hspace=0.3,
                        )
    for i in range(mcm.shape[0]):
        row = i // eff_cols
        col = 2 * (i % eff_cols)
        sns.heatmap(mcm[i], annot=True, ax=ax[row, col], xticklabels=[1, 0], yticklabels=[1, 0], cmap='Blues',
                    cbar=False, fmt="d")
        title = target_names[i][:title_max_chars] + ('...' if len(target_names[i]) > title_max_chars else '')
        ax[row, col].set_title(title, fontsize=8)
        ax[row, col].set_xlabel('Predicted')
        ax[row, col].set_ylabel('Ground Truth')

        # Determine fpr, tpr, th and plot custom roc curve
        fpr, tpr, th = roc_curve(np.array(y_true)[:, i], np.array(y_proba)[:, i])
        roc_auc = auc(fpr, tpr)
        th[0] = min(th[0], 1.0)
        dists, max_idx = get_max_dist(fpr, tpr)
        chance = [i for i in np.linspace(0, 1, len(tpr))]

        col = 2 * (i % eff_cols) + 1
        ax[row, col].plot(fpr, tpr, label='estimator (AUC = {:.3f})'.format(roc_auc))
        ax[row, col].plot(chance, chance, 'k--', label='chance')
        ax[row, col].text(0, 0.9, 'max @({:.2f}, {:.2f})\nth={:.2f}'.format(fpr[max_idx], tpr[max_idx], th[max_idx]))
        ax[row, col].legend(loc='lower right', prop=dict(size=8))
        ax[row, col].set_title('ROC ' + title, fontsize=8)
        ax[row, col].set_xlabel('False Positive Rate')
        ax[row, col].set_ylabel('True Positive Rate')

    plt.show()
    return cr


def main(args):
    # data_path = "/dbfs/mnt/output-data/labelled/"
    data_path = args.input_data
    print('input_data folder: ', data_path)

    # Read word-to-idx and class-to-idx dictionaries
    with open(os.path.join(data_path, 'word_to_idx.csv')) as csv_file:
        reader = csv.reader(csv_file)
        word_to_idx = dict(reader)

    word_to_idx = {k: int(v) for k, v in word_to_idx.items()}   # convert idx to int
    idx_to_word = {v: k for k, v in word_to_idx.items()}

    with open(os.path.join(data_path, 'class_to_idx.csv')) as csv_file:
        reader = csv.reader(csv_file)
        class_to_idx = dict(reader)

    class_to_idx = {k: int(v) for k, v in class_to_idx.items()}   # convert idx to int
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    exp_name = 'lstm_train_hp'
    output_dir = os.path.join(args.train_output, 'ray_results')
    os.makedirs(output_dir, exist_ok=True)
    tuner, results = driver(args, data_path, class_to_idx, word_to_idx, output_dir, exp_name)

    if results.errors:
        print("ERROR: One or more of the trials failed!")
    else:
        print("No errors!")

    best_result = results.get_best_result()

    # Plot and evaluate results
    plot_results(best_result)
    best_cfg = best_result.config
    device = torch.device('cpu')

    best_model = term_topic_LSTM(
        vocab_size=len(word_to_idx) + 1, num_classes=len(class_to_idx),
        activation=best_cfg['activation'],
        emb_dim=best_cfg['emb_dim'], hidden_dim=best_cfg['hidden_dim'],
        num_layers=best_cfg['num_layers'], bidir=best_cfg['bidir']
    ).to(device)

    best_model, _ = load_checkpoint(best_result.checkpoint, best_model)

    func = torch.from_numpy
    test_ds = TermsDataset(os.path.join(data_path, 'test.csv'), transform=func, target_transform=func)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    # NOTE:
    # threshold > 0.5 => improves precision
    # threshold < 0.5 => improves recall
    cr = evaluate(model=best_model, class_dict=class_to_idx, test_loader=test_dl, device=device, threshold=0.5)
    print('Class Metrics (json)\n', cr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM PyTorch Hyperparam tuning")
    parser.add_argument("--input_data", type=str, dest='input_data', help="input dataset")
    parser.add_argument("--train_output", type=str, dest='train_output', help="train output location")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="enables CUDA training")
    parser.add_argument("--smoke-test", action="store_true", help="Finish quickly for testing")
    # parser.add_argument("--ray-address", type=str, help="The address of the cluster.")

    args = parser.parse_args()
    run = Run.get_context()

    tmpfile = tmp.TemporaryFile()
    lock = mp.Lock()

    print('INFO: Initializing ray runtime ...')
    # ray.init(address=args.ray_address, num_cpus=8 if args.smoke_test else None, object_store_memory=195*1024*1024*1024)
    ray.init()

    pd.set_option('display.max_colwidth', 256)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    main(args)

    # Shut down ray runtime server
    if ray.is_initialized():
        print('INFO: Ray runtime exists - shutting down runtime...')
        ray.shutdown()  # call this if there are existing ray instances (initialized via ray.init())
        print('INFO: Ray runtime shut down complete')

    run.complete()

