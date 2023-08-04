import os
import sys
import argparse
import csv
import math
from functools import partial
from tqdm import tqdm

import numpy as np
import pandas as pd

from azureml.core import Run
import mlflow

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim

import ray

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, multilabel_confusion_matrix, roc_auc_score, hamming_loss, roc_curve, auc
import seaborn as sns

torch.manual_seed(1)
mlflow.autolog()


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


def get_words(np_arr, idx_to_word):
    all_words = []
    for r in np_arr:
        all_words.append(' '.join([idx_to_word.get(idx, '') for idx in r]).strip())
    return all_words


def get_classes(np_arr, idx_to_class):
    all_cls = []
    for r in np_arr:
        all_cls.append([idx_to_class[idx] for idx, val in enumerate(r) if val == 1])
    return all_cls


class term_topic_LSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, act, emb_dim=300, hidden_dim=128, num_layers=1, bidir=False):
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
        self.act = act
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


def load_checkpoint(load_path, model, optimizer, device):
    if load_path is None:
        return
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    return state_dict['valid_loss']


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


def plot_results(train_loss_list, valid_loss_list, global_steps_list):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Valid')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    # return fig
    mlflow.log_figure(fig, 'train-valid_loss.png')


def euc_dist(p1=(0.0, 0.0), p2=(1.0, 0.0)):
    print(p1, p2)
    sq_dist = math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2)
    print(sq_dist)
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
    #             print(type(output), output.shape)

    # In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    print('Accuracy (% of exactly matched samples):', accuracy_score(y_true, y_pred))
    mlflow.log_metric('Accuracy (% of exactly matched samples):', accuracy_score(y_true, y_pred))

    # hamming loss is the fraction of labels that are incorrectly predicted
    print('Hamming Loss: (% of incorrect labels in all samples)', hamming_loss(y_true, y_pred))
    mlflow.log_metric('Hamming Loss: (% of incorrect labels in all samples)', hamming_loss(y_true, y_pred))

    # roc_auc score per class
    print('ROC-AUC Score: (per class)\n', roc_auc_score(y_true, y_proba, average=None))
    # mlflow.log_metric('ROC-AUC Score: (per class)\n', roc_auc_score(y_true, y_proba, average=None))

    print(f'Classification Report (threshold = {threshold}):')
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=4))
    cr = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, digits=4)

    #     cm = multilabel_confusion_matrix(y_true, y_pred, labels=[1,0])
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    mcm = np.flip(mcm, axis=(1, 2))  # reverse orders of rows and columns to have TP in (0,0), FN in (0,1), FP in (1,0), TN in (1,1)
    print('mcm shape: ', mcm.shape)
    print('--- END evaluate() ---')
    # print(mcm)

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

    # plt.show()
    mlflow.log_figure(fig, 'Confusion_Matrices_and_ROCs.png')
    mlflow.log_dict(cr, 'classification_report.yaml')
    return cr


# Training Function
def train(model,
          optimizer,
          criterion,
          train_loader,
          valid_loader,
          model_path,
          device,
          num_epochs=5,
          eval_every=100,
          best_valid_loss=float("Inf")):
    # initialize running values
    train_loss = 0.0
    valid_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in tqdm(range(num_epochs)):
        #     for epoch in range(num_epochs):
        for termsb, terms_lenb, labelsb in train_loader:
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
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    # validation loop
                    for termsb, terms_lenb, labelsb in valid_loader:
                        labelsb = labelsb.to(device, dtype=torch.float)
                        termsb = termsb.to(device)
                        terms_lenb = terms_lenb.to(device)
                        output = model(termsb, terms_lenb)

                        loss = criterion(output, labelsb)
                        valid_loss += loss.item()

                # evaluation
                average_train_loss = train_loss / eval_every
                average_valid_loss = valid_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)
                mlflow.log_metric('train_loss', average_train_loss, step=global_step)
                mlflow.log_metric('valid_loss', average_valid_loss, step=global_step)

                # resetting running values
                train_loss = 0.0
                valid_loss = 0.0
                model.train()

                # print progress
                print('\nEpoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(os.path.join(model_path, 'model.pt'), model, optimizer, best_valid_loss)
                    save_metrics(os.path.join(model_path, 'metrics.pt'), train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(os.path.join(model_path, 'metrics.pt'), train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

    return train_loss_list, valid_loss_list, global_steps_list


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

    p = {
        'batch_size': 64,
        'emb_dim': 128,
        'hidden_dim': 128,
        'num_layers': 1,
        'bidir': False,
        'lr': 0.001,
        'num_epochs': 3,
        'threshold': 0.5
    }
    mlflow.log_params(p)

    # Load and combine label-encoded datasets
    data_splits = ['train', 'valid', 'test']
    func = torch.from_numpy
    train_ds = TermsDataset(os.path.join(data_path, data_splits[0] + '.csv'), transform=func, target_transform=func)
    valid_ds = TermsDataset(os.path.join(data_path, data_splits[1] + '.csv'), transform=func, target_transform=func)
    test_ds = TermsDataset(os.path.join(data_path, data_splits[2] + '.csv'), transform=func, target_transform=func)

    # batch_size = 64
    train_dl = DataLoader(train_ds, batch_size=p['batch_size'], shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=p['batch_size'], shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=p['batch_size'], shuffle=False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # emb_dim = 128
    # hidden_dim = 128
    # num_layers = 1
    # bidir = False
    model = term_topic_LSTM(
        vocab_size=len(word_to_idx) + 1,
        num_classes=len(class_to_idx),
        #     act=nn.GELU,
        act=F.relu,
        emb_dim=p['emb_dim'], hidden_dim=p['hidden_dim'],
        num_layers=p['num_layers'], bidir=p['bidir']
    ).to(device)

    # lr = 0.001
    # num_epochs = 3
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=p['lr'])
    # model_path = "/dbfs/mnt/output-data/train/models/lstm"
    output_dir = args.train_output
    print(f'Path {output_dir} | Files/Folders: {os.listdir(output_dir)}')
    model_path = os.path.abspath(os.path.join(output_dir, f'bidir={p["bidir"]},num_layers={p["num_layers"]},emb_dim={p["emb_dim"]},hidden_dim={p["hidden_dim"]},lr={p["lr"]},num_epochs={p["num_epochs"]}'))
    print('model_path:', model_path)
    os.makedirs(model_path, exist_ok=True)
    train_loss_list, valid_loss_list, global_steps_list = train(
        model=model, optimizer=optimizer, criterion=criterion, train_loader=train_dl, valid_loader=valid_dl,
        model_path=model_path, device=device, eval_every=len(train_dl) // 2, num_epochs=p['num_epochs'])

    # Plot and evaluate results
    plot_results(train_loss_list, valid_loss_list, global_steps_list)

    val_loss = load_checkpoint(os.path.join(model_path, 'model.pt'), model, optimizer, device)
    print('LSTM model best validation loss:', val_loss)
    mlflow.log_metric('LSTM model best validation loss:', val_loss)

    # NOTE:
    # threshold > 0.5 => improves precision
    # threshold < 0.5 => improves recall
    cr = evaluate(model=model, class_dict=class_to_idx, test_loader=test_dl, device=device, threshold=p['threshold'])
    print('Class Metrics (json)\n', cr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, dest='input_data', help="input dataset")
    parser.add_argument("--train_output", type=str, dest='train_output', help="train output location")

    args = parser.parse_args()
    run = Run.get_context()

    pd.set_option('display.max_colwidth', 256)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    main(args)

    run.complete()


