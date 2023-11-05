import argparse
import datetime
import pdb
import numpy as np
from torch import nn
import os
from Arguments import params_appliance_UKDALE
from nilm_metric import get_sae, recall_precision_accuracy_f1, get_abs_error, get_EA
import torch
import pandas as pd
from utils import output_preprocess, draw, get_error_set, filter_by_split_threshold
from group_annotation import *

def get_model_proposed(padding='same',bias='True',parameters=None):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    torch.device(device)
    net = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=30, kernel_size=10, stride=1, padding=padding, bias=bias),
        nn.ReLU(True),
        nn.Conv1d(in_channels=30, out_channels=30, kernel_size=8, stride=1, padding=padding, bias=bias),
        nn.ReLU(True),
        nn.Conv1d(in_channels=30, out_channels=40, kernel_size=6, stride=1, padding=padding, bias=bias),
        nn.ReLU(True),
        nn.Conv1d(in_channels=40, out_channels=50, kernel_size=5, stride=1, padding=padding, bias=bias),
        nn.ReLU(True),
        nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5, stride=1, padding=padding, bias=bias),
        nn.ReLU(True),
        # nn.Dropout(0.3),
        nn.Flatten(),
        nn.Linear(599 * 50, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 1)
    )
    if parameters:
        print("Resume from checkpoint...")
        checkpoint = torch.load(parameters)
        net.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        print("====>loaded checkpoint (epoch{})".format(checkpoint['epoch']))
        net.to(device)
        return net,epoch
    net.to(device)
    return net

def get_model_original(padding='same',bias='True',parameters=None):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    torch.device(device)
    net = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=30, kernel_size=10, stride=1, padding=padding, bias=bias),
        nn.ReLU(True),
        nn.Conv1d(in_channels=30, out_channels=30, kernel_size=8, stride=1, padding=padding, bias=bias),
        nn.ReLU(True),
        nn.Conv1d(in_channels=30, out_channels=40, kernel_size=6, stride=1, padding=padding, bias=bias),
        nn.ReLU(True),
        nn.Conv1d(in_channels=40, out_channels=50, kernel_size=5, stride=1, padding=padding, bias=bias),
        nn.ReLU(True),
        nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5, stride=1, padding=padding, bias=bias),
        nn.ReLU(True),
        nn.Flatten(),
        nn.Linear(599 * 50, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 1)
    )
    if parameters:
        print("Resume from checkpoint...")
        params = torch.load(parameters)
        net.load_state_dict(params)
        net.to(device)
        return net
    net.to(device)
    return net

def test(net,criterion,batch_size,train_chunks):
    actuals = []
    predicts = []
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    for i,train_chunk in enumerate(train_chunks):
        train_chunk = train_chunk.values
        train_chunk = train_chunk.astype(float)
        train_features = torch.tensor(train_chunk[:, :-1], dtype=torch.float32, requires_grad=False).reshape(-1, 1, 599).to(device)
        train_labels = torch.tensor(train_chunk[:, -1], dtype=torch.float32, requires_grad=False).reshape(-1, 1).to(device)
        with torch.no_grad():
            samples_len = len(train_features)
            samples_len -= samples_len % batch_size
            for start in range(0, samples_len, batch_size):
                end = start + batch_size
                x = train_features[start:end]
                y = train_labels[start:end]
                predictions = net(x)
                loss = criterion(predictions, y)
                batch_loss = loss.item()
                predicts.append(predictions.cpu().data.numpy())
                actuals.append(y.cpu().data.numpy())
    return predicts,actuals


def test_on_group(net,criterion,batch_size,csv_test_path,group):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    actuals = []
    predicts = []
    data = np.array(group)
    train_features = torch.tensor(data[:, :-1], dtype=torch.float32, requires_grad=False).reshape(-1, 1, 599).to(device)
    train_labels = torch.tensor(data[:, -1], dtype=torch.float32, requires_grad=False).reshape(-1, 1).to(device)
    with torch.no_grad():
        samples_len = len(train_features)
        for start in range(0, samples_len, batch_size):
            end = start + batch_size
            if samples_len - start < batch_size:
                end = samples_len
            x = train_features[start:end]
            y = train_labels[start:end]
            predictions = net(x)
            loss = criterion(predictions, y)
            batch_loss = loss.item()
            predicts.append(predictions.cpu().data.numpy())
            actuals.append(y.cpu().data.numpy())

    return predicts,actuals

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='input test_model_path')
    parser.add_argument('-t', '--tmp', type=str, help='Description of test_model_path parameter')
    parser.add_argument('--group',  type=bool, default=False, help='test in group')
    parser.add_argument('--ds',  type=str,  help='dataset ')
    args = parser.parse_args()

    test_model_path = args.tmp
    group = args.group
    dataset = args.ds
    batch_size = 1024
    chunksize = batch_size * 100
    appliance = ['fridge','dishwasher','washingmachine','kettle','microwave','laptop','toaster']
    app_name = [name for name in appliance if name in test_model_path][0]
    criterion = torch.nn.MSELoss(reduction='mean')
    mode = 'test'
    csv_test_path = r'/home/mengyuan/LinuxProject/'+dataset+'(preprocessed)/ZQ-'+app_name+'-'+mode+'.csv'
    if 'train' in csv_test_path:
        mode = 'training'
    dtypes = {i: float for i in range(600)}

    if 'original' in test_model_path:
        if 'best' in test_model_path:
            net = get_model_original(parameters = test_model_path)
            epoch = 0
        if 'last' in test_model_path:
            net = get_model_original(parameters = test_model_path)
            epoch = test_model_path.split('_')[1]
    else :
        if 'best' in test_model_path:
            net,epoch = get_model_proposed(parameters = test_model_path)
        if 'last' in test_model_path:
            net = get_model_original(parameters=test_model_path)
            epoch = test_model_path.split('_')[1]

    if group:
        train_chunks = pd.read_csv(csv_test_path, header=None)
        index = filter_by_split_threshold(csv_test_path,params_appliance_UKDALE[app_name]['on_power_threshold'])
        group = train_chunks.iloc[index]
        predicts, actuals = test_on_group(net, criterion, batch_size, csv_test_path, group)

    else:
        group = ''
        train_chunks = pd.read_csv(csv_test_path, header=None, chunksize=chunksize, dtype=dtypes)
        predicts, actuals = test(net, criterion, batch_size, train_chunks)
    predicts, actuals = output_preprocess(predicts, actuals, app_name)
    nan_indices = np.where(np.isnan(predicts))[0]

    for i in nan_indices:
        predicts[i] = predicts[i - 1]
    SAE = get_sae(actuals, predicts, 1)
    EA = get_EA(actuals, predicts)
    MAE, std, min_v, max_v, quartile1, median, quartile2, data = get_abs_error(actuals, predicts)
    on_power_threshold = params_appliance_UKDALE[app_name]['on_power_threshold']
    recall, precision, f1, accuracy = recall_precision_accuracy_f1(target=actuals, prediction=predicts,
                                                                   threshold=on_power_threshold)

    df = pd.DataFrame({'actuals': actuals, 'predicts': predicts})
    file_name_only = os.path.basename(test_model_path)
    file_parts = file_name_only.split("_best_model.pt")
    result = './results_test/forOriginPlot/' + mode + '/' + file_parts[0] + "_best_model.csv"
    if not os.path.exists(result):
        df.to_csv(result, index=False)
    file = open('./results_test/test'+'.txt', 'a')
    file.write(
        f'{datetime.datetime.now()}, appliance:{app_name}, this_epoch:{epoch}, EA:{EA}, SAE:{SAE}, MAE:{MAE}, accuracy:{accuracy}, precision:{precision}, f1:{f1}, recall:{recall}, on_power_threshold:{on_power_threshold}\n')
    file.close()
    print(test_model_path)
    print(
        f'best_epoch:{epoch}, EA:{EA}, SAE:{SAE}, MAE:{MAE}, accuracy:{accuracy}, precision:{precision}, f1:{f1}, recall:{recall} \n')
