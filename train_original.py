import argparse
import datetime
import random
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from train_ukdale import validation, get_model
from utils import softmax, handle_loss_history, data_slider, output_preprocess,draw
from utils import data_iter,cal_time
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from Arguments import params_appliance_UKDALE
from tensorboardX import SummaryWriter
from nilm_metric import get_EA, get_sae, recall_precision_accuracy_f1, get_abs_error



@cal_time
def train(net,lr,criterion,optimizer,train_features,train_labels):
    '''
    :param net:neuralnet model.
    :param lr:learning rate.
    :param criterion:loss function definition.
    :param optimizer:str = 'SGD' or 'Adam' or 'Adagrad'
    :param batch_size:batch size.
    :param train_samples:  for DataLoader.
    :param loss_history:loss generated from last epoch.
    :param beta:robust hyperparameter.
    :param method:str = 'mean' or 'sum' or 'each'.
    :return:net,loss_history.
    '''
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),lr=lr)
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    if optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(),lr=lr)

    loss_history = []
    train_dl = data_iter(X=train_features, Y=train_labels, batch_size=batch_size)
    for i,data in enumerate(train_dl):
        train_features,train_labels = data
        train_features = train_features.to('cuda:0')
        train_labels = train_labels.to('cuda:0')
        prediction = net(train_features)
        loss = criterion(prediction,train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.cpu().data.numpy())
        print(f'{i+1}minibatch train loss: {loss.item()}')
        print(f'mean prediction:{prediction.cpu().data.numpy().mean()}  mean actual:{train_labels.cpu().data.numpy().mean()}')
    del train_dl
    return net,loss_history

if __name__ =='__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='input alpha,beta')

    parser.add_argument('-app', '--app_name', type=str, default=None, help='app_name parameter')
    parser.add_argument('-ds', '--dataset', type=str,  help='dataset parameter')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate parameter')
    args = parser.parse_args()

    app_name = args.app_name
    dataset = args.dataset
    lr = args.lr
    csv_train_path = r'./'+dataset+'(preprocessed)/ZQ-'+app_name+'-train.csv'
    csv_valid_path = r'./'+dataset+'(preprocessed)/ZQ-'+app_name+'-validation.csv'
    train_samples = pd.read_csv(csv_train_path,header=None)
    train_samples = data_slider(train_samples,0).values

    net = get_model()
    criterion = nn.MSELoss(reduction='mean')
    optimizer = 'Adam' 
    epochs = 50
    batch_size = 256

    np.random.seed(111)
    np.random.shuffle(train_samples)
    EAs = [];EPOCHs=[];RECALLs=[];PRECISIONs=[]; F1s=[]; ACCURACIEs=[]; SAEs=[]; MAEs=[]
    max_EA = -1e5
    chunksize = batch_size * 100

    date = datetime.datetime.now()
    date = date.strftime('%Y-%m-%d-%H-%M-%S')
    info = [app_name, epochs, batch_size, lr, date]
    mark = '_'.join(map(str, info))
    params_best_model_path = 'parameters/' + dataset + '/' + mark + '_best_model(original).pt'
    params_last_model_path = 'parameters/' + dataset + '/' + mark + '_last_model(original).pt'

    train_features = torch.tensor(train_samples[:, :-1], dtype=torch.float32, requires_grad=True).reshape(-1, 1, 599)
    train_labels = torch.tensor(train_samples[:, -1], dtype=torch.float32, requires_grad=True).reshape(-1, 1)

    train_features = train_features.clone().detach()
    train_labels = train_labels.clone().detach()
    writer = SummaryWriter('./runs/'+dataset+'/'+mark)
    file = open('results/' + dataset + '_validation(original).txt', 'a')
    file.write(f'model path:{params_best_model_path}   {params_last_model_path}\n')
    file.close()
    for epoch in range(epochs):
        print(f'Training {epoch+1} epoch...')
        tup,duration = train(net,lr=lr,criterion=criterion,optimizer=optimizer,train_features=train_features,train_labels=train_labels)
        net,loss_history = tup
        if epoch % 1 == 0:
            train_chunks = pd.read_csv(csv_valid_path, header=None, chunksize=chunksize)
            predicts, actuals = validation(net, criterion, batch_size, train_chunks)
            predicts, actuals = output_preprocess(predicts, actuals, app_name)
            EA = get_EA(actuals, predicts)
            SAE = get_sae(actuals, predicts, 1)
            MAE, std, min_v, max_v, quartile1, median, quartile2, data = get_abs_error(actuals, predicts)
            recall, precision, f1, accuracy = recall_precision_accuracy_f1(target=actuals, prediction=predicts,
                                                                           threshold=params_appliance_UKDALE[app_name][
                                                                               'on_power_threshold'])
            file = open('results/' + dataset + '_validation(original).txt', 'a')
            file.write(
                f'{datetime.datetime.now()}, duration:{duration}s, lr:{lr}, appliance:{app_name}, batchsize:{batch_size}, epoch:{epoch + 1}, recall:{recall}, precision:{precision}, f1:{f1}, accuracy:{accuracy}, EA:{EA}, SAE:{SAE}, MAE:{MAE}\n')

            lr *= 0.99
            if max_EA <= EA:
                max_EA = EA
                best_epoch = epoch + 1
                torch.save(net.state_dict(), params_best_model_path)
                print(
                    f'epoch{epoch+1}: recall:{recall}, precision:{precision}, f1:{f1}, accuracy:{accuracy}, EA:{EA}, SAE:{SAE}, MAE:{MAE}')
            if epoch == epochs-1:
                file.write(f'max validation EA:{max_EA} best epoch:{best_epoch}\n')
                file.write('\n\n')
            file.close()
            writer.add_scalar('EA', EA, epoch+1)
            writer.add_scalar('loss', np.mean(loss_history), epoch+1)
            writer.add_scalar('MAE', MAE, epoch+1)
            writer.add_scalar('SAE', SAE, epoch+1)

            EAs.append(EA)
            EPOCHs.append(epoch+2)
            RECALLs.append(recall)
            PRECISIONs.append(precision)
            F1s.append(f1)
            ACCURACIEs.append(accuracy)
            SAEs.append(SAE)
            MAEs.append(MAE)
            torch.save(net.state_dict(), params_last_model_path)
    writer.close()
    for i in range(len(EAs)):
        print(f'epoch:{EPOCHs[i]}, recall:{RECALLs[i]}, precision:{PRECISIONs[i]}, f1-score:{F1s[i]}, accuracy:{ACCURACIEs[i]}, EAs:{EAs[i]}, SAE:{SAEs[i]}, MAE:{MAEs[i]}')
    print(f'Max EA:{max_EA}, best_epoch:{best_epoch}')
    end_time = time.time()
    print(f'Training complete!Total time:{(end_time-start_time)//3600} h {(end_time-start_time)%3600//60} m ')
