import argparse
import datetime
import time
import random
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils import softmax, handle_loss_history, data_slider, \
    output_preprocess, draw, cal_time, phi_function,data_iter,get_error_set
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from Arguments import params_appliance_UKDALE
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from nilm_metric import get_EA, get_sae, recall_precision_accuracy_f1, get_abs_error, get_metric


def get_model(padding='same',bias='True',parameters=None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
        checkpoint = torch.load(parameters)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']+1
        beta = checkpoint['beta']
        max_EA = checkpoint['max_EA']
        print("====>loaded checkpoint (epoch{})".format(checkpoint['epoch']))
        net.to(device)
        return net,optimizer,epoch,beta,max_EA
    net.to(device)
    return net


@cal_time
def train(net,criterion,optimizer,batch_size,train_samples,epoch,loss_history=None,beta=None,method=None,first_epoch=True,phi='KL',importance_weights=0,speedUp=1):
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

    if first_epoch:
        if loss_history == 'pretraining':
            loss_history = []
            print('pretraining start...')
        if not loss_history:
            loss_history = []
            print('Training 1 epoch...')
        train_dl = data_iter(X=train_samples[:, :-1],Y=train_samples[:, -1],batch_size=batch_size)
        for i,data in enumerate(train_dl):
            x,y = data
            train_features = torch.tensor(x, dtype=torch.float32, requires_grad=True).reshape(-1, 1, 599).cuda()
            train_labels = torch.tensor(y, dtype=torch.float32, requires_grad=True).reshape(-1, 1).cuda()
            prediction = net(train_features)
            loss = criterion(prediction,train_labels)
            optimizer.zero_grad()
            loss.backward()  
            optimizer.step()
            l = loss.cpu().data.numpy()
            loss_history.append(l)
        return net,loss_history
    else:
        print('Training '+str(epoch)+' epoch...')
        train_features = torch.tensor(train_samples[:, :-1],dtype=torch.float32,requires_grad=True).reshape(-1, 1, 599)
        train_labels =  torch.tensor(train_samples[:, -1],dtype=torch.float32,requires_grad=True).reshape(-1, 1)
        train_features = train_features.clone().detach()
        train_labels = train_labels.clone().detach()
        loss_history = handle_loss_history(loss_history,method)
        cal_q = phi_function(phi)
        for j in range(len(loss_history)):
            if j % speedUp == 0:
                q_batch = cal_q(loss_history,beta)

            if method == 'each':
                q_batch_sum = []
                q_batch_len = len(q_batch)
                for start in range(0,q_batch_len,batch_size):
                    end = (start + batch_size) if start + batch_size < q_batch_len else q_batch_len
                    q_batch_sum.append(q_batch[start:end].sum())
                batch_index = np.random.choice(list(range(len(q_batch_sum))), size=1, replace=True, p=q_batch_sum)[0]
            else:
                batch_index = np.random.choice(list(range(len(q_batch))), size=1, replace=True, p=q_batch)[0]

            samples_len = len(train_samples)
            start = batch_index * batch_size
            end = (start + batch_size) if start + batch_size < samples_len else samples_len

            x = train_features[start:end].cuda()
            y = train_labels[start:end].cuda()
            prediction = net(x)
            loss = criterion(prediction, y)
            optimizer.zero_grad()
            if importance_weights:
                if method == 'mean' or 'sum':
                    importance_weights = np.exp(beta * (loss_history[batch_index]-loss.cpu().data.numpy()))
                if method == 'each':
                    importance_weights = np.exp(beta * (loss_history[start:end]-loss.cpu().data.numpy()))
            importance_weights = np.clip(importance_weights,0.1,10)
            loss *= torch.tensor(importance_weights)
            importance_weights = 1
            loss.backward()
            optimizer.step()

            if method == 'mean' :
                loss_history[batch_index] = loss.cpu().data.numpy()
            elif method == 'sum' :
                loss_history[batch_index] = torch.nn.MSELoss(reduction='sum')(prediction,y).cpu().data.numpy()
            elif method == 'each' :
                start = batch_index * batch_size
                end = (start + batch_size) if start + batch_size < samples_len else samples_len
                loss_history[start:end] = torch.nn.MSELoss(reduction='none')(prediction,y).cpu().data.numpy()
        return net,loss_history

def validation(net,criterion,batch_size,train_chunks):
    print('validating...')
    actuals = []
    predicts = []
    for i,train_chunk in enumerate(train_chunks):
        train_chunk = np.array(train_chunk)
        train_features = torch.tensor(train_chunk[:, :-1], dtype=torch.float32, requires_grad=False).reshape(-1, 1, 599).cuda()
        train_labels = torch.tensor(train_chunk[:, -1], dtype=torch.float32, requires_grad=False).reshape(-1, 1).cuda()
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

def validation_group(net,criterion,batch_size,csv_valid_path,group):
    print('validating...')
    train_chunks = pd.read_csv(csv_valid_path)
    data = pd.concat([train_chunks.iloc[(start-300):(end + 1 - 300), :] for start, end in group], ignore_index=True)
    del train_chunks
    actuals = []
    predicts = []
    data = np.array(data)
    train_features = torch.tensor(data[:, :-1], dtype=torch.float32, requires_grad=False).reshape(-1, 1, 599).cuda()
    train_labels = torch.tensor(data[:, -1], dtype=torch.float32, requires_grad=False).reshape(-1, 1).cuda()
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
            print(f'{start/batch_size+1}minibatch validation loss: {batch_loss}')
            print(f'mean prediction:{predictions.cpu().data.numpy().mean()}  mean actual:{y.cpu().data.numpy().mean()}')

    return predicts,actuals
@cal_time
def train_SGD(net,criterion,optimizer,batch_size,train_samples,epoch):
    print('Training ' + str(epoch) + ' epoch...')
    train_dl = data_iter(X=train_samples[:, :-1], Y=train_samples[:, -1], batch_size=batch_size)
    loss_history = []
    for i, data in enumerate(train_dl):
        x, y = data
        train_features = torch.tensor(x, dtype=torch.float32, requires_grad=True).reshape(-1, 1, 599).cuda()
        train_labels = torch.tensor(y, dtype=torch.float32, requires_grad=True).reshape(-1, 1).cuda()
        prediction = net(train_features)
        loss = criterion(prediction, train_labels)
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()
        l = loss.cpu().data.numpy()
        loss_history.append(l)
    return net, loss_history

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='input alpha,beta')
    parser.add_argument('-a', '--alpha', type=float, default=-1, help='alpha parameter')
    parser.add_argument('-b', '--beta', type=float, default=None, help='beta parameter')
    parser.add_argument('-app', '--app_name', type=str, default=None, help='app_name parameter')
    parser.add_argument('-su', '--speedUp', type=int, default=None, help='speedUp parameter')
    parser.add_argument('--lr',  type=float, default=0.0002, help='learning rate parameter')
    parser.add_argument('--phi',  type=str, default='modified_x2_distance', help='phi parameter')
    args = parser.parse_args()
    alpha = args.alpha
    beta = args.beta
    app_name = args.app_name
    speedUp = args.speedUp
    lr = args.lr
    phi = args.phi
    dataset = 'UKDALE'
    csv_train_path = r'/home/mengyuan/LinuxProject/'+dataset+'(preprocessed)/ZQ-'+app_name+'-train.csv'
    csv_valid_path = r'/home/mengyuan/LinuxProject/'+dataset+'(preprocessed)/ZQ-'+app_name+'-validation.csv'


    train_samples = pd.read_csv(csv_train_path,header=None)
    train_samples = data_slider(train_samples,0).values

    net = get_model()
    criterion = nn.MSELoss(reduction='mean')
    EAs = [];EPOCHs=[];RECALLs=[];PRECISIONs=[]; F1s=[]; ACCURACIEs=[]; SAEs=[]; MAEs=[]
    method = 'mean'
    if method == 'mean':
        criterion = nn.MSELoss(reduction='mean')
    if method == 'sum':
        criterion = nn.MSELoss(reduction='sum')
    if method == 'each':
        criterion = nn.MSELoss(reduction='none')
    epochs = 50
    batch_size = 256
    max_EA = -1e5
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)  #对下降路径优化 SGDM    mini-batch GD == SGD
    continue_train = False
    shuffle_inteval = 0
    importance_weights = 1

    date = datetime.datetime.now()
    date = date.strftime('%Y-%m-%d-%H-%M-%S')
    info = [app_name,epochs,batch_size,lr,alpha if alpha>=0 else beta,shuffle_inteval,phi,speedUp,date]
    mark = '_'.join(map(str,info))
    loss_history_path = './loss_histories/' +  dataset + '/' + mark + '.npy'
    params_best_model_path = './parameters/' + dataset + '/' + mark + '_best_model.pt'
    params_last_model_path = './parameters/' + dataset + '/' + mark + '_last_model.pt'
    params_path = params_best_model_path
    np.random.shuffle(train_samples)
    chunksize = batch_size * 100

    writer = SummaryWriter('./runs/UKDALE/'+mark)

    if not continue_train:
        tup, duration = train(net, criterion=criterion, optimizer=optimizer, batch_size=batch_size,
                              train_samples=train_samples, loss_history='pretraining',
                              beta=None, method=method, first_epoch=True, epoch=1,importance_weights=importance_weights,speedUp=speedUp)
        net, loss_history = tup
        del loss_history
        file = open('./results/' + dataset + '_validation.txt', 'a')
        file.write(f'model path:{params_best_model_path}   {params_last_model_path}\n')
        file.write(
            f'{datetime.datetime.now()},pretraining duration:{duration}s, appliance:{app_name}, batchsize:{batch_size}\n')
        file.close()


        tup,duration = train(net,criterion=criterion,optimizer=optimizer,batch_size=batch_size,
              train_samples=train_samples,loss_history=None,
              beta=None,method=method,first_epoch=True,epoch=1,importance_weights=importance_weights,speedUp=speedUp)
        net, loss_history = tup
        pre_loss_history = loss_history

        train_chunks = pd.read_csv(csv_valid_path, header=None, chunksize=chunksize, engine='python')
        predicts, actuals = validation(net, criterion, batch_size, train_chunks)
        del train_chunks
        predicts, actuals = output_preprocess(predicts, actuals, app_name)
        recall, precision, f1, accuracy, MAE, SAE, EA = get_metric(actuals,predicts,app_name)
        file = open('./results/'+dataset+'_validation.txt', 'a')
        file.write(
            f'{datetime.datetime.now()},duration:{duration}s, appliance:{app_name}, lr:{lr}, batchsize:{batch_size}, epoch:1/{epochs}, EA:{EA}, SAE:{SAE}, MAE:{MAE}， accuracy:{accuracy}, precision:{precision}, f1:{f1}, recall:{recall}\n')
        file.close()

        loss_history = handle_loss_history(loss_history,method='mean')

        if alpha != -1:
            beta = alpha / handle_loss_history(loss_history, 'mean').mean()

        writer.add_scalar('EA',EA,1)
        writer.add_scalar('beta',beta,1)
        writer.add_scalar('loss',np.mean(loss_history),1)
        writer.add_scalar('MAE',MAE,1)
        writer.add_scalar('SAE',SAE,1)

        epochs = epochs - 1
        epoch = 0
    else:
        loss_history = np.load(loss_history_path)
        net,optimizer,epoch,beta,max_EA = get_model(parameters=params_path)
        print('epoch:{},beta:{},max_EA:{}'.format(epoch,beta,max_EA))
    for epoch in range(epoch,epochs):
        if shuffle_inteval:
            if epoch % shuffle_inteval == 0:
                np.random.shuffle(train_samples)
                tup, duration = train_SGD(net,criterion,optimizer,batch_size,train_samples,epoch+2)
                net, loss_history = tup

        else:
            tup, duration = train(net, criterion=criterion, optimizer=optimizer, batch_size=batch_size,
                                  train_samples=train_samples, loss_history=loss_history,
                                  beta=beta, method=method, first_epoch=False, epoch=epoch + 2, phi=phi, importance_weights=importance_weights,speedUp=speedUp)
            net, loss_history = tup
       
        if alpha != -1:
            beta = alpha/ handle_loss_history(loss_history, 'mean').mean()
        optimizer = torch.optim.Adam(net.parameters(),lr=lr)
        if epoch % 1 == 0:
            train_chunks = pd.read_csv(csv_valid_path, header=None, chunksize=chunksize)
            predicts, actuals = validation(net, criterion, batch_size, train_chunks)
            predicts, actuals = output_preprocess(predicts, actuals, app_name)
            recall, precision, f1, accuracy, MAE, SAE, EA = get_metric(actuals,predicts,app_name)
            file = open('./results/'+dataset+'_validation.txt', 'a')
            file.write(
                f'{datetime.datetime.now()},duration:{duration}s, appliance:{app_name}, lr:{lr}, batchsize:{batch_size}, epoch:{epoch + 2}/{epochs+1},  EA:{EA}, SAE:{SAE}, MAE:{MAE}， accuracy:{accuracy}, precision:{precision}, f1:{f1}, recall:{recall}\n')
            print(f'{datetime.datetime.now()},duration:{duration}s, appliance:{app_name}, lr:{lr}, batchsize:{batch_size}, epoch:{epoch + 2}/{epochs+1}, recall:{recall}, precision:{precision}, f1:{f1}, accuracy:{accuracy}, EA:{EA}, SAE:{SAE}, MAE:{MAE}\n')

            lr *= 0.99
            if max_EA <= EA:
                max_EA = EA
                best_epoch = epoch+2
                checkpoint = {'model_state_dict':net.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'epoch':epoch+2,'beta':beta,'max_EA':max_EA}
                torch.save(checkpoint, params_best_model_path)
                np.save(loss_history_path, loss_history)
                
            if epoch == epochs-1:
                end_time = time.time()
                file.write(f'max validation EA:{max_EA} best epoch:{best_epoch} Total time:{(end_time-start_time)//3600} h {(end_time-start_time)%3600//60} m \n')
                file.write('\n\n')
            file.close()
            writer.add_scalar('EA', EA, epoch + 2)
            writer.add_scalar('beta', beta, epoch + 2)
            writer.add_scalar('loss', np.mean(loss_history), epoch + 2)
            writer.add_scalar('MAE', MAE, epoch + 2)
            writer.add_scalar('SAE', SAE, epoch + 2)

            EAs.append(EA)
            EPOCHs.append(epoch+2)
            RECALLs.append(recall)
            PRECISIONs.append(precision)
            F1s.append(f1)
            ACCURACIEs.append(accuracy)
            SAEs.append(SAE)
            MAEs.append(MAE)
            torch.save(net.state_dict(),params_last_model_path)
    print()
    writer.close()
    for i in range(len(EAs)):
        print(f'epoch:{EPOCHs[i]}, recall:{RECALLs[i]}, precision:{PRECISIONs[i]}, f1-score:{F1s[i]}, accuracy:{ACCURACIEs[i]}, EA:{EAs[i]}, SAE:{SAEs[i]}, MAE:{MAEs[i]}')
    print(f'Max EA:{max_EA}, best_epoch:{best_epoch}')
    print(f'Training complete!Total time:{(end_time-start_time)//3600} h {(end_time-start_time)%3600//60} m ')
