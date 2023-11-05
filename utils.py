import itertools
import os
import time
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Arguments import params_appliance_UKDALE, params_appliance_REDD
import csv
from Phi import Phi
from group_annotation import test_group

# Generate 599 feature 1 label data
def save_datas(data_path,save_path):
    samples = []
    features = []
    lable_curve = open(data_path,'r')
    with open(data_path,'r') as file:
        length = len(file.readlines())
        file.seek(0)
        for index in range(299):
            lable_curve.readline()
        for index in range(599):
            line = file.readline()
            record = line.split(',')
            features.append(record[0])
            if(index==299):
                lable = record[1]
        features.append(float(lable))
        samples.append(features)

        for start in range(length-599+1):
            original = [i for i in samples[start]]
            del original[0]
            del original[-1]
            line = file.readline()
            record = line.split(',')
            original.append(record[0])
            line = lable_curve.readline()
            record = line.split(',')
            original.append(float(record[1]))
            samples.append(original)

    with open(save_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(samples)

# squared_loss
def squared_loss(y_hat,y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

# sgd
def sgd(parameters_,lr,batch_size):
    with torch.no_grad():
        for param in parameters_:
            param -= lr * param.grad / batch_size
            # grad initiate zeros
            param.grad.zero_()

# initiate weights
def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight,mean=0,std=0.1)

def softmax(x,beta):
    '''
    φkl(t) =  t log t − t + 1

    '''
    x = x*beta
    exp_x = np.exp(x)
    exp_x = np.where(np.isinf(exp_x),np.exp(700),exp_x)
    sum_exp_x = np.sum(exp_x)
    p = exp_x / sum_exp_x
    eps = 0.0001 / len(x)
    p[p <= eps] = eps
    p /= p.sum()
    return p

def modified_x2_distance(x,beta):

    N = len(x)
    sum_L = sum(x)
    betaNli = [N*beta*li for li in x]
    p = [(betanli+2*N-sum_L*beta) / (2*N*N) for betanli in betaNli]
    p = np.array(p)
    eps = 0.0001 / N
    p[p <= eps] = eps
    p /= p.sum()
    #ret = np.exp(ret)
    return p



def data_slider(dataset,stride):
    df = pd.DataFrame(dataset)
    df = df.fillna(method='ffill', axis=1)  
    if stride==0:
        return df
    a = []
    for i in range(0,len(df),stride):  
        a.append(i)
    samples = df.iloc[a]
    return samples



def handle_loss_history(loss_history,method):
    if method == 'mean' or method == 'sum':
        loss_history = np.array(loss_history,dtype=np.float64).flatten()
    if method == 'each':
        loss_history = np.array([j for i in loss_history for j in i],dtype=np.float64)
    return loss_history

def output_preprocess(predicts,actuals,appliance):
    mean = params_appliance_UKDALE[appliance]['mean']
    std = params_appliance_UKDALE[appliance]['std']
    predicts = np.concatenate(predicts).reshape(-1,)
    actuals = np.concatenate(actuals).reshape(-1,)
    predicts = predicts * std + mean
    actuals = actuals * std + mean
    return predicts,actuals

def get_total_power(appliance,mode,dataset,group=False):
    samples = pd.read_csv(r'/home/mengyuan/LinuxProject/'+dataset+'/'+appliance+'/'+appliance+'_'+mode+'_.csv',skiprows=297)

    if group:
        group = test_group[appliance][group]
        total_power = pd.concat([samples.iloc[(start-300):(end + 1 -300),0] for start, end in group],
                         ignore_index=True).values * 814 + 522
    else:
        samples = samples.values
        total_power = samples[:-299, 0] * 814 + 522
    return total_power

def draw(predicts,actuals,appliance,dataset,mode,epoch=None,group=False):
    '''
    :param mode: 'test' or 'validation'
    '''
    total_power = get_total_power(appliance,mode,dataset,group)
    on_power_threshold = params_appliance_UKDALE[appliance]['on_power_threshold']
    predicts = np.where(predicts < on_power_threshold, 0, predicts)
    plt.axhline(y = on_power_threshold,color='green',linestyle='-.', label='Power threshold')
    plt.fill_between(list(range(len(total_power))),total_power,color='gray',alpha=0.3)
    plt.plot(list(range(len(total_power))), total_power, color='0.5', alpha=0.3, label='Aggregate')
    plt.plot(list(range(len(actuals))), actuals, '0.2', alpha=1, label='Actuals')
    plt.plot(list(range(len(predicts))), predicts, 'r-', alpha=1, label='Predicts')


    title = 'Validation'+str(epoch) if mode=='validation' else 'Test'
    title += '('+appliance+')'
    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('Power Consumption(Watt)')
    plt.legend()
    plt.show()

def data_iter(X,Y,batch_size):
    num_examples = len(X)
    indices = list(range(num_examples))
    for i in range(0,num_examples,batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i+batch_size,num_examples)]
        )
        yield X[batch_indices],Y[batch_indices]


def iter_count(file_name):
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)

def cal_time(func):
    def wrapper(*args,**kwargs):
        t1 = time.time()
        tup = func(*args,**kwargs)
        t2 = time.time()
        duration = int(t2-t1)
        print('Duration:{} s'.format(duration))
        return tup,duration
    return wrapper

def phi_function(phi):
    phi_obj = Phi()
    if phi == 'Kullback-Leibler':
        return softmax
    elif phi == 'Burg_entropy':
        return phi_obj.Burg_entropy
    elif phi == 'J_divergence':
        return phi_obj.J_divergence
    elif phi == 'x2_distance':
        return phi_obj.x2_distance
    elif phi == 'modified_x2_distance':
        return modified_x2_distance
    elif phi == 'Hellinger_distance':
        return phi_obj.Hellinger_distance

def filter_by_distance(model_path,distance):
    batch_size = 1024
    chunksize = batch_size * 100
    appliance = ['fridge', 'dishwasher', 'washingmachine', 'kettle', 'microwave']
    app_name = [name for name in appliance if name in model_path][0]
    dataset = 'REDD' if 'REDD' in model_path else 'UKDALE'
    criterion = torch.nn.MSELoss(reduction='mean')
    net,epoch = get_model_proposed(parameters=model_path)
    mode = 'train'
    csv_train_path = r'G:/' + dataset + '(preprocessed)/ZQ-' + app_name + '-' + mode + '.csv'
    train_chunks = pd.read_csv(csv_train_path, header=None, chunksize=chunksize)
    predicts, actuals = test(net,criterion,batch_size,train_chunks)
    df = pd.DataFrame({'predicts': predicts, 'actuals': actuals})
    diff = df['actuals'] - df['predicts']
    idx = np.where(diff >= distance)[0]
    idx -= 300
    return idx

def filter_by_split_threshold(csv_path,split_threshold):
    appliance = ['fridge', 'dishwasher', 'washingmachine', 'kettle', 'microwave']
    app_name = [name for name in appliance if name in csv_path][0]
    df = pd.read_csv(csv_path,header=None)
    total_power = df.iloc[:,0]
    actuals = df.iloc[:,-1]
    mean = params_appliance_UKDALE[app_name]['mean']
    std = params_appliance_UKDALE[app_name]['std']
    diff = actuals * std + mean - split_threshold
    idx = np.where(diff > 0)[0]
    return idx

def get_error_set(model_or_csv_path,train_samples,split_threshold,times=1,distance=0):
    if distance:
        index = filter_by_distance(model_or_csv_path,distance)
    if split_threshold:
        index = filter_by_split_threshold(model_or_csv_path, split_threshold)
    error_samples = train_samples.iloc[index]
    expanded_error_samples = pd.concat([error_samples] * times, ignore_index=True)
    return expanded_error_samples
