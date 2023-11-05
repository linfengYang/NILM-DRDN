import colorsys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from basic_units import cm, inch
from mpl_toolkits.mplot3d import Axes3D
from Arguments import params_appliance_UKDALE
from utils import get_total_power
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import seaborn as sns
from scipy import interpolate

def agg_actual(csv_path):
    df = pd.read_csv(csv_path)
    appliance = ['fridge','dishwasher','washingmachine','kettle','microwave']
    app_name = [name for name in appliance if name in csv_path][0]
    total_power = df.iloc[:,0]
    actuals = df.iloc[:,1]
    mean = params_appliance_UKDALE[app_name]['mean']
    std = params_appliance_UKDALE[app_name]['std']
    total_power = total_power * 814 + 522
    actuals = actuals * std + mean
    plt.plot(list(range(len(total_power))), total_power, color='0.5', alpha=0.3, label='Aggregate')
    plt.fill_between(list(range(len(total_power))),total_power,color = 'gray',alpha=0.3)
    plt.plot(list(range(len(actuals))), actuals, '0.2', alpha=1,label='Actuals')
    title = csv_path.split('_')[-2]+' set'
    title += '(' + app_name + ')'
    plt.title(title)
    plt.legend()
    plt.xlabel('samples')
    plt.ylabel('Power Consumption(Watt)')
    plt.show()

def draw(csv_path):
    '''
    Individual appliance prediction results
    csv_path：actuals,predicts
    '''
    appliance = ['fridge','dishwasher','washingmachine','kettle','microwave']
    app_name = [name for name in appliance if name in csv_path][0]
    dataset = 'REDD' if 'REDD' in csv_path else 'UKDALE'
    df = pd.read_csv(csv_path)
    results = df.iloc[1:,:2]
    actuals = results.iloc[:,0]
    predicts = results.iloc[:,1]

    total_power = get_total_power(app_name, 'test', dataset)
    on_power_threshold = params_appliance_UKDALE[app_name]['on_power_threshold']
    predicts = np.where(predicts < on_power_threshold, 0, predicts)

    plt.axhline(y=on_power_threshold, color='green', linestyle='-.', label='Power threshold')
    plt.plot(list(range(len(total_power))), total_power, color='0.5', alpha=0.3, label='Aggregate')
    plt.fill_between(list(range(len(total_power))),total_power,color = 'gray',alpha=0.3)
    plt.plot(list(range(len(actuals))), actuals, '0.2', alpha=1,label='Actuals')
    plt.plot(list(range(len(predicts))), predicts, 'r', alpha=1,label='Predicts')

    title = 'Test'
    title += '(' + app_name + ')'
    plt.title(title)
    plt.legend()
    plt.xlabel('samples')
    plt.ylabel('Power Consumption(Watt)')
    plt.show()

def plot_subplots(csv_paths,titles,start=0,end=None):
    num_plots = len(csv_paths)
    num_rows = int(np.ceil(np.sqrt(num_plots)))
    num_cols = int(np.ceil(num_plots / num_rows))

    appliance = ['fridge','dishwasher','washingmachine','kettle','microwave']
    app_name = [name for name in appliance if name in csv_paths[0]][0]
    dataset = 'REDD' if 'REDD' in csv_paths[0] else 'UKDALE'
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 10))
    total_power = get_total_power(app_name, 'test', dataset)
    on_power_threshold = params_appliance_UKDALE[app_name]['on_power_threshold']
    total_power = total_power[start:end]
    for i, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        df = df.iloc[1:, :2]
        y1 = df.iloc[start:end, 0]
        y2 = df.iloc[start:end, 1]
        y2 = np.where(y2 < on_power_threshold, 0, y2)
        x = np.arange(start, end if end is not None else len(y1))

        row = i // num_cols
        col = i % num_cols

        if num_plots == 1:
            ax = axes
        elif num_rows == 1:
            ax = axes[col]
        elif num_cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]

        ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')  

        ax.axhline(y=on_power_threshold, color='green', linestyle='-.', label='Power threshold')
        ax.plot(x, total_power, color='0.5', alpha=0.3, label='Aggregate')
        ax.fill_between(x, total_power, color='gray', alpha=0.3)
        ax.plot(x, y1, color='0.2',  alpha=1, label='Actuals')
        ax.plot(x, y2, color='r',    alpha=1, label='Predicts')
        ax.tick_params(axis='x', labelsize=12)
        ax.set_xlabel('Samples')
        ax.set_ylabel('Power Consumption(Watt)')
        ax.set_title(dataset+' '+app_name+' '+titles[i])
        ax.legend()

    plt.tight_layout()
    plt.show()

def subplot_REDD(csv_path,csv_ranges):
    "csv_path :(s2p,ours)"
    appliance = ['fridge','dishwasher','washingmachine','kettle','microwave']
    dataset = 'REDD' if 'REDD' in csv_paths[0] else 'UKDALE'
    num_plots = len(csv_path)
    num_cols = 2
    num_rows = num_plots // num_cols + (1 if num_plots % num_cols != 0 else 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    for i, ((path1, path2), (start, end)) in enumerate(zip(csv_path, csv_ranges)):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        app_name = [name for name in appliance if name in path1][0]
        total_power = get_total_power(app_name, 'test', dataset)
        on_power_threshold = params_appliance_UKDALE[app_name]['on_power_threshold']

        df1 = pd.read_csv(path1, header=None, skiprows=start, nrows=end-start)
        df2 = pd.read_csv(path2, header=None, skiprows=start, nrows=end-start)
        tp = total_power[start:end]
        x = np.arange(0, end-start)

        ax.axhline(y=on_power_threshold, color='green', linestyle='-.', label='Power threshold')
        ax.plot(x, tp, color='0.5', alpha=0.3, label='Mains')
        ax.fill_between(x, tp, color='gray', alpha=0.3)
        ax.plot(x, df1.iloc[:,0], color='0.2',  alpha=1,label='Ground truth')
        ax.plot(x, np.where(df1.iloc[:,1] < on_power_threshold, 0, df1.iloc[:,1]), color='#3a86ff',    alpha=1,label='Baseline(S2P)')
        ax.plot(x,np.where(df2.iloc[:,1] < on_power_threshold, 0, df2.iloc[:,1]), color='#fb5607',    alpha=1, label='Ours')
        if col == 0:
            ax.set_ylabel('Power consumption(Watt)', color='black', fontname='Times New Roman', fontsize=12)
        ax.set_xlabel('Samples', color='black', fontname='Times New Roman', fontsize=16)
        ax.set_title(app_name.capitalize(), color='black', fontname='Times New Roman', fontsize=16,fontweight='bold')
        ax.legend(loc='upper right')

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.tight_layout()
    plt.show()
def subplot_UK(csv_path,csv_ranges):
    "csv_path :(s2p,ours)"
    appliance = ['fridge','dishwasher','washingmachine','kettle','microwave']
    #app_name = [name for name in appliance if name in csv_paths[0]][0]
    dataset = 'REDD' if 'REDD' in csv_paths[0] else 'UKDALE'

    num_plots = len(csv_path)
    num_cols = 2
    num_rows = num_plots // num_cols + (1 if num_plots % num_cols != 0 else 0)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    axes[2, 1].axis('off')
    for i, ((path1, path2), (start, end)) in enumerate(zip(csv_path, csv_ranges)):
        row = i // num_cols
        col = i % num_cols
        if row==2 and col==1:
            break
        ax = axes[row, col]
        app_name = [name for name in appliance if name in path1][0]
        total_power = get_total_power(app_name, 'test', dataset)
        on_power_threshold = params_appliance_UKDALE[app_name]['on_power_threshold']

        df1 = pd.read_csv(path1, header=None, skiprows=start, nrows=end-start)
        df2 = pd.read_csv(path2, header=None, skiprows=start, nrows=end-start)
        tp = total_power[start:end]
        x = np.arange(0, end-start)


        ax.axhline(y=on_power_threshold, color='green', linestyle='-.', label='Power threshold')
        ax.plot(x, tp, color='0.5', alpha=0.3, label='Mains')
        ax.fill_between(x, tp, color='gray', alpha=0.3)
        ax.plot(x, df1.iloc[:,0], color='0.2',  alpha=1,label='Ground truth')
        if app_name == 'fridge':
            ax.plot(x, df1.iloc[:, 1], color='#3a86ff', alpha=1,
                    label='Baseline(S2P)')
            ax.plot(x, df2.iloc[:, 1], color='#fb5607', alpha=1,
                    label='Ours')
        else:
            ax.plot(x, np.where(df1.iloc[:,1] < on_power_threshold, 0, df1.iloc[:,1]), color='#3a86ff',    alpha=1,label='S2P(Zhang)')
            ax.plot(x,np.where(df2.iloc[:,1] < on_power_threshold, 0, df2.iloc[:,1]), color= '#fb5607',    alpha=1, label='Ours')
        if col==0:
            ax.set_ylabel('Power consumption(Watt)', color='black', fontname='Times New Roman', fontsize=12)
        ax.set_xlabel('Samples', color='black', fontname='Times New Roman', fontsize=16)
        ax.set_title(app_name.capitalize(), color='black', fontname='Times New Roman', fontsize=16,fontweight='bold')
        if app_name=='fridge':
            ax.set_ylim(0, 400)
        #ax.text(0.1, 0.1, 'Label', ha='right', va='top')
        ax.legend(loc='upper right')
        if app_name=='dishwasher':
            ax.legend(bbox_to_anchor=(0.5,1))

    #fig.suptitle("Disaggregation results on UKDALE", fontsize=16)
    #fig.text(0.1, 0.05, 'Label', ha='right', va='top')
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.tight_layout()
    plt.show()


def subplot_REDD_by_one(csv_path,csv_ranges):
    appliance = ['fridge','dishwasher','washingmachine','kettle','microwave']
    dataset = 'REDD' if 'REDD' in csv_paths[0] else 'UKDALE'
    for i, ((path1, path2), (start, end)) in enumerate(zip(csv_path, csv_ranges)):
        fig, ax = plt.subplots(figsize=(12, 8),dpi=200)  

        app_name = [name for name in appliance if name in path1][0]
        total_power = get_total_power(app_name, 'test', dataset)
        on_power_threshold = params_appliance_UKDALE[app_name]['on_power_threshold']

        df1 = pd.read_csv(path1, header=None, skiprows=start, nrows=end-start)
        df2 = pd.read_csv(path2, header=None, skiprows=start, nrows=end-start)
        tp = total_power[start:end]
        x = np.arange(0, end-start)

        ax.axhline(y=on_power_threshold, color='green', linestyle='-.', label='Power threshold')
        ax.plot(x, tp, color='0.5', alpha=0.3, label='Mains')
        ax.fill_between(x, tp, color='gray', alpha=0.3)
        ax.plot(x, df1.iloc[:,0], color='0.2',  alpha=1,label='Ground truth')
        ax.plot(x, np.where(df1.iloc[:,1] < on_power_threshold, 0, df1.iloc[:,1]), color='#3a86ff',    alpha=1,label='Baseline(S2P)')
        ax.plot(x,np.where(df2.iloc[:,1] < on_power_threshold, 0, df2.iloc[:,1]), color='#fb5607',    alpha=1, label='Ours')

        ax.set_ylabel('Power consumption(Watt)', color='black', fontname='Times New Roman', fontsize=20)
        ax.set_xlabel('Samples', color='black', fontname='Times New Roman', fontsize=20)
        ax.set_title(app_name.capitalize(), color='black', fontname='Times New Roman', fontsize=20,fontweight='bold')

        ax.tick_params(axis='x', labelsize=18)  
        ax.tick_params(axis='y', labelsize=18)  

        ax.legend(loc='upper right', fontsize=18)

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        plt.tight_layout()
        plt.show()  

def subplot_UKDALE_by_one(csv_path,csv_ranges):
    appliance = ['fridge','dishwasher','washingmachine','kettle','microwave']
    dataset = 'REDD' if 'REDD' in csv_paths[0] else 'UKDALE'
    for i, ((path1, path2), (start, end)) in enumerate(zip(csv_path, csv_ranges)):
        fig, ax = plt.subplots(figsize=(12, 8),dpi=200)  

        app_name = [name for name in appliance if name in path1][0]
        total_power = get_total_power(app_name, 'test', dataset)
        on_power_threshold = params_appliance_UKDALE[app_name]['on_power_threshold']

        df1 = pd.read_csv(path1, header=None, skiprows=start, nrows=end-start)
        df2 = pd.read_csv(path2, header=None, skiprows=start, nrows=end-start)
        tp = total_power[start:end]
        x = np.arange(0, end-start)

        ax.axhline(y=on_power_threshold, color='green', linestyle='-.', label='Power threshold')
        ax.plot(x, tp, color='0.5', alpha=0.3, label='Mains')
        ax.fill_between(x, tp, color='gray', alpha=0.3)
        ax.plot(x, df1.iloc[:,0], color='0.2',  alpha=1,label='Ground truth')
        ax.plot(x, np.where(df1.iloc[:,1] < on_power_threshold, 0, df1.iloc[:,1]), color='#3a86ff',    alpha=1,label='Baseline(S2P)')
        ax.plot(x,np.where(df2.iloc[:,1] < on_power_threshold, 0, df2.iloc[:,1]), color='#fb5607',    alpha=1, label='Ours')

        ax.set_ylabel('Power consumption(Watt)', color='black', fontname='Times New Roman', fontsize=20)
        ax.set_xlabel('Samples', color='black', fontname='Times New Roman', fontsize=20)
        ax.set_title(app_name.capitalize(), color='black', fontname='Times New Roman', fontsize=20,fontweight='bold')

        ax.tick_params(axis='x', labelsize=18)  
        ax.tick_params(axis='y', labelsize=18)  

        ax.legend(loc='upper right', fontsize=18)

        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        plt.tight_layout()
        plt.show()  

def group_bar_chart_UK():
    appliance = ("WM", "KT", "DW", "FG", "MW", "Overall")
    penguin_MAE = {
        'CNN(S2S)': (6.937, 3.171, 3.675, 12.016, 2.114, 5.582),
        'CNN(S2P)': (5.704, 3.708, 3.633, 11.209, 2.200, 5.291),
        'FCN': (4.999, 3.091, 3.626, 9.192, 2.108, 4.603),
        'BitcnNILM': (4.063, 2.160, 3.490, 8.136, 1.305, 3.831),
        'Proposed':(2.734,2.157,2.413,6.138,1.230,2.9344)
    }
    penguin_SAE = {
        'CNN(S2S)': (0.078, 0.006, 0.033, 0.051, 0.186, 0.0708),
        'CNN(S2P)': (0.098, 0.023, 0.021, 0.096, 0.182, 0.084),
        'FCN':      (0.208, 0.036, 0.032, 0.042, 0.195, 0.1026),
        'BitcnNILM': (0.143, 0.045, 0.002, 0.040, 0.111, 0.068),
        'Proposed':  (0.085, 0.002, 0.0004, 0.02, 0.029, 0.027)
    }
    x = np.arange(len(appliance))  
    width = 0.15  
    multiplier = 0

    fig, ax1 = plt.subplots(layout='constrained')

    colors = {'cnn(s2s)':   '#e76f51',
              'cnn(s2p)':   '#f4a261',
              'fcn':        '#e9c46a',
              'bitcnnilm':  '#2a9d8f',
              'proposed':   '#264653'}

    paddings = [5,[5,5,5,6,3],[9,7,5,3,5],5,[5,7,5,5,3],5]
    paddings = [(5,5,9,5,5,5),(5,5,7,5,7,5),(5,5,5,5,5,5),(5,6,3,5,5,5),(5,3,5,5,3,5)]
    padding_dict = {
        'CNN(S2S)': (6, 3),
        'CNN(S2P)': (9, 7, 5, 3),
        'FCN': (7, 5, 5, 3),
        'BitcnNILM': (5, 7, 5, 5, 3),
        'Proposed': (5, 7, 5, 3)
    }
    for i,(attribute, measurement) in enumerate(penguin_MAE.items()):
        offset = width * multiplier
        rects = ax1.bar(x + offset, measurement, width, label=attribute, color=colors[attribute.lower()])
        # if i == 1:
        #     ax1.bar_label(ax1.containers[0][3:], padding=(6, 3), fontsize=8)
        # elif i == 2:
        #     ax1.bar_label(ax1.containers[0][0:-1], padding=(9, 7, 5, 3), fontsize=8)
        # elif i == 4:
        #     ax1.bar_label(ax1.containers[0][0:4], padding=(5, 7, 5, 5, 3), fontsize=8)
        # else:

        #ax1.bar_label(rects, padding=5, fontsize=8)

        #ax1.bar_label(rects, padding=padding_dict[attribute], fontsize=8)
        multiplier += 1
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    ax1.set_xticks(x + width * 2.5)
    ax1.set_xticklabels(appliance, ha='center', fontsize=13, fontname='Times New Roman')
    ax1.set_ylabel('MAE(Watt)', color='black', fontname='Times New Roman', fontsize=15)
    #ax1.set_title('Results on UK-DALE',fontname='Times New Roman', fontsize=12, fontweight='bold')
    #ax1.set_xticks(x + width * 2.5, appliance)
    ax1.legend(loc='upper left', )
    ax1.set_ylim(0, 14)
    ax1.yaxis.set_tick_params(labelsize=13)

    ax2 = ax1.twinx()
    lines = []
    line_labels = []
    x = np.arange(len(appliance))
    multiplier = 0
    for attribute, measurement in penguin_SAE.items():
        line, = ax2.plot([(x[multiplier] + width * i) for i in range(5)], [penguin_SAE[key][multiplier] for key in penguin_SAE],  color='red',
                         linestyle='--', marker='*')
        lines.append(line)
        line_labels.append(attribute)
        multiplier += 1
    line, = ax2.plot([(x[multiplier] + width * i) for i in range(5)],
                     [penguin_SAE[key][multiplier] for key in penguin_SAE],
                     color='red',
                     linestyle='--',
                     marker='*')
    lines.append(line)

    ax2.spines['right'].set_color('black')
    ax2.yaxis.label.set_color('black')
    ax2.tick_params(axis='y', colors='black')
    ax2.yaxis.set_tick_params(labelsize=13)

    ax2.set_ylabel('SAE (Watt)', color='black', fontname='Times New Roman', fontsize=15)
    ax2.set_ylim(0, 0.4)
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(appliance, fontname='Times New Roman', fontsize=13)
    ax2.legend(lines, ['SAE'], loc='upper right', bbox_to_anchor=(1, 1))
    plt.show()

def group_bar_chart_REDD():
    appliance = ("WM", "DW", "FG", "MW", "Overall")
    penguin_MAE = {
        'CNN(S2S)': (22.857, 19.449, 30.630, 33.272, 26.552),
        'CNN(S2P)': (18.423, 20.048, 28.104, 28.199, 23.694),
        'FCN':      (12.972, 22.153, 32.139, 19.884, 21.787),
        'BitcnNILM': (12.764,16.525, 26.802, 19.456, 18.887),
        'Proposed':  (12.037, 14.418, 28.604, 17.407, 18.116)
    }
    penguin_SAE = {
        'CNN(S2S)': (0.509, 0.557, 0.114, 0.242, 0.357),
        'CNN(S2P)': (0.277, 0.567, 0.180, 0.059, 0.271),
        'FCN':      (0.067, 0.540, 0.019, 0.230, 0.214),
        'BitcnNILM': (0.048, 0.537, 0.064, 0.118, 0.192),
        'Proposed':  (0.012, 0.357, 0.023, 0.157, 0.137)
    }
    x = np.arange(len(appliance))  
    width = 0.15  
    multiplier = 0

    fig, ax1 = plt.subplots(layout='constrained')

    colors = {'cnn(s2s)':   '#e76f51',
              'cnn(s2p)':   '#f4a261',
              'fcn':        '#e9c46a',
              'bitcnnilm':  '#2a9d8f',
              'proposed':   '#264653'}


    for i,(attribute, measurement) in enumerate(penguin_MAE.items()):
        offset = width * multiplier
        rects = ax1.bar(x + offset, measurement, width, label=attribute, color=colors[attribute.lower()])
        #ax1.bar_label(rects, padding=5, fontsize=8)
        #ax1.bar_label(rects, padding=padding_dict[attribute], fontsize=8)
        multiplier += 1
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    ax1.set_xticks(x + width * 2.5)
    ax1.set_xticklabels(appliance, ha='center', fontsize=13, fontname='Times New Roman')
    ax1.set_ylabel('MAE(Watt)', color='black', fontname='Times New Roman', fontsize=15)
    #ax1.set_title('Results on REDD',fontname='Times New Roman', fontsize=12, fontweight='bold')
    #ax1.set_xticks(x + width * 2.5, appliance)
    ax1.legend(loc='upper left', )
    ax1.set_ylim(0, 40)
    ax1.yaxis.set_tick_params(labelsize=13)

    ax2 = ax1.twinx()
    lines = []
    line_labels = []
    x = np.arange(len(appliance))
    multiplier = 0
    for attribute, measurement in penguin_SAE.items():
        line, = ax2.plot([(x[multiplier] + width * i) for i in range(5)], [penguin_SAE[key][multiplier] for key in penguin_SAE],  color='red',
                         linestyle='--', marker='*')
        lines.append(line)
        line_labels.append(attribute)
        multiplier += 1
    line, = ax2.plot([(x[4] + width * i) for i in range(5)],
                     [penguin_SAE[key][-1] for key in penguin_SAE],
                     color='red',
                     linestyle='--',
                     marker='*')
    lines.append(line)

    ax2.spines['right'].set_color('black')
    ax2.yaxis.label.set_color('black')
    ax2.tick_params(axis='y', colors='black')
    ax2.yaxis.set_tick_params(labelsize=13)

    ax2.set_ylabel('SAE (Watt)', color='black', fontname='Times New Roman', fontsize=15)
    ax2.set_ylim(0, 0.8)
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(appliance, fontname='Times New Roman', fontsize=13)
    ax2.legend(lines, ['SAE'], loc='upper right', bbox_to_anchor=(1, 1))
    plt.show()
def group_bar_phi_REDD():
    appliance = ("WM", "DW", "FG", "MW", "Overall")
    penguin_SAE = {
        'Baseline':             (0.277 ,0.567 ,0.180 ,0.059 ,0.271),
        'Kullback-Leibler':     (0.103 ,0.741 ,0.109 ,0.002 ,0.239),
        'Burg_entropy' :        (0.246 ,0.591 ,0.080 ,0.409 ,0.332 ,),
        'X2_distance':          (0.201 ,0.679 ,0.058 ,0.416 ,0.339 ,),
        'Hellinger_distance':(0.207 ,0.611 ,0.041 ,0.435 ,0.324 ,),
        'J_divergence': (0.051, 0.610, 0.051, 0.389, 0.275,),
        'Modified_x2_distance': (0.171, 0.357, 0.033, 0.278, 0.210,),
        #'Average':(0.149 ,0.665 ,0.081 ,0.313 ,0.302 ,),

    }
    penguin_MAE = {
        'Baseline':(18.423 ,20.048 ,28.104 ,28.199 ,23.694 ,),
        'Kullback-Leibler': (13.658 ,21.991 ,31.041 ,25.320 ,23.002 ,),
        'Burg_entropy' : (13.394 ,18.204 ,31.431 ,20.769 ,20.950 ,),
        'X2_distance':  (12.864 ,20.362 ,31.666 ,18.378 ,20.817 ,),
        'Hellinger_distance':(13.006 ,18.509 ,31.601 ,19.681 ,20.699 ,),
        'J_divergence': (11.866, 19.901, 30.940, 18.908, 20.404,),
        'Modified_x2_distance': (12.037, 14.418, 28.604, 17.407, 18.116,),
    }
    x = np.arange(len(appliance))  
    width = 0.12  
    multiplier = 0

    fig, ax1 = plt.subplots(layout='constrained')

    colors = {'Baseline':              '#797d62',
              'Kullback-Leibler':      '#9b9b7a',
              'modified_x2_distance':  '#d9ae94',
              'Burg_entropy' :         '#f1dca7',
              'J_divergence':          '#ffcb69',
              'x2_distance':            '#d08c60',
              'Hellinger_distance':     '#997b66',
              }
    colors = {'Baseline': '#2ec4b6',
              'Kullback-Leibler': '#ff595e',
              'modified_x2_distance': '#1982c4',
              'Burg_entropy': '#ffca3a',
              'J_divergence': '#8ac926',
              'x2_distance': '#d62828',
              'Hellinger_distance': '#6a4c93',
              }
    colors = {'Baseline': '#3851a3',
              'Kullback-Leibler': '#72aacc',
              'modified_x2_distance': '#cae9f3',
              'Burg_entropy': '#fefbba',
              'J_divergence': '#fdba6c',
              'x2_distance': '#eb5d3b',
              'Hellinger_distance': '#a90226',
              }
    colors = {'Baseline': '#3851a3',
              'Kullback-Leibler': '#72aacc',
              'Modified_x2_distance': '#a90226',
              'Burg_entropy': '#cae9f3',
              'J_divergence': '#eb5d3b',
              'X2_distance': '#fefbba',
              'Hellinger_distance': '#fdba6c',
              }

    for i,(attribute, measurement) in enumerate(penguin_MAE.items()):
        offset = width * multiplier
        rects = ax1.bar(x + offset, measurement, width,  color=colors[attribute], )
        multiplier += 1

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    ax1.set_xticks(x + width * 3.5)
    ax1.set_xticklabels(appliance, ha='center', fontsize=15, fontname='Times New Roman')
    ax1.set_ylabel('MAE(Watt)', color='black', fontname='Times New Roman', fontsize=15)
    ax1.tick_params(axis='y', colors='black', labelsize=15, direction='in')
    ax1.set_ylim(0, 40)

    ax2 = ax1.twinx()
    lines = []
    line_labels = []
    x = np.arange(len(appliance))
    multiplier = 0
    for _ in range(len(appliance)):
        line, = ax2.plot([(x[multiplier] + width * i) for i in range(7)], [penguin_SAE[key][multiplier] for key in penguin_SAE],  color='red',
                         linestyle='--', marker='*')
        lines.append(line)
        multiplier += 1
    line, = ax2.plot([(x[4] + width * i) for i in range(7)],
                     [penguin_SAE[key][-1] for key in penguin_SAE],
                     color='red',
                     linestyle='--',
                     marker='*')
    lines.append(line)

    ax2.spines['right'].set_color('black')
    ax2.yaxis.label.set_color('black')
    ax2.tick_params(axis='y', colors='black',labelsize=15, direction='in')
    ax2.tick_params(axis='x', colors='black',labelsize=12, direction='in')

    ax1.set_xlabel('Appliance',fontsize=15)
    ax2.set_ylabel('SAE (Watt)', color='black', fontname='Times New Roman', fontsize=15)
    ax2.set_ylim(0, 0.8)
    ax2.set_xticks(x + width * 3,)
    ax2.set_xticklabels(appliance, fontname='Times New Roman', fontsize=10)
    #ax2.legend(lines, ['SAE'], loc='upper right', bbox_to_anchor=(1, 1))
    plt.show()
def group_bar_phi_UK():
    appliance = ("WM", "KT", "DW", "FG", "MW", "Overall")
    penguin_SAE = {
        'Baseline': (0.098, 0.023, 0.021, 0.096, 0.182, 0.084,),
        'Kullback-Leibler':     (0.045 ,0.032 ,0.009 ,0.025 ,0.094 ,0.041 ,),
        'Burg_entropy' :        (0.057 ,0.004 ,0.006 ,0.019 ,0.014 ,0.020 ,),
        'X2_distance': (0.086, 0.032, 0.003, 0.007, 0.006, 0.027,),
        'Hellinger_distance': (0.081, 0.010, 0.021, 0.030, 0.066, 0.042,),
        'J_divergence':         (0.047 ,0.026 ,0.005 ,0.012 ,0.024 ,0.023 ,),
        'Modified_x2_distance': (0.068, 0.004, 0.004, 0.020, 0.026, 0.024,),

        #'Average':(0.149 ,0.665 ,0.081 ,0.313 ,0.302 ,),

    }
    penguin_MAE = {
        'Baseline': (5.704, 3.708, 3.633, 11.209, 2.200, 5.291,),
        'Kullback-Leibler': (3.258 ,3.212 ,3.101 ,5.925 ,2.251 ,3.549 ,),
        'Burg_entropy': (3.493, 4.066, 2.828, 5.779, 2.126, 3.658,),
        'X2_distance': (3.461, 3.325, 2.991, 5.358, 1.833, 3.394,),
        'Hellinger_distance': (3.083, 3.586, 4.684, 5.658, 2.130, 3.828,),
        'J_divergence': (3.333, 4.195, 3.410, 5.479, 2.129, 3.709,),
        'Modified_x2_distance': (3.344, 3.924, 3.001, 6.138, 2.103, 3.702,),


    }
    x = np.arange(len(appliance))  
    width = 0.12  
    multiplier = 0

    fig, ax1 = plt.subplots(layout='constrained')

    colors = {'Baseline':              '#797d62',
              'Kullback-Leibler':      '#9b9b7a',
              'Modified_x2_distance':  '#d9ae94',
              'Burg_entropy' :         '#f1dca7',
              'J_divergence':          '#ffcb69',
              'X2_distance':            '#d08c60',
              'Hellinger_distance':     '#997b66',
              }
    colors = {'Baseline': '#fd98c9',
              'Kullback-Leibler': '#ad91cb',
              'Modified_x2_distance': '#fe61ad',
              'Burg_entropy': '#7b52ae',
              'J_divergence': '#2049ff',
              'X2_distance': '#67329f',
              'Hellinger_distance': '#6b86ff',
              }
    colors = {'Baseline': '#2ec4b6',
              'Kullback-Leibler': '#ff595e',
              'Modified_x2_distance': '#1982c4',
              'Burg_entropy': '#ffca3a',
              'J_divergence': '#8ac926',
              'x2_distance': '#d62828',
              'Hellinger_distance': '#6a4c93',
              }
    colors = {'Baseline': '#f94144',
              'Kullback-Leibler': '#f3722c',
              'modified_x2_distance': '#f8961e',
              'Burg_entropy': '#f9c74f',
              'J_divergence': '#90be6d',
              'x2_distance': '#43aa8b',
              'Hellinger_distance': '#577590',
              }
    colors = {'Baseline': '#999999',
              'Kullback-Leibler': '#E7DAD2',
              'modified_x2_distance': '#82B0D2',
              'Burg_entropy': '#82B0D2',
              'J_divergence': '#FA7F6F',
              'x2_distance': '#FFBE7A',
              'Hellinger_distance': '#8ECFC9',
              }
    colors = {'Baseline': '#3851a3',
              'Kullback-Leibler': '#72aacc',
              'Modified_x2_distance': '#a90226',
              'Burg_entropy': '#cae9f3',
              'J_divergence': '#eb5d3b',
              'X2_distance': '#fefbba',
              'Hellinger_distance': '#fdba6c',
              }

    for i,(attribute, measurement) in enumerate(penguin_MAE.items()):
        offset = width * multiplier
        #rects = ax1.bar(x + offset, measurement, width, label=attribute, color=colors[attribute], )
        rects = ax1.bar(x + offset, measurement, width, color=colors[attribute], )
        #ax1.bar_label(rects, padding=5, fontsize=8)
        #ax1.bar_label(rects, padding=padding_dict[attribute], fontsize=8)
        multiplier += 1

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

    ax1.set_xticks(x + width * 3.5)
    ax1.set_xticklabels(appliance, ha='center', fontsize=15, fontname='Times New Roman')
    ax1.set_ylabel('MAE(Watt)', color='black', fontname='Times New Roman', fontsize=15)
    ax1.tick_params(axis='y', colors='black', labelsize=15, direction='in')
    #ax1.legend(loc='upper left',fontsize=10 )
    ax1.set_ylim(0, 15)

    ax2 = ax1.twinx()
    lines = []
    line_labels = []
    x = np.arange(len(appliance))
    multiplier = 0
    for _ in range(len(appliance)):
        line, = ax2.plot([(x[multiplier] + width * i) for i in range(7)], [penguin_SAE[key][multiplier] for key in penguin_SAE],  color='red',
                         linestyle='--', marker='*')
        lines.append(line)
        multiplier += 1
    # line, = ax2.plot([(x[4] + width * i) for i in range(7)],
    #                  [penguin_SAE[key][-1] for key in penguin_SAE],
    #                  color='red',
    #                  linestyle='--',
    #                  marker='*')
    lines.append(line)

    ax2.spines['right'].set_color('black')
    ax2.yaxis.label.set_color('black')
    ax2.tick_params(axis='y', colors='black',labelsize=15, direction='in')
    ax2.tick_params(axis='x', colors='black',labelsize=12, direction='in')

    ax1.set_xlabel('Appliance',fontsize=15)
    ax2.set_ylabel('SAE (Watt)', color='black', fontname='Times New Roman', fontsize=15)
    ax2.set_ylim(0, 0.2)
    ax2.set_xticks(x + width * 3,)
    ax2.set_xticklabels(appliance, fontname='Times New Roman', fontsize=10)
    #ax2.legend(lines, ['SAE'], loc='upper right', bbox_to_anchor=(1, 1))
    plt.show()

def bar4_unit_UK():
    category_names = ['F1', 'ACCURACY', 'PRECISION', 'RECALL']
    results = {
        'CNN(S2S)': [0.798, 0.979, 0.967, 0.704],
        'CNN(S2P)': [0.803, 0.977, 0.963, 0.719],
        'FCN': [0.845, 0.980, 0.966, 0.762],
        'BitcnNILM': [0.875, 0.984, 0.932, 0.831],
        'Proposed': [0.913, 0.991, 0.871, 0.959],
    }

    colors = {'CNN(S2S)': '#e76f51',
              'CNN(S2P)': '#f4a261',
              'FCN': '#e9c46a',
              'BitcnNILM': '#2a9d8f',
              'Proposed': '#264653'}

    plt.rc('font', family='Times New Roman', size=8)

    np.random.seed(19680801)
    plt.rcdefaults()
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=100, sharey=True, gridspec_kw={'wspace': 0.25, 'hspace': 0.25}, squeeze=True)
    #fig.suptitle('Performance Comparison on UK-DALE', fontsize=12, fontweight='bold')

    method = results.keys()
    #method = list(results.keys())[::-1]
    y_pos = np.arange(len(method))

    for i, category in enumerate(category_names):
        performance = [results[key][i] for key in results.keys()]
        ax = axs[i // 2, i % 2]
        ax.barh(y_pos, performance,  align='center', color=[colors[method] for method in results.keys()])
        ax.set_yticks(y_pos, labels=method)
        ax.invert_yaxis()  
        #ax.set_xlabel('Performance')
        ax.set_title(category)
        ax.set_xlim([0.7, 1])
        # for y, value in zip(y_pos, performance):
        #     ax.text(value, y, f'{value:.3f}', ha='left', va='center', fontsize=8)

    fig.tight_layout()
    plt.show()
def bar4_unit_REDD():
    category_names = ['F1', 'ACCURACY', 'PRECISION', 'RECALL']
    results = {
        'FCN':       [0.585, 0.933, 0.839, 0.486],
        'BitcnNILM': [0.701, 0.961, 0.776, 0.646],
        'Proposed':  [0.709, 0.959, 0.639, 0.796],
    }

    colors = {'CNN(S2S)': '#e76f51',
              'CNN(S2P)': '#f4a261',
              'FCN': '#e9c46a',
              'BitcnNILM': '#2a9d8f',
              'Proposed': '#264653'}

    plt.rc('font', family='Times New Roman', size=8)
    plt.rcdefaults()
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=100, sharey=True, gridspec_kw={'wspace': 0.25, 'hspace': 0.25}, squeeze=True)
    #fig.suptitle('Performance Comparison on REDD', fontsize=12, fontweight='bold')
    method = results.keys()
    y_pos = np.arange(len(method))

    for i, category in enumerate(category_names):
        performance = [results[key][i] for key in results.keys()]
        ax = axs[i // 2, i % 2]
        ax.barh(y_pos, performance,  align='center', color=[colors[method] for method in results.keys()])
        ax.set_yticks(y_pos, labels=method)
        ax.invert_yaxis() 
        #ax.set_xlabel('Performance')
        ax.set_title(category)
        ax.set_xlim([0.4, 1])
        # for y, value in zip(y_pos, performance):
        #     ax.text(value, y, f'{value:.3f}', ha='left', va='center', fontsize=8)

    fig.tight_layout()
    plt.show()

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 8


def line_alpha_UK():
    alpha = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 2]
    overall_MAE = [3.582, 3.090, 3.741, 3.642, 3.863, 3.697, 3.533, 3.801, 4.036]
    overall_SAE = [0.031, 0.030, 0.015, 0.034, 0.059, 0.040, 0.050, 0.034, 0.020]
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    fig, ax1 = plt.subplots()
    f_MAE = interpolate.interp1d(np.arange(len(overall_MAE)), overall_MAE, kind='linear')
    x_new_MAE = np.linspace(0, len(overall_MAE) - 1, 300)
    y_new_MAE = f_MAE(x_new_MAE)
    ax1.plot(x_new_MAE, y_new_MAE, color='#3a86ff', alpha=1, label='MAE', linewidth=2)#marker='*')
    ax1.set_xticks(np.arange(len(alpha)))
    ax1.set_xticklabels(alpha)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', color='gray')  
    ax1.grid(axis='x', alpha=0.3, linestyle='--', color='gray')  
    #ax1.plot(x_markers, y_markers, 'o', color='black', markersize=8, label='Markers')
    ax1.set_ylabel('MAE (Watt)', color='black', fontname='Times New Roman', fontsize=20)
    ax1.set_ylim(2, 5)
    ax1.axhline(y=4.249, color='#3a86ff', linestyle='-.')
    ax1.legend(loc='upper left',fontsize=18)

    ax2 = ax1.twinx()
    f_SAE = interpolate.interp1d(np.arange(len(overall_SAE)), overall_SAE, kind='linear')
    x_new_SAE = np.linspace(0, len(overall_SAE) - 1, 300)
    y_new_SAE = f_SAE(x_new_SAE)

    ax2.plot(x_new_SAE, y_new_SAE, color='#fb5607', alpha=1, label='SAE', linewidth=2)#marker='*')
    ax2.set_ylabel('SAE (Watt)', color='black', fontname='Times New Roman', fontsize=20)
    ax2.set_ylim(0, 0.08)
    ax2.legend(loc='upper right',fontsize=18)
    #plt.axvline(x=alpha.index(0.02) , color='green', linestyle='--')
    x_markers = [alpha.index(f) for f in alpha ]
    y1_markers = f_MAE(np.array(x_markers))
    y2_markers = f_SAE(np.array(x_markers))
    ax1.plot(x_markers, y1_markers, 'o', label='Markers', color='#3a86ff', markersize=8)
    ax2.plot(x_markers, y2_markers, '*', label='Markers', color='#fb5607', markersize=10)
    ax1.set_xlabel('Unnormalized robust parameter β', color='black', fontname='Times New Roman', fontsize=20)
    ax1.tick_params(axis='x', labelsize=18, direction='in') 
    ax1.tick_params(axis='y', labelsize=18, direction='in')  
    ax2.tick_params(axis='y', labelsize=18, direction='in')  
    plt.tight_layout()
    plt.show()


def loss_redd(directory):
    colors = {'Baseline': '#3851a3',
              'Kullback-Leibler': '#72aacc',
              'Modified_x2_distance': '#a90226',
              'Burg_entropy': '#cae9f3',
              'J_divergence': '#eb5d3b',
              'X2_distance': '#fefbba',
              'Hellinger_distance': '#fdba6c',
              }
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    df = pd.DataFrame()

    for file in csv_files:
        file_path = os.path.join(directory, file)
        temp_df = pd.read_csv(file_path, usecols=[2])
        temp_df.columns = [file]
        df = pd.concat([df, temp_df], axis=1)

    phi_dict = {0: 'Kullback-Leibler', 1: 'Burg_entropy', 2: 'J_divergence', 3: 'X2_distance',
                4: 'Modified_x2_distance', 5: 'Hellinger_distance', 6: 'original'}
    colors = ['#ff595e','#ffca3a','#8ac926','#d62828','#1982c4','#6a4c93','#2ec4b6',]
    colors = ['#72aacc','#cae9f3','#eb5d3b','#fefbba','#a90226','#fdba6c','#3851a3',]
    markers = ['o', '^', 's', 'x', '*', 'd']
    plt.figure(figsize=(10, 8))
    for i,col in enumerate(df.columns):
        marker = markers[i % len(markers)]
        lab = 'Baseline'  
        for key, value in phi_dict.items():
            if value in col:
                lab = value
        plt.plot(df.index, df[col], label=lab, color=colors[i], marker=marker)
    plt.ylim([0,0.001])
    plt.legend(fontsize=15)
    plt.tick_params(axis='x',direction='in')
    plt.tick_params(axis='y',direction='in')
    plt.xlabel('Epoches', color='black', fontname='Times New Roman', fontsize=15)
    plt.ylabel('Loss', color='black', fontname='Times New Roman', fontsize=15)
    #plt.title('Training loss', color='black', fontname='Times New Roman', fontsize=12)
    plt.xticks(fontsize=15)  
    plt.yticks(fontsize=15)  
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0), )
    plt.tight_layout()
    plt.show()

def line_frequency_REDD():
    frequency = [1, 5, 10, 15, 20, 25, 30]
    overall_MAE = [22.410, 22.366, 21.493, 23.692, 20.719, 22.144, 22.074]
    overall_SAE = [0.255, 0.254, 0.307, 0.350, 0.323, 0.261, 0.302]
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    fig, ax1 = plt.subplots()
    f_MAE = interpolate.interp1d(np.arange(len(overall_MAE)), overall_MAE, kind='linear')
    x_new_MAE = np.linspace(0, len(overall_MAE) - 1, 300)
    y_new_MAE = f_MAE(x_new_MAE)
    ax1.plot(x_new_MAE, y_new_MAE, color='#3a86ff', alpha=1, label='MAE', linewidth=2)#marker='*')
    ax1.set_xticks(np.arange(len(frequency)))
    ax1.set_xticklabels(frequency)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', color='gray')  
    ax1.grid(axis='x', alpha=0.3, linestyle='--', color='gray')  

    #ax1.plot(x_markers, y_markers, 'o', color='black', markersize=8, label='Markers')
    ax1.set_ylabel('MAE (Watt)', color='black', fontname='Times New Roman', fontsize=20)
    ax1.set_ylim(20, 25)
    ax1.legend(loc='upper left',fontsize=18)
    ax2 = ax1.twinx()
    f_SAE = interpolate.interp1d(np.arange(len(overall_SAE)), overall_SAE, kind='linear')
    x_new_SAE = np.linspace(0, len(overall_SAE) - 1, 300)
    y_new_SAE = f_SAE(x_new_SAE)

    ax2.plot(x_new_SAE, y_new_SAE, color='#fb5607', alpha=1, label='SAE', linewidth=2)#marker='*')
    ax2.set_ylabel('SAE (Watt)', color='black', fontname='Times New Roman', fontsize=20)
    ax2.set_ylim(0, 0.5)
    ax2.legend(loc='upper right',fontsize=18)
    plt.axvline(x=frequency.index(20) , color='green', linestyle='--')
    x_markers = [frequency.index(f) for f in frequency]
    y1_markers = f_MAE(np.array(x_markers))
    y2_markers = f_SAE(np.array(x_markers))
    ax1.plot(x_markers, y1_markers, 'o', label='MAE Markers', color='#3a86ff', markersize=8)
    ax2.plot(x_markers, y2_markers, '*', label='SAE Markers', color='#fb5607', markersize=10)
    ax1.set_xlabel('Distribution updating frequency \n(b)', color='black', fontname='Times New Roman', fontsize=20)
    ax1.tick_params(axis='x', labelsize=18, direction='in')  
    ax1.tick_params(axis='y', labelsize=18, direction='in')  
    ax2.tick_params(axis='y', labelsize=18, direction='in')  

    plt.tight_layout()
    plt.show()
def line_frequency_UK():
    frequency = [1, 5, 10, 15, 20, 25, 30]
    overall_MAE = [3.742, 	3.825, 	3.800, 	3.695, 	3.593, 	3.635, 	3.679 ]
    overall_SAE = [0.024, 	0.037, 	0.040, 	0.038, 	0.018, 	0.016, 	0.033 ]
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    fig, ax1 = plt.subplots()
    f_MAE = interpolate.interp1d(np.arange(len(overall_MAE)), overall_MAE, kind='linear')
    x_new_MAE = np.linspace(0, len(overall_MAE) - 1, 300)
    y_new_MAE = f_MAE(x_new_MAE)
    ax1.plot(x_new_MAE, y_new_MAE, color='#3a86ff', alpha=1, label='MAE', linewidth=2)#marker='*')
    ax1.set_xticks(np.arange(len(frequency)))
    ax1.set_xticklabels(frequency)
    ax1.grid(axis='y', alpha=0.3, linestyle='--', color='gray')  
    ax1.grid(axis='x', alpha=0.3, linestyle='--', color='gray')  
    #ax1.plot(x_markers, y_markers, 'o', color='black', markersize=8, label='Markers')
    ax1.set_ylabel('MAE (Watt)', color='black', fontname='Times New Roman', fontsize=20)
    ax1.set_ylim(3, 4)
    ax1.legend(loc='upper left',fontsize=18)#
    ax2 = ax1.twinx()
    f_SAE = interpolate.interp1d(np.arange(len(overall_SAE)), overall_SAE, kind='linear')
    x_new_SAE = np.linspace(0, len(overall_SAE) - 1, 300)
    y_new_SAE = f_SAE(x_new_SAE)

    ax2.plot(x_new_SAE, y_new_SAE, color='#fb5607', alpha=1, label='SAE', linewidth=2)#marker='*')
    ax2.set_ylabel('SAE (Watt)', color='black', fontname='Times New Roman', fontsize=20)
    ax2.set_ylim(0, 0.1)
    ax2.legend(loc='upper right',fontsize=18)
    plt.axvline(x=frequency.index(20) , color='green', linestyle='--')
    x_markers = [frequency.index(f) for f in frequency]
    y1_markers = f_MAE(np.array(x_markers))
    y2_markers = f_SAE(np.array(x_markers))
    ax1.plot(x_markers, y1_markers, 'o', label='MAE Markers', color='#3a86ff', markersize=8)
    ax2.plot(x_markers, y2_markers, '*', label='SAE Markers', color='#fb5607', markersize=10)
    ax1.set_xlabel('Distribution updating frequency \n(a)', color='black', fontname='Times New Roman', fontsize=20)
    ax1.tick_params(axis='x', labelsize=18, direction='in')  
    ax1.tick_params(axis='y', labelsize=18, direction='in')  
    ax2.tick_params(axis='y', labelsize=18, direction='in')  

    plt.tight_layout()
    plt.show()
def scale_by_average(data):
    data_array = np.array(data)
    average = np.mean(data_array)
    scaled_data = data_array / average
    return scaled_data.tolist()
def line_frequency_both():
    frequency = [1, 5, 10, 15, 20, 25, 30]
    overall_MAE_REDD = [22.410, 22.366, 21.493, 23.692, 20.719, 22.144, 22.074]
    overall_SAE_REDD = [0.255, 0.254, 0.307, 0.350, 0.323, 0.261, 0.302]
    overall_MAE_UK = [3.742, 	3.825, 	3.800, 	3.695, 	3.593, 	3.635, 	3.679 ]
    overall_SAE_UK = [0.024, 	0.037, 	0.040, 	0.038, 	0.018, 	0.016, 	0.033 ]
    overall_MAE_REDD = scale_by_average(overall_MAE_REDD)
    overall_MAE_UK = scale_by_average(overall_MAE_UK)
    print(overall_MAE_REDD)
    print(overall_MAE_UK)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    fig, ax1 = plt.subplots()
    f_MAE_UK = interpolate.interp1d(np.arange(len(overall_MAE_UK)), overall_MAE_UK, kind='linear')
    x_new_MAE_UK = np.linspace(0, len(overall_MAE_UK) - 1, 300)
    y_new_MAE_UK = f_MAE_UK(x_new_MAE_UK)

    f_MAE_REDD = interpolate.interp1d(np.arange(len(overall_MAE_REDD)), overall_MAE_REDD, kind='linear')
    x_new_MAE_REDD = np.linspace(0, len(overall_MAE_REDD) - 1, 300)
    y_new_MAE_REDD = f_MAE_REDD(x_new_MAE_REDD)

    ax1.plot(x_new_MAE_UK, y_new_MAE_UK, color='#3a86ff', alpha=1, label='MAE_UK', linewidth=2)#marker='*')
    ax1.set_xticks(np.arange(len(frequency)))
    ax1.set_xticklabels(frequency)

    #ax1.plot(x_markers, y_markers, 'o', color='black', markersize=8, label='Markers')
    ax1.set_ylabel('MAE (Watt)', color='black', fontname='Times New Roman', fontsize=15)
    ax1.set_ylim(0.8,1)
    ax1.legend(loc='upper left',fontsize=15)#
    ax2 = ax1.twinx()
    f_SAE_UK = interpolate.interp1d(np.arange(len(overall_SAE_UK)), overall_SAE_UK, kind='linear')
    x_new_SAE_UK = np.linspace(0, len(overall_SAE_UK) - 1, 300)
    y_new_SAE_UK = f_SAE_UK(x_new_SAE_UK)

    f_SAE_REDD = interpolate.interp1d(np.arange(len(overall_SAE_REDD)), overall_SAE_REDD, kind='linear')
    x_new_SAE_REDD = np.linspace(0, len(overall_SAE_REDD) - 1, 300)
    y_new_SAE_REDD = f_SAE_REDD(x_new_SAE_REDD)


    ax2.plot(x_new_MAE_REDD, y_new_MAE_REDD, color='#3a86ff', alpha=1, label='MAE_REDD', linewidth=2)#marker='*')

    #ax2.plot(x_new_SAE_UK, y_new_SAE_UK, color='#fb5607', alpha=1, label='SAE_UK', linewidth=2)#marker='*')
    #ax2.plot(x_new_SAE_REDD, y_new_SAE_REDD, color='#fb5607', alpha=1, label='SAE_REDD', linewidth=2)#marker='*')
    ax2.set_ylabel('SAE (Watt)', color='black', fontname='Times New Roman', fontsize=15)
    ax2.set_ylim(0.8, 1)
    ax2.legend(loc='upper right',fontsize=15)
    plt.axvline(x=frequency.index(20) , color='green', linestyle='--')
    x_markers = [frequency.index(f) for f in frequency]
    y1_markers = f_MAE_REDD(np.array(x_markers))
    y2_markers = f_MAE_UK(np.array(x_markers))
    # y3_markers = f_SAE_REDD(np.array(x_markers))
    # y4_markers = f_SAE_UK(np.array(x_markers))
    ax1.plot(x_markers, y1_markers, 'o', label='MAE Markers', color='#3a86ff', markersize=8)
    ax1.plot(x_markers, y2_markers, '*', label='MAE Markers', color='#3a86ff', markersize=8)
    #ax2.plot(x_markers, y3_markers, '*', label='SAE Markers', color='#fb5607', markersize=10)
    #ax2.plot(x_markers, y4_markers, '*', label='SAE Markers', color='#fb5607', markersize=10)
    ax1.set_xlabel('Frequency', color='black', fontname='Times New Roman', fontsize=15)
    ax1.tick_params(axis='x', labelsize=15, direction='in')  
    ax1.tick_params(axis='y', labelsize=15, direction='in')  
    ax2.tick_params(axis='y', labelsize=15, direction='in')  

    plt.tight_layout()
    plt.show()
def probability_dist(data):
    bins = 10
    breaks = np.linspace(np.min(data), np.max(data), bins)
    counts = {}
    for i in range(len(breaks) - 1):
        left, right = breaks[i], breaks[i + 1]
        label = f"[{left:.15f}, {right:.15f}]"
        count = 0
        for val in data:
            if i == 0:  
                if val >= left and val <= right:
                    count += 1
            elif i == len(breaks) - 2:  
                if val > left and val >= right:
                    count += 1
            else: 
                if val > left and val <= right:
                    count += 1
        counts.update({label: count})
    return counts

def draw_multiple_density_charts(datasets, labels, title='Density Chart'):
    for data, label in zip(datasets, labels):
        sns.kdeplot(data, fill=False, label=label,  alpha=0.7,  bw_adjust=0.5)
    sns.set_palette('colorblind')
    sns.set_style('whitegrid')

    plt.title(title)
    plt.xlabel('Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.xlim(-0.0025,0.0075)
    plt.show()



    