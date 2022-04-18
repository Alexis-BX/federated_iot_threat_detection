import pandas as pd
import numpy as np
import os

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

dic_protocoles = {protocol:i for i,protocol in enumerate(['DNS', 'ARP', 'DHCP', 'ICMP', 'TCP', 'TLSv1.2', 'NTP', 'ICMPv6'])}
dic_attacks = {attack:i for i, attack in enumerate(['normal', 'mirai'])}


def get_data(data_type, device, name, extension, sep):
    """
    data_type: type of data to return: Normal, Mirai, RouterSploit, UFONet
    device: the device name: archer, camera, indoor, philips, wr940n
    name: network or energy
    extension: file format
    sep: file delimiter
    """
    path = './../QRS_dataset'
    types = {'mirai':'Mirai', 'normal':'Normal', 'rs':'RouterSploit', 'ufo':'UFONet'}

    if data_type in types.keys():
        data_type = types[data_type]
    
    if data_type not in os.listdir(path):
        print('Bad data type')
        return False
    
    path += f'/{data_type}'
    if data_type == 'Normal': 
        path += '/Normal'
    
    path += f'/{name}/{device}'
    
    if data_type == 'Normal': 
        path += '-normal1'
    else:
        path += '-attack1'

    path += extension

    return pd.read_csv(path, sep=sep, on_bad_lines='skip')

def get_data_network(data_type, device):
    data = get_data(data_type, device, 'network', '.csv', ';')
    data.drop(['Info', 'No.'], axis=1, inplace=True)
    
    return data

def get_data_energy(data_type, device):
    data = get_data(data_type, device, 'energy', '.amp', ',')
    data['time'] /= 1000    # converting millis seconds
    data.drop('timestamp', axis=1, inplace=True)

    # rounding time to the nearest second
    data['time'] = data['time'].apply(round)
    data.drop_duplicates('time', inplace=True)
    data.reset_index(inplace=True, drop=True)

    return data
    
def get_data_and_preprocess(data_type, device):
    """
    data_type: type of data to return: Normal, Mirai, RouterSploit, UFONet
    device: the device name: archer, camera, indoor, philips, wr940n
    """
    data_network = get_data_network(data_type, device)
    data_energy = get_data_energy(data_type, device)

    # adding the energy usage for every packet
    data_network['Energy_Time'] = data_network['Time'].apply(round)
    data_network.drop(inplace=True, index=np.where(data_network['Energy_Time']>=data_energy['time'].max())[0])
    data_network['energy'] = list(data_energy.loc[list(data_network['Energy_Time']), 'energy'])
    data_network.drop('Energy_Time', axis=1, inplace=True)

    data_network['target'] = data_type

    data_network.reset_index(inplace=True, drop=True)

    return data_network, data_energy

def translate_to_ints(data):
    data['Protocol'] = data['Protocol'].map(dic_protocoles)

    le = preprocessing.LabelEncoder()
    data['Source'] = le.fit_transform(data.Source.values)
    data['Destination'] = le.fit_transform(data.Destination.values)

    data['target'] = data['target'].map(dic_attacks)

    return data

def translate_to_readable(data):
    dic_protocoles_inv = {v:k for k,v in dic_protocoles.items()}
    data['Protocol'] = data['Protocol'].map(dic_protocoles_inv)

    dic_attacks_inv = {v:k for k,v in dic_attacks.items()}
    data['target'] = data['target'].map(dic_attacks_inv)

    return data

def requeset_data():
    """ Create a complete table with all of the packet data """
    normal_network, normal_energy = get_data_and_preprocess('normal', 'archer')
    mirai_network, mirai_energy = get_data_and_preprocess('mirai', 'archer')
    
    # Get both network datasets in the same time-frame
    max_time = min(normal_network['Time'].max(), normal_network['Time'].max())

    normal_network.drop(inplace=True, index=np.where(normal_network['Time']>max_time)[0])
    normal_energy.drop(inplace=True, index=np.where(normal_energy['time']>max_time)[0])
    mirai_network.drop(inplace=True, index=list(np.where(mirai_network['Time']>max_time)[0]))
    mirai_energy.drop(inplace=True, index=list(np.where(mirai_energy['time']>max_time)[0]))

    normal_network.reset_index(inplace=True, drop=True)
    normal_energy.reset_index(inplace=True, drop=True)
    mirai_network.reset_index(inplace=True, drop=True)
    mirai_energy.reset_index(inplace=True, drop=True)

    # merge both dataframes together
    Final_merge = normal_network.append(mirai_network)
    Final_merge = Final_merge[Final_merge.Protocol != 0]

    # feature mapping
    Final_merge = translate_to_ints(Final_merge)

    return Final_merge

def get_1_all(Final_merge=None):
    """ Extract 1 packet of each type for the normal and attack """
    if Final_merge is None:
        Final_merge = requeset_data()

    y = Final_merge[['target']]
    X = Final_merge.drop(['target', ], axis = 1)
    
    # one of each 
    idx = [max(list(np.where(np.logical_and(Final_merge['target']==i, Final_merge['Protocol']==j))[0])+[0]) for i in range(2) for j in range(8)]

    X_init, y_init = X.iloc[idx], y.iloc[idx]
    return X_init, y_init

def request_data_server():
    """ Create the datasets to be used by the server """
    Final_merge = requeset_data()
    X_init, y_init = get_1_all(Final_merge)

    y = Final_merge[['target']]
    X = Final_merge.drop(['target', ], axis = 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.6, random_state = 42)
    
    return (X_train, y_train), (X_test, y_test), (X_init, y_init)

def request_data_client(num):
    """ Create the datasets to be used by the client """
    Final_merge = requeset_data()
    X_init, y_init = get_1_all(Final_merge)

    split = Final_merge.shape[0]//2
    if num%2:
        Final_merge = Final_merge.iloc[:split]
    else:
        Final_merge = Final_merge.iloc[split:]

    # Target variable and train set
    y = Final_merge[['target']]
    X = Final_merge.drop(['target', ], axis = 1)

    # Split test and train data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.6, random_state = 42)

    # need to have one of each for the trees to have the correct shape
    X_train = X_train.append(X_init).reset_index(drop=True)
    y_train = y_train.append(y_init).reset_index(drop=True)
    
    return (X_train, y_train), (X_test, y_test), (X_init, y_init)    