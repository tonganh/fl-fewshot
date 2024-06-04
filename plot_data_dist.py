import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import Counter
import os

def plot_data_dist(data, labels=None, save_path=None):
    if labels is None:
        labels = [f'client_{i}' for i in range(len(data))]
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels)
    plt.xticks(rotation=-45, ha='left', rotation_mode='anchor')

    plt.xlabel('Client')
    plt.ylabel('Number of data per class')

    save_path = 'test.png' if save_path is None else save_path
    plt.savefig(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir')
    parser.add_argument('--plot_type')
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.plot_type == 'client':
        with open(args.data_path) as f:
            data = json.load(f)
        client_data = data['client_data']
        t2 = []
        labels = []
        for i, t in enumerate(client_data):
            t1 = dict(Counter(t['train_labels']))
            t1 = list(t1.values())
            t2.append(t1)
            labels.append(f'c_{i}')
        
        chunk = 10
        save_dir = args.save_dir
        for i in range(0, len(t2), chunk):
            l = i
            r = min(i+chunk, len(t2))
            plot_data_dist(t2[l:r], labels[l:r], f'{save_dir}/c_{l}_{r}.png')
    
    elif args.plot_type == 'data':
        
        with open(args.data_path) as f:
            data = json.load(f)
        client_data = data['client_data']
        data_hist = {}
        for i, t in enumerate(client_data):
            t1 = dict(Counter(t['train_labels']))
            for t2 in t1:
                if t2 not in data_hist:
                    data_hist[t2] = []
                data_hist[t2].append(t1[t2])
        bin_edges = np.arange(9.5, 25.5, 1)
        for t in data_hist:
            save_path = os.path.join(args.save_dir, f'label{t}.png')
            plt.figure()
            plt.hist(data_hist[t], bins=bin_edges, edgecolor='black')
            plt.savefig(save_path)



    


    
