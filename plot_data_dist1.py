import os
import json
import argparse
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path')
    parser.add_argument('--plot_type')
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.save_path) , exist_ok=True)
    with open(args.data_path) as f:
        data = json.load(f)
    client_data = data['client_data']
    client_cnt = {}
    num_cls = 70
    for i, t in enumerate(client_data):
        t1 = dict(Counter(t['train_labels']))
        for j in range(num_cls):
            if j not in t1:
                t1[j] = 0
        client_cnt[i] = t1
    num_client_plot = 70
    num_cls_plot = 70
    plt.figure()
    plt.xlabel('Client')
    plt.ylabel('Number of data')
    colors = plt.cm.tab20(np.linspace(0, 1, num_cls_plot))
    prev_data = None
    max_val = 0
    for i in range(num_cls_plot):
        data = []
        for j in range(num_client_plot):
            data.append(client_cnt[j][i])
            max_val = max(max_val, client_cnt[j][i])
        print(sum(data))
        plt.bar(range(num_client_plot), data, label=f'Class {i}', color=colors[i], bottom=prev_data)
        prev_data = data
    plt.savefig(args.save_path)
    plt.clf()
    print(max_val)
    

    
    