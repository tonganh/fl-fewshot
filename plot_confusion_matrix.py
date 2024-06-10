import json
import argparse
import random
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import shutil
import pandas as pd


def plot_confusion_matrix(data, save_path):
    preds = [i[0] for i in data]
    gts = [i[1] for i in data]
    cm = confusion_matrix(gts, preds, normalize='true')
    
    plt.figure()
    plt.imshow(cm, cmap='hot', interpolation='nearest')
    
    # Adding text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, '{:.1f}'.format(cm[i, j]),
                    horizontalalignment='center',
                    color='white' if cm[i, j] > thresh else 'black')
    
    plt.colorbar()
    plt.savefig(save_path)
    plt.clf()

def compute_confusion_matrix(data, save_path):
    preds = [i[0] for i in data]
    gts = [i[1] for i in data]
    num_cls = 70
    cm = confusion_matrix(gts, preds, normalize='true', labels=range(num_cls))
    df = pd.DataFrame(cm, index=range(num_cls), columns=range(num_cls))
    df.to_excel(save_path)
    return cm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--clear_dir', action='store_true', default=False)
    args = parser.parse_args()
    if args.clear_dir and os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    data = []
    with open(args.data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    num_clients = 5
    total_clients = 70
    client_ids = random.sample(range(total_clients), num_clients)

    client_res = {}
    global_res = {}
    for i in range(len(data)):
        client_id = data[i]['client_id']
        round_id = data[i]['round_id']
        preds = data[i]['preds']
        if client_id == -1:
            global_res[round_id] = preds
        elif client_id in client_ids:
            if client_id not in client_res:
                client_res[client_id] = {}
            is_local = data[i]['data']
            if is_local not in client_res[client_id]:
                client_res[client_id][is_local] = {}
            client_res[client_id][is_local][round_id] = preds
    
    total_acc = {}
    for round_id in global_res:
        # save_path = os.path.join(args.save_dir, f'global_{round_id}.png')
        # plot_confusion_matrix(global_res[round_id], save_path)
        save_path = os.path.join(args.save_dir, f'global_{round_id}.xlsx')
        cm = compute_confusion_matrix(global_res[round_id], save_path)   
        
        id = save_path.strip('.xlsx')
        tmp = global_res[round_id]
        pred = [i[0] for i in tmp]
        gt = [i[1] for i in tmp]
        acc = np.mean(np.array(pred) == np.array(gt))
        total_acc[id] = acc
    
    for client_id in client_res:
        for is_local in client_res[client_id]:
            for round_id in client_res[client_id][is_local]:
                # save_path = os.path.join(args.save_dir, f'client{client_id}_{is_local}_{round_id}.png')
                # plot_confusion_matrix(client_res[client_id][is_local][round_id], save_path)
                save_path = os.path.join(args.save_dir, f'client{client_id}_{is_local}_{round_id}.xlsx')
                cm = compute_confusion_matrix(client_res[client_id][is_local][round_id], save_path)

                id = save_path.strip('.xlsx')
                tmp = global_res[round_id]
                pred = [i[0] for i in tmp]
                gt = [i[1] for i in tmp]
                acc = np.mean(np.array(pred) == np.array(gt))
                total_acc[id] = acc
    
    df = pd.DataFrame({'acc': total_acc.values()}, index=total_acc.keys())
    save_path = os.path.join(args.save_dir, 'acc.xlsx')
    df.to_excel(save_path)
