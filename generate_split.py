import argparse
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import ConcatDataset
import random
import json

def basic_stats(data):
    max_v = np.max(data)
    min_v = np.min(data)
    print(max_v)
    print(min_v)

def split_cls_between_client(total_cls, num_client, num_cls_per_client):
    client_cls = []
    num_client_per_cls = {i:0 for i in total_cls}
    for i in range(num_client):
        client_cls.append([])
        for j in range(i, i + num_cls_per_client):
            cls_id = j % len(total_cls)
            cls_id = total_cls[cls_id]
            num_client_per_cls[cls_id] += 1
            client_cls[i].append(cls_id)
    
    return client_cls, num_client_per_cls

def split_data(dataset, num_clients=None, num_train_cls_per_client=None, num_test_cls_per_client=None):
    # caution this only work if class have equal number of data
    data = {}
    for i in range(len(dataset)):
        X, y = dataset[i]
        if y not in data.keys():
            data[y] = []
        data[y].append(i)
    
    # import json
    # with open("labels.json", 'r') as f:
    #     labels = json.load(f)
    # if labels == tmp:
    #     print("labels are the same")
    # else:
    #     print("labels are not the same")
    # import pdb; pdb.set_trace()

    cur_client = {}
    for cls_id in data.keys():
        cur_client[cls_id] = 0
        random.shuffle(data[cls_id])
    

    cls = list(data.keys())
    num_train_cls = int(len(cls) * 0.7)
    num_test_cls = len(cls) - num_train_cls

    test_cls = random.sample(cls, num_test_cls)
    train_cls = [i for i in cls if i not in test_cls]


    client_train_cls, num_client_per_cls = split_cls_between_client(train_cls, num_clients, num_train_cls_per_client)
    client_test_cls, num_client_per_cls1 = split_cls_between_client(test_cls, num_clients, num_test_cls_per_client)                                                           
    num_client_per_cls.update(num_client_per_cls1)
    num_data_per_client_per_cls = {i: int(len(data[i]) / num_client_per_cls[i]) for i in data.keys()}

    client_datas = []
    for client_id in range(num_clients):
        client_data = {"train": [], 'test': [], 'train_labels': [], 'test_labels': []}
        for cls_id in client_train_cls[client_id]:
            l = cur_client[cls_id] * num_data_per_client_per_cls[cls_id]
            r = len(data[cls_id]) if cur_client[cls_id] == num_client_per_cls[cls_id] - 1 else (cur_client[cls_id] + 1) * num_data_per_client_per_cls[cls_id]
            client_data['train'] += data[cls_id][l:r]
            client_data['train_labels'] += [cls_id] * (r-l)
            cur_client[cls_id] += 1
    
        for cls_id in client_test_cls[client_id]:
            l = cur_client[cls_id] * num_data_per_client_per_cls[cls_id]
            r = len(data[cls_id]) if cur_client[cls_id] == num_client_per_cls[cls_id] - 1 else (cur_client[cls_id] + 1) * num_data_per_client_per_cls[cls_id]
            client_data['test'] += data[cls_id][l:r]
            client_data['test_labels'] += [cls_id] * (r-l)
            cur_client[cls_id] += 1
        client_datas.append(client_data)
    
    return client_datas

def split_data1(dataset, num_clients=None, num_train_cls_per_client=None, num_test_cls_per_client=None):
    # this only differ from the above that test data now used for global model
    # caution this only work if class have equal number of data
    data = {}
    for i in range(len(dataset)):
        X, y = dataset[i]
        if y not in data.keys():
            data[y] = []
        data[y].append(i)

    cur_client = {}
    for cls_id in data.keys():
        cur_client[cls_id] = 0
        random.shuffle(data[cls_id])
    

    cls = list(data.keys())
    num_train_cls = int(len(cls) * 0.7)
    num_test_cls = len(cls) - num_train_cls

    test_cls = random.sample(cls, num_test_cls)
    train_cls = [i for i in cls if i not in test_cls]

    client_train_cls, num_client_per_cls = split_cls_between_client(train_cls, num_clients, num_train_cls_per_client)
    num_data_per_client_per_cls = {i: int(len(data[i]) / num_client_per_cls[i]) for i in num_client_per_cls.keys()}


    test_data_ids = []
    test_data_labels = []
    for t_cls in test_cls:
        test_data_ids += data[t_cls]
        test_data_labels += [t_cls] * len(data[t_cls])

    client_datas = []
    for client_id in range(num_clients):
        client_data = {"train": [], 'test': [], 'train_labels': [], 'test_labels': []}
        for cls_id in client_train_cls[client_id]:
            l = cur_client[cls_id] * num_data_per_client_per_cls[cls_id]
            r = len(data[cls_id]) if cur_client[cls_id] == num_client_per_cls[cls_id] - 1 else (cur_client[cls_id] + 1) * num_data_per_client_per_cls[cls_id]
            client_data['train'] += data[cls_id][l:r]
            client_data['train_labels'] += [cls_id] * (r-l)
            cur_client[cls_id] += 1
    
        client_datas.append(client_data)
    
    return {"test_data_ids": test_data_ids, 'test_data_labels': test_data_labels, 'client_data': client_datas}

def split_train_test(dataset, train_ratio=0.7):
    # the train dataset contain <train_ration> of the total class
    data = {}
    for i in range(len(dataset)):
        X, y = dataset[i]
        if y not in data.keys():
            data[y] = []
        data[y].append(i)
    
    cls = list(data.keys())
    num_train_cls = int(len(cls) * 0.7)
    num_test_cls = len(cls) - num_train_cls

    test_cls = random.sample(cls, num_test_cls)
    train_cls = [i for i in cls if i not in test_cls]

    train_data = {}
    test_data = {}
    for cls_id in cls:
        if cls_id in train_cls:
            train_data[cls_id] = data[cls_id]
        else:
            test_data[cls_id] = data[cls_id]
    
    return train_data, test_data

def split_data_dirichlet(data, num_clients, minvol=10, skewness=0.5, min_data_per_class=5):
    """label_skew_dirichlet"""
    min_size = 0
    total_len = sum([len(data[k]) for k in data])
    while min_size < minvol:
        idx_batch = [[] for i in range(num_clients)]
        for k in data:
            np.random.shuffle(data[k])
            proportions = np.random.dirichlet(
                np.repeat(skewness, num_clients)
            )
            ## Balance
            proportions = np.array(
                [
                    p * (len(idx_j) < total_len / num_clients)
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )
            idx_k = list(zip(data[k], [k] * len(data[k])))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    
    client_data = []
    for j in range(num_clients):
        
        cls_data = {}
        for x in idx_batch[j]:
            if x[1] not in cls_data.keys():
                cls_data[x[1]] = []
            cls_data[x[1]].append(x[0])
        
        train_ids = []
        train_labels = []
        for cls in cls_data:
            if(len(cls_data[cls]) < min_data_per_class):
                cls_data[cls] += random.sample(data[cls], min_data_per_class - len(cls_data[cls]))
            train_ids += cls_data[cls]
            train_labels += [cls] * len(cls_data[cls])

        client_data.append({"train": train_ids, "train_labels": train_labels})    
    
    return client_data
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=int, help='client data distribution')
    parser.add_argument("--save_path", type=str, help="path to save the data split")
    
    opts = parser.parse_args()
    # dist 0: iid 
    # 1: label dirichlet
    # hyperparameter
    '''
        format of data split file:
        {
            test_data_ids: list of instance index
            test_data_labels: list of corresponding class labels
            client_cls_ids: list of train class ids
            client_data: [
                {
                    train: list of instance indexes
                    train_labels: list of corresponding class labels
                }
            ]
        }
    '''
    num_clients = 70
    train_ratio = 0.7
    rawdata_path = "./benchmark/cifar100/data/"
    
    train_data = datasets.CIFAR100(
        rawdata_path,
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    test_data = datasets.CIFAR100(
        rawdata_path,
        train=False,
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    dataset = ConcatDataset([train_data, test_data])
    
    if opts.dist == 0:
        num_train_cls_per_client = 20
        num_test_cls_per_client = 8
        data_split = split_data1(dataset, num_clients, num_train_cls_per_client, num_test_cls_per_client)
    elif opts.dist == 1:
        train_data, test_data = split_train_test(dataset, train_ratio)
        client_data = split_data_dirichlet(train_data, num_clients, min_data_per_class=10)
        
        test_data_ids = []
        test_data_labels = []
        for x, y in test_data.items():
            test_data_ids += y
            test_data_labels += [x] * len(y)

        data_split = {
            "client_data": client_data,
            "test_data_ids": test_data_ids,
            "test_data_labels": test_data_labels
        }
    with open(opts.save_path, 'w') as f:
        json.dump(data_split, f)



