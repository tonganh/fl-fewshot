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
            client_data['train_labels'] += [cls_id] * (r-l+1)
            cur_client[cls_id] += 1
    
        for cls_id in client_test_cls[client_id]:
            l = cur_client[cls_id] * num_data_per_client_per_cls[cls_id]
            r = len(data[cls_id]) if cur_client[cls_id] == num_client_per_cls[cls_id] - 1 else (cur_client[cls_id] + 1) * num_data_per_client_per_cls[cls_id]
            client_data['test'] += data[cls_id][l:r]
            client_data['test_labels'] += [cls_id] * (r-l+1)
            cur_client[cls_id] += 1

        client_datas.append(client_data)
    
    return client_datas


if __name__ == "__main__":
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
    # dataset.targets
    num_clients = 70
    num_train_cls_per_client = 20
    num_test_cls_per_client = 8
    data = split_data(dataset, num_clients, num_train_cls_per_client, num_test_cls_per_client)
    with open("client_data.json", 'w') as f:
        json.dump(data, f)



