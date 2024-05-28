from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
import json
import torch
import random
rawdata_path = 'benchmark/cifar100/data'
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
data = []
for i in range(len(dataset)):
    data.append((i, dataset[i][1]))

class_ids = list(range(100))
random.shuffle(class_ids)
num_test_cls = 30
num_train_cls = 70

test_cls = class_ids[:num_test_cls]
train_cls = class_ids[num_test_cls:]

train_data_ids = []
train_data_labels = []
test_data_ids = []
test_data_labels = []

for idx, label in data:
    if label in test_cls:
        train_data_ids.append(idx)
        train_data_labels.append(label)
    else:
        test_data_ids.append(idx)
        test_data_labels.append(label)

split = {
    "test_data_ids": test_data_ids,
    "test_data_labels": test_data_labels,
    "client_data": [
        {
            "train": train_data_ids,
            "train_labels": train_data_labels
        }
    ]
}
with open("data_split/cifar100/c1.json", "w") as f:
    json.dump(split, f)

