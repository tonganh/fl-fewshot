from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, DefaultTaskGen, XYTaskReader
from torch.utils.data import DataLoader
import numpy as np


# def iid_partition(generator):
#     print(generator)
#     # import pdb; pdb.set_trace()
#     labels = np.unique(generator.train_data.y)
#     local_datas = [[] for _ in range(generator.num_clients)]
#     for label in labels:
#         permutation = np.random.permutation(
#             np.where(generator.train_data.y == label)[0]
#         )
#         split = np.array_split(permutation, generator.num_clients)
#         for i, idxs in enumerate(split):
#             local_datas[i] += idxs.tolist()
#     # import pdb; pdb.set_trace()
#     return local_datas


class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients=1, skewness=0.5, number_class_per_client=2):
        super(TaskGen, self).__init__(
            benchmark="cifar100",
            dist_id=dist_id,
            num_clients=num_clients,
            skewness=skewness,
            rawdata_path="./benchmark/cifar100/data",
            number_class_per_client=number_class_per_client,
        )
        self.num_classes = 100
        if self.dist_id == 9:
            self.save_data = self.XYData_to_json_fewshot_seen_unseen
        else:
            self.save_data = self.XYData_to_json
        # self.partition = iid_partition

    def load_data(self):
        self.train_data = datasets.CIFAR100(
            self.rawdata_path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        )
        self.test_data = datasets.CIFAR100(
            self.rawdata_path,
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        )

    def convert_data_for_saving(self):
        train_x = [
            self.train_data[did][0].tolist() for did in range(len(self.train_data))
        ]
        train_y = [self.train_data[did][1] for did in range(len(self.train_data))]
        test_x = [self.test_data[did][0].tolist() for did in range(len(self.test_data))]
        test_y = [self.test_data[did][1] for did in range(len(self.test_data))]
        self.train_data = {"x": train_x, "y": train_y}
        self.test_data = {"x": test_x, "y": test_y}
        return


class TaskReader(XYTaskReader):
    def __init__(self, taskpath=""):
        super(TaskReader, self).__init__(taskpath)
        self.DataLoader = DataLoader


class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)
