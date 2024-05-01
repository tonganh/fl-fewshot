"""
DISTRIBUTION OF DATASET
-----------------------------------------------------------------------------------
balance:
    iid:            0 : identical and independent distributions of the dataset among clients
    label skew:     1 Quantity:  each party owns data samples of a fixed number of labels.
                    2 Dirichlet: each party is allocated a proportion of the samples of each label according to Dirichlet distribution.
                    3 Shard: each party is allocated the same numbers of shards that is sorted by the labels of the data
-----------------------------------------------------------------------------------
depends on partitions:
    feature skew:   4 Noise: each party owns data samples of a fixed number of labels.
                    5 ID: For Shakespeare\FEMNIST, we divide and assign the writers (and their characters) into each party randomly and equally.
-----------------------------------------------------------------------------------
imbalance:
    iid:            6 Vol: only the vol of local dataset varies.
    niid:           7 Vol: for generating synthetic data
"""

import torch
import ujson
import numpy as np
import os.path
import random
import urllib
import zipfile
import os
import ssl
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context
import importlib


def set_random_seed(seed=0):
    """Set random seed"""
    random.seed(3 + seed)
    np.random.seed(97 + seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def download_from_url(url=None, filepath="."):
    """Download dataset from url to filepath."""
    if url:
        urllib.request.urlretrieve(url, filepath)
    return filepath


def extract_from_zip(src_path, target_path):
    """Unzip the .zip file (src_path) to target_path"""
    f = zipfile.ZipFile(src_path)
    f.extractall(target_path)
    targets = f.namelist()
    f.close()
    return [os.path.join(target_path, tar) for tar in targets]


class BasicTaskGen:
    _TYPE_DIST = {
        0: "iid",
        1: "label_skew_quantity",
        2: "label_skew_dirichlet",
        3: "label_skew_shard",
        4: "feature_skew_noise",
        5: "feature_skew_id",
        6: "iid_volumn_skew",
        7: "niid_volumn_skew",
        8: "concept skew",
        9: "concept and feature skew and balance",
        10: "concept and feature skew and imbalance",
    }
    _TYPE_DATASET = ["2DImage", "3DImage", "Text", "Sequential", "Graph", "Tabular"]

    def __init__(
        self,
        benchmark,
        dist_id,
        skewness,
        rawdata_path,
        seed=0,
        number_class_per_client=2,
    ):
        self.benchmark = benchmark
        self.rootpath = "./fedtask"
        if not os.path.exists(self.rootpath):
            os.mkdir(self.rootpath)

        self.rawdata_path = rawdata_path
        self.dist_id = dist_id
        self.dist_name = self._TYPE_DIST[dist_id]
        self.skewness = 0 if dist_id == 0 else skewness
        self.num_clients = -1
        self.seed = seed
        self.number_class_per_client = number_class_per_client
        set_random_seed(self.seed)

    def run(self):
        """The whole process to generate federated task."""
        pass

    def load_data(self):
        """Download and load dataset into memory."""
        pass

    def partition(self):
        """Partition the data according to 'dist' and 'skewness'"""
        pass

    def save_data(self):
        """Save the federated dataset to the task_path/data.
        This algorithm should be implemented as the way to read
        data from disk that is defined by DataReader.read_data()
        """
        pass

    def save_info(self):
        """Save the task infomation to the .json file stored in taskpath"""
        pass

    def get_taskname(self):
        """Create task name and return it."""
        taskname = "_".join(
            [
                self.benchmark,
                "cnum" + str(self.num_clients),
                "dist" + str(self.dist_id),
                "skew" + str(self.skewness).replace(" ", ""),
                "seed" + str(self.seed),
            ]
        )
        return taskname

    def get_client_names(self):
        k = str(len(str(self.num_clients)))
        return [("Client{:0>" + k + "d}").format(i) for i in range(self.num_clients)]

    def create_task_directories(self):
        """Create the directories of the task."""
        taskname = self.get_taskname()
        taskpath = os.path.join(self.rootpath, taskname)
        os.makedirs(taskpath)
        os.makedirs(os.path.join(taskpath, "record"))

    def _check_task_exist(self):
        """Check whether the task already exists."""
        taskname = self.get_taskname()
        return os.path.exists(os.path.join(self.rootpath, taskname))


class DefaultTaskGen(BasicTaskGen):
    def __init__(
        self,
        benchmark,
        dist_id,
        skewness,
        rawdata_path,
        num_clients=1,
        minvol=10,
        seed=0,
        number_class_per_client=2,
    ):
        super(DefaultTaskGen, self).__init__(
            benchmark, dist_id, skewness, rawdata_path, seed, number_class_per_client
        )
        self.minvol = minvol
        self.num_classes = -1
        self.train_data = None
        self.test_data = None
        self.num_clients = num_clients
        self.cnames = self.get_client_names()
        self.taskname = self.get_taskname()
        self.taskpath = os.path.join(self.rootpath, self.taskname)
        self.save_data = self.XYData_to_json
        self.label_after_sort_case3 = None
        self.number_class_per_client = number_class_per_client
        self.datasrc = {"lib": None, "class_name": None, "args": []}

    def run(self):
        """Generate federated task"""
        # check if the task exists
        if not self._check_task_exist():
            self.create_task_directories()
        else:
            print("Task Already Exists.")
            return
        # read raw_data into self.train_data and self.test_data
        print("-----------------------------------------------------")
        print("Loading...")
        self.load_data()
        print("Done.")
        # partition data and hold-out for each local dataset
        print("-----------------------------------------------------")
        print("Partitioning data...")
        if self.dist_id == 9:
            local_datas_seen, local_datas_unseen = self.partition()
            train_cidxs = local_datas_seen
            valid_cidxs = local_datas_unseen
        else:
            local_datas = self.partition()
            train_cidxs, valid_cidxs = self.local_holdout(
                local_datas, rate=0.8, shuffle=True
            )
        print("Done.")
        # save task infomation as .json file and the federated dataset
        print("-----------------------------------------------------")
        print("Saving data...")
        self.save_info()
        self.save_data(train_cidxs, valid_cidxs)
        print("Done.")
        return

    def load_data(self):
        """load and pre-process the raw data"""
        return

    def partition(self):
        # Partition self.train_data according to the delimiter and return indexes of data owned by each client as [c1data_idxs, ...] where the type of each element is list(int)
        if self.dist_id == 0:
            """IID"""
            d_idxs = np.random.permutation(len(self.train_data))
            local_datas = np.array_split(d_idxs, self.num_clients)

        elif self.dist_id == 1:
            """label_skew_quantity"""
            self.skewness = min(max(0, self.skewness), 1.0)
            dpairs = [
                [did, self.train_data[did][-1]] for did in range(len(self.train_data))
            ]
            num = max(int((1 - self.skewness) * self.num_classes), 1)
            K = self.num_classes
            local_datas = [[] for _ in range(self.num_clients)]
            if num == K:
                for k in range(K):
                    idx_k = [p[0] for p in dpairs if p[1] == k]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, self.num_clients)
                    for cid in range(self.num_clients):
                        local_datas[cid].extend(split[cid].tolist())
            else:
                times = [0 for _ in range(self.num_classes)]
                contain = []
                for i in range(self.num_clients):
                    current = [i % K]
                    times[i % K] += 1
                    j = 1
                    while j < num:
                        ind = random.randint(0, K - 1)
                        if ind not in current:
                            j = j + 1
                            current.append(ind)
                            times[ind] += 1
                    contain.append(current)
                for k in range(K):
                    idx_k = [p[0] for p in dpairs if p[1] == k]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, times[k])
                    ids = 0
                    for cid in range(self.num_clients):
                        if k in contain[cid]:
                            local_datas[cid].extend(split[ids].tolist())
                            ids += 1

        elif self.dist_id == 2:
            """label_skew_dirichlet"""
            min_size = 0
            dpairs = [
                [did, self.train_data[did][-1]] for did in range(len(self.train_data))
            ]
            local_datas = [[] for _ in range(self.num_clients)]
            while min_size < self.minvol:
                idx_batch = [[] for i in range(self.num_clients)]
                for k in range(self.num_classes):
                    idx_k = [p[0] for p in dpairs if p[1] == k]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(
                        np.repeat(self.skewness, self.num_clients)
                    )
                    ## Balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < len(self.train_data) / self.num_clients)
                            for p, idx_j in zip(proportions, idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                    ]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(self.num_clients):
                np.random.shuffle(idx_batch[j])
                local_datas[j].extend(idx_batch[j])

        elif self.dist_id == 3:
            """label_skew_shard"""
            dpairs = [
                [did, self.train_data[did][-1]] for did in range(len(self.train_data))
            ]
            self.skewness = min(max(0, self.skewness), 1.0)
            num_shards = max(int((1 - self.skewness) * self.num_classes * 2), 1)
            client_datasize = int(len(self.train_data) / self.num_clients)
            all_idxs = [i for i in range(len(self.train_data))]
            z = zip([p[1] for p in dpairs], all_idxs)
            z = sorted(z)
            labels, all_idxs = zip(*z)
            shardsize = int(client_datasize / num_shards)
            idxs_shard = range(int(self.num_clients * num_shards))
            local_datas = [[] for i in range(self.num_clients)]
            for i in range(self.num_clients):
                rand_set = set(np.random.choice(idxs_shard, num_shards, replace=False))
                idxs_shard = list(set(idxs_shard) - rand_set)
                for rand in rand_set:
                    local_datas[i].extend(
                        all_idxs[rand * shardsize : (rand + 1) * shardsize]
                    )

        elif self.dist_id == 4:
            pass

        elif self.dist_id == 5:
            """feature_skew_id"""
            if not isinstance(self.train_data, TupleDataset):
                raise RuntimeError(
                    "Support for dist_id=5 only after setting the type of self.train_data is TupleDataset"
                )
            Xs, IDs, Ys = self.train_data.tolist()
            self.num_clients = len(set(IDs))
            local_datas = [[] for _ in range(self.num_clients)]
            for did in range(len(IDs)):
                local_datas[IDs[did]].append(did)

        elif self.dist_id == 6:
            minv = 0
            d_idxs = np.random.permutation(len(self.train_data))
            while minv < self.minvol:
                proportions = np.random.dirichlet(
                    np.repeat(self.skewness, self.num_clients)
                )
                proportions = proportions / proportions.sum()
                minv = np.min(proportions * len(self.train_data))
            proportions = (np.cumsum(proportions) * len(d_idxs)).astype(int)[:-1]
            local_datas = np.split(d_idxs, proportions)
        elif self.dist_id == 9:
            local_datas_seen, local_datas_unseen = self.divide_balance_data(
                number_class_per_client_train=self.number_class_per_client,
                number_class_per_client_test=7,
            )
            return local_datas_seen, local_datas_unseen

        return local_datas

    def divide_seen_unseen_data(
        self, number_class_per_client, total_classes, file_name_log
    ):
        config_data = {}
        config_class = {}  # Configuration of class distribution in clients
        config_division = {}  # Count of the classes for division
        for i in range(self.num_clients):
            config_class["f_{0:05d}".format(i)] = []
            for j in range(number_class_per_client):
                cls = (i + j) % total_classes
                if cls not in config_division:
                    config_division[cls] = 1
                    config_data[cls] = [0, []]

                else:
                    config_division[cls] += 1
                config_class["f_{0:05d}".format(i)].append(cls)
        dpairs = [
            [did, self.train_data[did][-1]] for did in range(len(self.train_data))
        ]
        train_targets = torch.tensor([p[1] for p in dpairs])

        for cls in config_division.keys():
            # ! indexes is index for this class in dataset
            indexes = torch.nonzero(train_targets == cls)
            num_datapoint = indexes.shape[0]
            indexes = indexes[torch.randperm(num_datapoint)]
            num_partition = num_datapoint // config_division[cls]
            for i_partition in range(config_division[cls]):
                if i_partition == config_division[cls] - 1:
                    config_data[cls][1].append(indexes[i_partition * num_partition :])
                else:
                    config_data[cls][1].append(
                        indexes[
                            i_partition
                            * num_partition : (i_partition + 1)
                            * num_partition
                        ]
                    )
        local_datas_in_def = [[] for i in range(self.num_clients)]
        # local_labels_in_def = [[] for i in range(self.num_clients)]
        index_client = 0
        for user in tqdm(config_class.keys()):
            user_data_indexes = torch.tensor([])
            for cls in config_class[user]:
                # !config_data[cls][0] auto là số 0 :| , [1] là các array chia cho data đó
                user_data_index = config_data[cls][1][config_data[cls][0]]
                len_user_data_index = len(user_data_index)
                # print(
                #     f"user: {user} cls: {cls} user_data_index: {len(user_data_index)}"
                # )
                with open(file_name_log, "a") as log_file:
                    log_file.write(
                        f"user: {user} waf cls: {cls} user_data_index: {len_user_data_index}\n"
                    )
                # for i_label in range(len_user_data_index):
                #     local_labels_in_def[index_client].append(cls)

                user_data_indexes = torch.cat((user_data_indexes, user_data_index))
                config_data[cls][0] += 1
                # print(len(user_data_indexes))
            indexs_of_user = [int(i[0]) for i in user_data_indexes.tolist()]
            local_datas_in_def[index_client] = indexs_of_user
            index_client += 1

        return local_datas_in_def

    def divide_balance_data(
        self, number_class_per_client_train, number_class_per_client_test, i_seed=0
    ):
        torch.manual_seed(i_seed)

        num_seen_class = int(self.num_classes * 70 / 100)
        num_unseen_class = self.num_classes - num_seen_class
        local_datas_seen = self.divide_seen_unseen_data(
            number_class_per_client_train, num_seen_class, "split_seen.log"
        )
        local_datas_unseen = self.divide_seen_unseen_data(
            number_class_per_client_test, num_unseen_class, "split_unseen.log"
        )
        return local_datas_seen, local_datas_unseen

    def local_holdout(self, local_datas, rate=0.8, shuffle=False):
        """split each local dataset into train data and valid data according the rate."""
        train_cidxs = []
        valid_cidxs = []
        for local_data in local_datas:
            if shuffle:
                np.random.shuffle(local_data)
            k = int(len(local_data) * rate)
            train_cidxs.append(local_data[:k])
            valid_cidxs.append(local_data[k:])
        return train_cidxs, valid_cidxs

    def save_info(self):
        info = {
            "benchmark": self.benchmark,  # name of the dataset
            "dist": self.dist_id,  # type of the partition way
            "skewness": self.skewness,  # hyper-parameter for controlling the degree of niid
            "num-clients": self.num_clients,  # numbers of all the clients
        }
        # save info.json
        with open(os.path.join(self.taskpath, "info.json"), "w") as outf:
            ujson.dump(info, outf)

    def convert_data_for_saving(self):
        """Convert self.train_data and self.test_data to list that can be stored as .json file and the converted dataset={'x':[], 'y':[]}"""
        pass

    def XYData_to_json(self, train_cidxs, valid_cidxs):
        self.convert_data_for_saving()
        breakpoint()
        # save federated dataset
        feddata = {"store": "XY", "client_names": self.cnames, "dtest": self.test_data}
        for cid in range(self.num_clients):
            feddata[self.cnames[cid]] = {
                "dtrain": {
                    "x": [self.train_data["x"][did] for did in train_cidxs[cid]],
                    "y": [self.train_data["y"][did] for did in train_cidxs[cid]],
                },
                "dvalid": {
                    "x": [self.train_data["x"][did] for did in valid_cidxs[cid]],
                    "y": [self.train_data["y"][did] for did in valid_cidxs[cid]],
                },
            }
        with open(os.path.join(self.taskpath, "data.json"), "w") as outf:
            ujson.dump(feddata, outf)
        return

    def XYData_to_json_fewshot_seen_unseen(self, train_cidxs, valid_cidxs):
        self.convert_data_for_saving()
        # save federated dataset
        feddata = {"store": "XY", "client_names": self.cnames, "dtest": self.test_data}
        for cid in range(self.num_clients):
            ids_dataset_train = train_cidxs[cid]
            ids_dataset_test = valid_cidxs[cid]
            image_datas_training = [
                self.train_data["x"][did] for did in ids_dataset_train
            ]
            label_datas_training = [
                self.train_data["y"][did] for did in ids_dataset_train
            ]
            image_datas_testing = [
                self.train_data["x"][did] for did in ids_dataset_test
            ]
            label_datas_testing = [
                self.train_data["y"][did] for did in ids_dataset_test
            ]

            feddata[self.cnames[cid]] = {
                "dtrain": {
                    "x": image_datas_training,
                    "y": label_datas_training,
                },
                "dvalid": {
                    "x": image_datas_testing,
                    "y": label_datas_testing,
                },
            }
        with open(os.path.join(self.taskpath, "data.json"), "w") as outf:
            ujson.dump(feddata, outf)
        return

    def IDXData_to_json(self, train_cidxs, valid_cidxs):
        if self.datasrc == None:
            raise RuntimeError(
                "Attr datasrc not Found. Please define it in __init__() before calling IndexData_to_json"
            )
        feddata = {
            "store": "IDX",
            "client_names": self.cnames,
            "dtest": [i for i in range(len(self.test_data))],
            "datasrc": self.datasrc,
        }
        for cid in range(self.num_clients):
            feddata[self.cnames[cid]] = {
                "dtrain": train_cidxs[cid],
                "dvalid": valid_cidxs[cid],
            }
        with open(os.path.join(self.taskpath, "data.json"), "w") as outf:
            ujson.dump(feddata, outf)
        return


class BasicTaskCalculator:

    _OPTIM = None

    def __init__(self, device):
        self.device = device
        self.lossfunc = None
        self.DataLoader = None

    def data_to_device(self, data):
        raise NotImplementedError

    def get_loss(self):
        raise NotImplementedError

    def get_evaluation(self):
        raise NotImplementedError

    def get_data_loader(self, data, batch_size=64):
        return NotImplementedError

    def test(self):
        raise NotImplementedError

    def get_optimizer(self, name="sgd", model=None, lr=0.1, weight_decay=0, momentum=0):
        # if self._OPTIM == None:
        #     raise RuntimeError("TaskCalculator._OPTIM Not Initialized.")
        if name.lower() == "sgd":
            return torch.optim.SGD(
                model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
            )
        elif name.lower() == "adam":
            return torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                weight_decay=weight_decay,
                amsgrad=True,
            )
        else:
            raise RuntimeError("Invalid Optimizer.")

    @classmethod
    def setOP(cls, OP):
        cls._OPTIM = OP


class ClassifyCalculator(BasicTaskCalculator):
    def __init__(self, device):
        super(ClassifyCalculator, self).__init__(device)
        self.lossfunc = torch.nn.CrossEntropyLoss()
        self.DataLoader = DataLoader

    def get_loss(self, model, data, device=None):
        tdata = self.data_to_device(data, device)
        outputs = model(tdata[0])
        loss = self.lossfunc(outputs, tdata[1])
        return loss

    @torch.no_grad()
    def get_evaluation(self, model, data):
        tdata = self.data_to_device(data)
        outputs = model(tdata)
        y_pred = outputs.data.max(1, keepdim=True)[1]
        correct = y_pred.eq(tdata[1].data.view_as(y_pred)).long().cpu().sum()
        return (1.0 * correct / len(tdata[1])).item()

    @torch.no_grad()
    def test(self, model, data, device=None):
        """Metric = Accuracy"""
        tdata = self.data_to_device(data, device)
        model = model.to(device)
        outputs = model(tdata[0])
        loss = self.lossfunc(outputs, tdata[-1])
        y_pred = outputs.data.max(1, keepdim=True)[1]
        correct = y_pred.eq(tdata[1].data.view_as(y_pred)).long().cpu().sum()
        return (1.0 * correct / len(tdata[1])).item(), loss.item()

    def data_to_device(self, data, device=None):
        if device is None:
            return data[0].to(self.device), data[1].to(self.device)
        else:
            return data[0].to(device), data[1].to(device)

    def get_data_loader(self, dataset, batch_size=64, shuffle=True, droplast=False):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=droplast
        )


class BasicTaskReader:
    def __init__(self, taskpath=""):
        self.taskpath = taskpath

    def read_data(self):
        """
        Reading the spilted dataset from disk files and loading data into the class 'LocalDataset'.
        This algorithm should read three types of data from the processed task:
            train_sets = [client1_train_data, ...] where each item is an instance of 'LocalDataset'
            valid_sets = [client1_valid_data, ...] where each item is an instance of 'LocalDataset'
            test_set = test_dataset
        Return train_sets, valid_sets, test_set, client_names
        """
        pass


class XYTaskReader(BasicTaskReader):
    def __init__(self, taskpath=""):
        super(XYTaskReader, self).__init__(taskpath)

    def read_data(self):
        with open(os.path.join(self.taskpath, "data.json"), "r") as inf:
            feddata = ujson.load(inf)
        test_data = XYDatasetFewShot(feddata["dtest"]["x"], feddata["dtest"]["y"])
        train_datas = [
            XYDatasetFewShot(feddata[name]["dtrain"]["x"], feddata[name]["dtrain"]["y"])
            for name in feddata["client_names"]
        ]
        valid_datas = [
            XYDatasetFewShot(feddata[name]["dvalid"]["x"], feddata[name]["dvalid"]["y"])
            for name in feddata["client_names"]
        ]
        return train_datas, valid_datas, test_data, feddata["client_names"]


class IDXTaskReader(BasicTaskReader):
    def __init__(self, taskpath=""):
        super(IDXTaskReader, self).__init__(taskpath)

    def read_data(self):
        with open(os.path.join(self.taskpath, "data.json"), "r") as inf:
            feddata = ujson.load(inf)
        DS = getattr(
            importlib.import_module(feddata["datasrc"]["lib"]),
            feddata["datasrc"]["class_name"],
        )
        arg_strings = "(" + ",".join(feddata["datasrc"]["args"])
        train_args = arg_strings + ", train=True)"
        test_args = arg_strings + ", train=False)"
        DS.SET_DATA(eval(feddata["datasrc"]["class_name"] + train_args))
        DS.SET_DATA(eval(feddata["datasrc"]["class_name"] + test_args), key="TEST")
        test_data = IDXDataset(feddata["dtest"], key="TEST")
        train_datas = [
            IDXDataset(feddata[name]["dtrain"]) for name in feddata["client_names"]
        ]
        valid_datas = [
            IDXDataset(feddata[name]["dvalid"]) for name in feddata["client_names"]
        ]
        return train_datas, valid_datas, test_data, feddata["client_names"]


class XYDataset(Dataset):
    def __init__(self, X=[], Y=[], totensor=True):
        """Init Dataset with pairs of features and labels/annotations.
        XYDataset transforms data that is list\array into tensor.
        The data is already loaded into memory before passing into XYDataset.__init__()
        and thus is only suitable for benchmarks with small size (e.g. CIFAR10, MNIST)
        Args:
            X: a list of features
            Y: a list of labels with the same length of X
        """
        if not self._check_equal_length(X, Y):
            raise RuntimeError("Different length of Y with X.")
        if totensor:
            try:
                self.X = torch.tensor(X)
                self.Y = torch.tensor(Y)
            except:
                raise RuntimeError("Failed to convert input into torch.Tensor.")
        else:
            self.X = X
            self.Y = Y
        self.all_labels = list(set(self.tolist()[1]))

    def __len__(self):
        # return len(self.Y)
        DEFAULT_LEN_ESTIMATE = 1000000
        return DEFAULT_LEN_ESTIMATE
        # return len(self.Y)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def tolist(self):
        if not isinstance(self.X, torch.Tensor):
            return self.X, self.Y
        return self.X.tolist(), self.Y.tolist()

    def _check_equal_length(self, X, Y):
        return len(X) == len(Y)

    def get_all_labels(self):
        return self.all_labels


def get_random_n_values_from_array(array, n):
    # Make sure not to request more items than are available in the array
    n = min(n, len(array))
    return random.sample(array, n)


class XYDatasetFewShot(Dataset):
    def __init__(
        self,
        data_repo,
        sample_ids,
        sample_labels,
        num_way=5,
        num_shot=5,
        num_query=5,
        totensor=True,
    ):
        """Init Dataset with pairs of features and labels/annotations.
        XYDataset transforms data that is list\array into tensor.
        The data is already loaded into memory before passing into XYDataset.__init__()
        and thus is only suitable for benchmarks with small size (e.g. CIFAR10, MNIST)
        Args:
            X: a list of features
            Y: a list of labels with the same length of X
        """
        self.data_repo = data_repo
        self.all_labels = list(set(sample_labels))
        self.cls_samples = {}
        self.data_len = len(sample_ids)
        for cls_id in self.all_labels:
            self.cls_samples[cls_id] = [
                sample_ids[i]
                for i in range(len(sample_ids))
                if sample_labels[i] == cls_id
            ]
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_query = num_query

        

    def __len__(self):
        # return len(self.Y)
        DEFAULT_LEN_ESTIMATE = 1000000
        return DEFAULT_LEN_ESTIMATE
        # return len(self.Y)

    def get_entropy(self):
        dist = []
        for k in self.cls_samples:
            dist.append(len(self.cls_samples[k])/self.data_len)
        entropy = -sum([p*np.log(p) for p in dist])
        return entropy

    def get_cls_data_len(self):
        return {k: len(self.cls_samples[k]) for k in self.cls_samples}

    def get_indices_for_label(self, label, n=5):
        # Find all indices where the value in self.Y equals the label
        indices = [i for i, y in enumerate(self.Y) if y == label]
        # Get up to n random indices from this list
        return get_random_n_values_from_array(indices, n)

    def _get_X_Y_support_query(self, labels_sample):
        indices_sample = []
        for label in labels_sample:
            indices_for_label = self.get_indices_for_label(label)
            indices_sample.extend(indices_for_label)
        X_samples = [self.X[i] for i in indices_sample]
        Y_samples = [self.Y[i] for i in indices_sample]
        return X_samples, Y_samples

    def __getitem__(self, item):
        n_way = min(len(self.all_labels), self.num_way)
        sampled_labels = random.sample(self.all_labels, n_way)
        support_set = []
        support_labels = []
        query_set = []
        query_labels = []
        for it, cls_id in enumerate(sampled_labels):
            sampled_data_ids = random.sample(
                self.cls_samples[cls_id], self.num_shot + self.num_query
            )
            sampled_data = [self.data_repo[i][0] for i in sampled_data_ids]
            support_set += sampled_data[: self.num_shot]
            query_set += sampled_data[self.num_shot :]
            support_labels += [it] * self.num_shot
            query_labels += [it] * self.num_query
        # return self.X[item], self.Y[item]
        result = {
            "support_set": torch.stack(support_set),
            "query_set": torch.stack(query_set),
            "support_labels": torch.tensor(support_labels),
            "query_labels": torch.tensor(query_labels),
            "class_ids": torch.tensor(sampled_labels),
        }
        return result

    def tolist(self):
        if not isinstance(self.X, torch.Tensor):
            return self.X, self.Y
        return self.X.tolist(), self.Y.tolist()

    def _check_equal_length(self, X, Y):
        return len(X) == len(Y)

    def get_all_labels(self):
        return self.all_labels

    def get_unique_classes(self):
        return self.all_labels

    def get_samples_by_cls(self, cls_id):
        data = [self.data_repo[i][0] for i in self.cls_samples[cls_id]]
        data = torch.stack(data)
        return data


class IDXDataset(Dataset):
    # The source dataset that can be indexed by IDXDataset
    _DATA = {"TRAIN": None, "TEST": None}

    def __init__(self, idxs, key="TRAIN"):
        """Init dataset with 'src_data' and a list of indexes that are used to position data in 'src_data'"""
        if not isinstance(idxs, list):
            raise RuntimeError("Invalid Indexes")
        self.idxs = idxs
        self.key = key

    @classmethod
    def SET_DATA(cls, dataset, key="TRAIN"):
        cls._DATA[key] = dataset

    @classmethod
    def ADD_KEY_TO_DATA(cls, key, value=None):
        if key == None:
            raise RuntimeError(
                "Empty key when calling class algorithm IDXData.ADD_KEY_TO_DATA"
            )
        cls._DATA[key] = value

    def __getitem__(self, item):
        idx = self.idxs[item]
        return self._DATA[self.key][idx]


class TupleDataset(Dataset):
    def __init__(self, X1=[], X2=[], Y=[], totensor=True):
        if totensor:
            try:
                self.X1 = torch.tensor(X1)
                self.X2 = torch.tensor(X2)
                self.Y = torch.tensor(Y)
            except:
                raise RuntimeError("Failed to convert input into torch.Tensor.")
        else:
            self.X1 = X1
            self.X2 = X2
            self.Y = Y

    def __getitem__(self, item):
        return self.X1[item], self.X2[item], self.Y[item]

    def __len__(self):
        return len(self.Y)

    def tolist(self):
        if not isinstance(self.X1, torch.Tensor):
            return self.X1, self.X2, self.Y
        return self.X1.tolist(), self.X2.tolist(), self.Y.tolist()
