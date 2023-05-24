# easyFL: A Lightning Framework for Federated Learning

This repository is PyTorch implementation for paper CADIS: Handling Cluster-skewed Non-IID Data in Federated Learning with Clustered Aggregation and Knowledge DIStilled Regularization which is accepted by CCGRID-23 Conference.

The repository is a modified version of easyFL, which was introduced by the authors in Federated Learning with Fair Averaging (IJCAI-21). EasyFL is a strong and reusable experimental platform for research on federated learning (FL) algorithm. It is easy for FL-researchers to quickly realize and compare popular centralized federated learning algorithms.

Our easyFL is a strong and reusable experimental platform for research on federated learning (FL) algorithm. It is easy for FL-researchers to quickly realize and compare popular centralized federated learning algorithms. 

## Table of Contents
- [Requirements](#Requirements)
- [QuickStart](#QuickStart)
- [Architecture](#Architecture)
- [Remark](#Remark)
- [Citation](#Citation)
- [Contacts](#Contacts)
- [References](#References)

## Requirements

The project is implemented using Python3 with dependencies below:

```
numpy>=1.17.2
pytorch>=1.3.1
torchvision>=0.4.2
cvxopt>=1.2.0
scipy>=1.3.1
matplotlib>=3.1.1
prettytable>=2.1.0
ujson>=4.0.2
```

## QuickStart

**First**, run the command below to get the splited dataset MNIST:

```sh
# generate the splited dataset
python generate_fedtask.py --dataset mnist --dist 0 --skew 0 --num_clients 100
```
**dist is from 0 to 6 (except 4), skew from 0 to 7**


**Second**, run the command below to quickly get a result of the basic algorithm FedAvg on MNIST with a simple CNN:

```sh
python main.py --task mnist_cnum100_dist0_skew0_seed0 --model cnn --algorithm fedavg --num_rounds 20 --num_epochs 5 --learning_rate 0.215 --proportion 0.1 --batch_size 10 --eval_interval 1
```

The result will be stored in ` ./fedtask/mnist_cnum100_dist0_skew0_seed0/record/`.

**Third**, run the command below to get a visualization of the result.

```sh
# change to the ./utils folder
cd ../utils
# visualize the results
python result_analysis.py
```

### Reproduced FL Algorithms
|Method|Reference|Publication|
|---|---|---|
|FedAvg|<a href='#refer-anchor-1'>[McMahan et al., 2017]</a>|AISTATS' 2017|
|FedProx|<a href='#refer-anchor-2'>[Li et al., 2020]</a>|MLSys' 2020|
|FedFV|<a href='#refer-anchor-3'>[Wang et al., 2021]</a>|IJCAI' 2021|
|qFFL|<a href='#refer-anchor-4'>[Li et al., 2019]</a>|ICLR' 2020|
|AFL|<a href='#refer-anchor-5'>[Mohri et al., 2019]</a>|ICML' 2019|
|FedMGDA+|<a href='#refer-anchor-6'>[Hu et al., 2020]</a>|pre-print|
|FedFA|<a href='#refer-anchor-7'>[Huang et al., 2020]</a>|pre-print|
|SCAFFOLD|<a href='#refer-anchor-11'>[Karimireddy et al., 2020]</a>|ICML' 2020|
| FedDyn      | <a href='#refer-anchor-12'>[Acar et al., 2021]</a>       | ICLR' 2021    |
| ...         |||

For those who want to realize their own federaed algorithms or reproduce others, please see `algorithms/readme.md`, where we take two simple examples to show how to use easyFL for the popurse.

### Options

Basic options:

* `task` is to choose the task of splited dataset. Options: name of fedtask (e.g. `mnist_client100_dist0_beta0_noise0`).

* `algorithm` is to choose the FL algorithm. Options: `fedfv`, `fedavg`, `fedprox`, …

* `model` should be the corresponding model of the dataset. Options: `mlp`, `cnn`, `resnet18.`

Server-side options:

* `sample` decides the way to sample clients in each round. Options: `uniform` means uniformly, `md` means choosing with probability.

* `aggregate` decides the way to aggregate clients' model. Options: `uniform`, `weighted_scale`, `weighted_com`

* `num_rounds` is the number of communication rounds.

* `proportion` is the proportion of clients to be selected in each round. 

* `lr_scheduler` is the global learning rate scheduler.

* `learning_rate_decay` is the decay rate of the global learning rate.

Client-side options:

* `num_epochs` is the number of local training epochs.

* `learning_rate ` is the step size when locally training.

* `batch_size ` is the size of one batch data during local training.

* `optimizer` is to choose the optimizer. Options: `SGD`, `Adam`.

* `momentum` is the ratio of the momentum item when the optimizer SGD taking each step. 

Other options:

* `seed ` is the initial random seed.

* `gpu ` is the id of the GPU device, `-1` for CPU.

* `eval_interval ` controls the interval between every two evaluations. 

* `net_drop` controls the dropout of clients after being selected in each communication round according to distribution Beta(net_drop,1). The larger this term is, the more possible for clients to drop.

* `net_active` controls the active rate of clients before being selected in each communication round according to distribution Beta(net_active,1). The larger this term is, the more possible for clients to be active.

* `num_threads` is the number of threads in the clients computing session that aims to accelarate the training process.

Additional hyper-parameters for particular federated algorithms:
* `mu` is the parameter for FedProx.
* `alpha` is the parameter for FedFV.
* `tau` is the parameter for FedFV.
* ...

Each additional parameter can be defined in `./utils/fflow.read_option`

## Architecture

We seperate the FL system into four parts: `benchmark`, `fedtask`, `method` and `utils`.
```
├─ benchmark
│  ├─ mnist							//mnist dataset
│  │  ├─ data							//data
│  │  ├─ model                   //the corresponding model
│  |  └─ core.py                 //the core supporting for the dataset, and each contains three necessary classes(e.g. TaskGen, TaskReader, TaskCalculator)							
│  ├─ ...
│  └─ toolkits.py						//the basic tools for generating federated dataset
├─ fedtask
│  ├─ mnist_client100_dist0_beta0_noise0//IID(beta=0) MNIST for 100 clients with not predefined noise
│  │  ├─ record							//record of result
│  │  ├─ info.json						//basic infomation of the task
│  |  └─ data.json						//the splitted federated dataset (fedtask)
|  └─ ...
├─ method
│  ├─ fedavg.py							//FL algorithm implementation inherit fedbase.py
│  ├─ fedbase.py						//FL algorithm superclass(i.e.,fedavg)
│  ├─ fedfv.py							
│  ├─ fedprox.py
|  └─ ...
├─ utils
│  ├─ fflow.py							//option to read, initialize,...
│  ├─ fmodule.py						//model-level operators
│  └─ result_analysis.py				        //to generate the visualization of record
├─ generate_fedtask.py					        //generate fedtask
├─ requirements.txt
└─ main.py
```
### Benchmark

This module is to generate `fedtask` by partitioning the particular distribution data through `generate_fedtask.py`. To generate different `fedtask`, there are three parameters: `dist`, `num_clients `, `beta`. `dist` denotes the distribution type (e.g. `0` denotes iid and balanced distribution, `1` denotes niid-label-quantity and balanced distribution). `num_clients` is the number of clients participate in FL system, and `beta` controls the degree of non-iid for different  `dist`. Each dataset can correspond to differrent models (mlp, cnn, resnet18, …). We refer to <a href='#refer-anchor-1'>[McMahan et al., 2017]</a>, <a href='#refer-anchor-2'>[Li et al., 2020]</a>, <a href='#refer-anchor-8'>[Li et al., 2021]</a>, <a href='#refer-anchor-4'>[Li et al., 2019]</a>, <a href='#refer-anchor-9'>[Caldas et al., 2018]</a>, <a href='#refer-anchor-10'>[He et al., 2020]</a> when realizing this module. Further details are described in `benchmark/README.md`.


### Algorithm
![image](https://github.com/WwZzz/myfigs/blob/master/fig0.png)
This module is the specific federated learning algorithm implementation. Each method contains two classes: the `Server` and the `Client`. 


#### Server

The whole FL system starts with the `main.py`, which runs `server.run()` after initialization. Then the server repeat the method `iterate()` for `num_rounds` times, which simulates the communication process in FL. In the `iterate()`, the `BaseServer` start with sampling clients by `select()`, and then exchanges model parameters with them by `communicate()`, and finally aggregate the different models into a new one with  `aggregate()`. Therefore, anyone who wants to customize its own method that specifies some operations on the server-side should rewrite the method `iterate()` and particular methods mentioned above.

#### Client

The clients reponse to the server after the server `communicate_with()` them, who first `unpack()` the received package and then train the model with their local dataset by `train()`. After training the model, the clients `pack()` send package (e.g. parameters, loss, gradient,... ) to the server through `reply()`.     

Further details of this module are described in `algorithm/README.md`.

### Utils
Utils is composed of commonly used operations: model-level operation (we convert model layers and parameters to dictionary type and apply it in the whole FL system), the flow controlling of the framework in and the supporting visualization templates to the result. To visualize the results, please run `./utils/result_analysis.py`. Further details are described in `utils/README.md`.

## Remark

Since we've made great changes on the latest version, to fully reproduce the reported results in our paper [Federated Learning with Fair Averaging](https://fanxlxmu.github.io/publication/ijcai2021/), please use another branch `easyFL v1.0` of this project.

## Citation

Please cite our paper in your publications if this code helps your research.

```
@article{hung2023cadis,
  title={CADIS: Handling Cluster-skewed Non-IID Data in Federated Learning with Clustered Aggregation and Knowledge DIStilled Regularization},
  author={Nang Hung Nguyen, Duc Long Nguyen, Trong Bang Nguyen, Thanh-Hung Nguyen, Hieu Pham, Truong Thao Nguyen and Phi Le Nguyenn},
  booktitle={The 23rd International Symposium on Cluster, Cloud and Internet Computing},
  year={2023}
}
```
