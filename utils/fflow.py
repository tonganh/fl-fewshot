from datetime import timedelta, datetime
import numpy as np
import argparse
import random
import torch
import os.path
import importlib
import os
import utils.fmodule
import ujson
import time
import json

sample_list=['uniform', 'md', 'active']
agg_list=['uniform', 'weighted_scale', 'weighted_com', 'none']
optimizer_list=['SGD', 'Adam']

def read_option():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument('--task', help='name of fedtask;', type=str, default='mnist_cnum100_dist0_skew0_seed0')
    parser.add_argument('--algorithm', help='name of algorithm;', type=str, default='fedavg')
    parser.add_argument('--model', help='name of model;', type=str, default='cnn')
    parser.add_argument('--data_split', help='path to client data split', type=str, default=None)
    parser.add_argument('--root_data', help='path to folder contain torch vision cifar 100 dataset', type=str, default=None)
    parser.add_argument('--prototype_loss_weight', help='weights for prototype loss', type=float, default=0)
    parser.add_argument('--use_wandb_logging', help='use wandb logging', action='store_true', default=False)
    parser.add_argument('--log_dir', help='', type=str, default='logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--num_round_per_aggregation', help='number of communication rounds per aggregation', type=int, default=1)
    parser.add_argument('--use_lrscheduler', action='store_true', default=False)
    parser.add_argument('--client_model_aggregation', help='type of client model aggregation. Ex: uniform, entropy', type=str, default="uniform")
    
    # logging
    parser.add_argument("--local_log", action="store_true", default=False)
    parser.add_argument("--log_checkpoint", action="store_true", default=False)
    parser.add_argument("--log_confusion_matrix", action="store_true", default=False)
    
    # config file 
    parser.add_argument("--cfg", default=None)
    #
    parser.add_argument("--num_val_steps_1", type=int, default=200)
    # methods of server side for sampling and aggregating
    parser.add_argument('--sample', help='methods for sampling clients', type=str, choices=sample_list, default='uniform')
    parser.add_argument('--aggregate', help='methods for aggregating models', type=str, choices=agg_list, default='none')
    parser.add_argument('--learning_rate_decay', help='learning rate decay for the training process;', type=float, default=0.998)
    parser.add_argument('--weight_decay', help='weight decay for the training process', type=float, default=0)
    parser.add_argument('--lr_scheduler', help='type of the global learning rate scheduler', type=int, default=-1)
    # hyper-parameters of training in server side
    parser.add_argument('--num_rounds', help='number of communication rounds', type=int, default=20)
    parser.add_argument('--proportion', help='proportion of clients sampled per round', type=float, default=0.2)
    # hyper-parameters of local training
    parser.add_argument('--num_epochs', help='number of epochs when clients trainset on data;', type=int, default=5)
    parser.add_argument('--learning_rate', help='learning rate for inner solver;', type=float, default=0.1)
    parser.add_argument('--batch_size', help='batch size when clients trainset on data;', type=int, default=64)
    parser.add_argument('--optimizer', help='select the optimizer for gd', type=str, choices=optimizer_list, default='SGD')
    parser.add_argument('--momentum', help='momentum of local update', type=float, default=0)
    parser.add_argument('--num_train_steps', help='number of train step on client', type=int, default=200)
    parser.add_argument('--num_val_steps', help='number of val step on client', type=int, default=100)

    parser.add_argument('--num_loader_workers', help='number of worker for client data loader', type=int, default=1)


    # machine environment settings
    parser.add_argument('--seed', help='seed for random initialization;', type=int, default=0)
    parser.add_argument('--eval_interval', help='evaluate every __ rounds;', type=int, default=1)
    parser.add_argument('--num_threads', help='the number of threads;', type=int, default=1)
    parser.add_argument('--num_threads_per_gpu', help="the number of threads per gpu in the clients computing session;", type=int, default=1)
    parser.add_argument('--num_gpus', default=3, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    # the simulating system settings of clients
    
    # constructing the heterogeity of the network
    parser.add_argument('--net_drop', help="controlling the dropout of clients after being selected in each communication round according to distribution Beta(drop,1)", type=float, default=0)
    parser.add_argument('--net_active', help="controlling the probability of clients being active and obey distribution Beta(active,1)", type=float, default=99999)
    # constructing the heterogeity of computing capability
    parser.add_argument('--capability', help="controlling the difference of local computing capability of each client", type=float, default=0)

    # hyper-parameters of different algorithms
    parser.add_argument('--learning_rate_lambda', help='η for λ in afl', type=float, default=0)
    parser.add_argument('--q', help='q in q-fedavg', type=float, default='0.0')
    parser.add_argument('--epsilon', help='ε in fedmgda+', type=float, default='0.0')
    parser.add_argument('--eta', help='global learning rate in fedmgda+', type=float, default='1.0')
    parser.add_argument('--tau', help='the length of recent history gradients to be contained in FedFAvg', type=int, default=0)
    parser.add_argument('--alpha', help='proportion of clients keeping original direction in FedFV/alpha in fedFA', type=float, default='0.0')
    parser.add_argument('--beta', help='beta in FedFA',type=float, default='1.0')
    parser.add_argument('--gamma', help='gamma in FedFA', type=float, default='0')
    parser.add_argument('--mu', help='mu in fedprox', type=float, default='0.1')
    # server gpu
    parser.add_argument('--server_gpu_id', help='server process on this gpu', type=int, default=0)
    
    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))

    if option['cfg'] != None:
        import yaml
        with open(option['cfg'], 'r') as f:
            cfg = yaml.safe_load(option['cfg'])
        for k, v in cfg.items():
            if k not in option:
                option[k] = v
        
    return option

def setup_seed(seed):
    random.seed(1+seed)
    np.random.seed(21+seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(12+seed)
    torch.cuda.manual_seed_all(123+seed)

def init_wandb_logger(option):
    if option['use_wandb_logging'] and not option['debug']:
        import wandb
        run_name = 'baseline' if option['prototype_loss_weight'] == 0 else 'proto_w_{}'.format(option['prototype_loss_weight'])
        run_name += '_train{}_eval{}'.format(option['num_train_steps'], option['num_val_steps'])
        wandb.login(key="835e03fed77f3418a3bcf4cd09a93bf951d32b91")
        wandb.init(project="fed_fewshot", entity="aiotlab", name=run_name, config=option, reinit=True)
        return wandb
    else:
        return None

def get_current_time_str():
    utc_now = datetime.utcnow()

    # Calculate the Vietnam local time (ICT, UTC+7)
    vietnam_time = utc_now + timedelta(hours=7)

    return vietnam_time.strftime("%d-%m_%H-%M")

class Custom_Logger:
    def __init__(self, option):
        current_time = get_current_time_str()
        self.log_dir = os.path.join(option['log_dir'], current_time)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, 'log.jsonl')
    
    def log(self, data):
        self.write_jsonl(self.log_file, data)
        
    def write_jsonl(self, file_path, data, mode='a'):
        with open(file_path, mode, encoding='utf8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')


def initialize(option):
    # init fedtask
    print("init fedtask...", end='')
    # dynamical initializing the configuration with the benchmark
    bmk_name = option['task'][:option['task'].find('cnum')-1].lower()
    bmk_model_path = '.'.join(['benchmark', bmk_name, 'model', option['model']])
    bmk_core_path = '.'.join(['benchmark', bmk_name, 'core'])
    utils.fmodule.device = torch.device('cuda:{}'.format(option['server_gpu_id']) if torch.cuda.is_available() and option['server_gpu_id'] != -1 else 'cpu')
    utils.fmodule.TaskCalculator = getattr(importlib.import_module(bmk_core_path), 'TaskCalculator')
    utils.fmodule.TaskCalculator.setOP(getattr(importlib.import_module('torch.optim'), option['optimizer']))
    utils.fmodule.Model = getattr(importlib.import_module(bmk_model_path), 'Model')
    utils.fmodule.wandb_logger = init_wandb_logger(option)
    # utils.fmodule.local_logger = Custom_Logger(option) if not option['debug'] else None

    # task_reader = getattr(importlib.import_module(bmk_core_path), 'TaskReader')(taskpath=os.path.join('fedtask', option['task']))
    task_reader = getattr(importlib.import_module(bmk_core_path), 'TaskReader')

    if option['data_split'] != None:
        task_reader = task_reader(data_path=option['root_data'], data_split_path=option['data_split'])
    else:
        task_reader = task_reader(taskpath=os.path.join('fedtask', option['task']))
    client_train_data, _, global_test_data, client_names = task_reader.read_data()
    num_clients = len(client_names)
    print("done")
    # init client
    print('init clients...', end='')
    client_path = '%s.%s' % ('algorithm', option['algorithm'])
    Client=getattr(importlib.import_module(client_path), 'Client')
    clients = [Client(option, name = client_names[cid], train_data = client_train_data[cid]) for cid in range(num_clients)]
    print('done')

    # init server
    print("init server...", end='')
    server_path = '%s.%s' % ('algorithm', option['algorithm'])
    server = getattr(importlib.import_module(server_path), 'Server')(option, utils.fmodule.Model().to(utils.fmodule.device), clients, global_test_data)
    print('done')
    return server

def output_filename(option, server):
    header = "{}_".format(option["algorithm"])
    for para in server.paras_name: header = header + para + "{}_".format(option[para])
    output_name = header + "M{}_R{}_B{}_E{}_LR{:.4f}_P{:.2f}_S{}_LD{:.3f}_WD{:.3f}_DR{:.2f}_AC{:.2f}.json".format(
        option['model'],
        option['num_rounds'],
        option['batch_size'],
        option['num_epochs'],
        option['learning_rate'],
        option['proportion'],
        option['seed'],
        option['lr_scheduler']+option['learning_rate_decay'],
        option['weight_decay'],
        option['net_drop'],
        option['net_active'])
    return output_name

class Logger:
    def __init__(self):
        self.output = {}
        self.current_round = -1
        self.temp = "{:<30s}{:.4f}"
        self.time_costs = []
        self.time_buf={}

    def check_if_log(self, round, eval_interval=-1):
        """For evaluating every 'eval_interval' rounds, check whether to log at 'round'."""
        self.current_round = round
        return eval_interval > 0 and (round == 0 or round % eval_interval == 0)

    def time_start(self, key = ''):
        """Create a timestamp of the event 'key' starting"""
        if key not in [k for k in self.time_buf.keys()]:
            self.time_buf[key] = []
        self.time_buf[key].append(time.time())

    def time_end(self, key = ''):
        """Create a timestamp that ends the event 'key' and print the time interval of the event."""
        if key not in [k for k in self.time_buf.keys()]:
            raise RuntimeError("Timer end before start.")
        else:
            self.time_buf[key][-1] =  time.time() - self.time_buf[key][-1]
            print("{:<30s}{:.4f}".format(key+":", self.time_buf[key][-1]) + 's')

    def save(self, filepath):
        """Save the self.output as .json file"""
        if self.output=={}: return
        with open(filepath, 'w') as outf:
            ujson.dump(self.output, outf)
            
    def write(self, var_name=None, var_value=None):
        """Add variable 'var_name' and its value var_value to logger"""
        if var_name==None: raise RuntimeError("Missing the name of the variable to be logged.")
        if var_name in [key for key in self.output.keys()]:
            self.output[var_name] = []
        self.output[var_name].append(var_value)
        return

    def log(self, server=None):
        pass
