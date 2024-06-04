import os
from utils.fflow import get_current_time_str
from .fedbase import BasicServer, BasicClient
from utils import fmodule
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import LambdaLR
import json

def compute_basic_stats(values):
    return torch.mean(values).item(), torch.min(values).item(), torch.max(values).item()


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data):
        super(Server, self).__init__(option, model, clients)
        self.cls_protos = {}
        self.round_id = 0

        self.test_data = test_data
        self.test_loader = DataLoader(
            self.test_data, batch_size=1, num_workers=self.option["num_loader_workers"]
        )

        if option['local_log']:
            current_time = get_current_time_str()
            self.log_dir = os.path.join(option['log_dir'], current_time)
            os.makedirs(self.log_dir, exist_ok=True)
            self.last_checkpoint = None
        
        self.log_checkpoint = option['log_checkpoint'] and option['local_log']
        self.log_confusion_matrix = option['log_confusion_matrix'] and option['local_log']
        self.cf_mat_rounds = [0, option['num_rounds']/2, option['num_rounds']-1]
        
        if self.log_confusion_matrix:
            self.preds_save_path = os.path.join(self.log_dir, 'predictions.jsonl')
            if "client_ids" not in option:
                self.option["client_ids"] = range(len(clients))
    
    def iterate(self, t):
        self.selected_clients = self.sample()
        # self.selected_clients = [0]

        data = self.communicate(self.selected_clients)

        if not self.selected_clients:
            return

        stats, clients_stats = self.aggregate(data, t)

        if (self.round_id + 1) % self.option['eval_interval'] == 0:
            # global_loss, global_acc = self.eval()
            res = self.eval()
            global_loss = res['test_loss']
            global_acc = res['test_acc']

            stats['global_loss'] = global_loss
            stats['global_acc'] = global_acc

        # checkpoint
        if self.log_checkpoint and ((self.round_id + 1) % self.option['eval_interval'] == 0 or self.round_id == self.option['num_rounds'] - 1):
            self.checkpoint(self.round_id)

        # confusion matrix
        if self.log_confusion_matrix and self.round_id in self.cf_mat_rounds:
            self.compute_confusion_matrix()

        # wandb logging
        self.log(stats, clients_stats)

        self.round_id += 1
    
    def compute_confusion_matrix(self):
        log = []        
        res = self.eval(num_val_steps=self.option['num_val_steps_1'])
        log.append({"client_id": -1, "round_id": self.round_id , "preds": res['preds']})
        for c in self.option['client_ids']:
            res = self.clients[c].eval(self.test_loader, self.option['num_val_steps_1'])
            if res != None:
                log.append({"client_id": c, "round_id": self.round_id , "preds": res['preds'], "data": "global"})
                res = self.clients[c].eval(num_val_steps = self.option['num_val_steps_1'])
                log.append({"client_id": c, "round_id": self.round_id , "preds": res['preds'], "data": "local"})
        
        with open(self.preds_save_path, 'a') as f:
            for line in log:
                json_data = json.dumps(line)
                f.write(json_data + "\n")

    
    def checkpoint(self, round_id):
        ckpt = {
            "global_model": self.model,
            "round_id": round_id,
            "cls_protos": self.cls_protos,
            "option": self.option,
        }
        if self.last_checkpoint is not None:
            os.remove(self.last_checkpoint)
        self.last_checkpoint = os.path.join(self.log_dir, f'ckpt_{round_id}.pth')
        torch.save(ckpt, self.last_checkpoint)

    def log(self, aggregate_stats, clients_stats):
        wandb_logger = fmodule.wandb_logger
        if wandb_logger is not None:
            wandb_logger.log(aggregate_stats)

        local_logger = getattr(fmodule, "local_logger", None)
        if local_logger is not None:
            local_logger.log(clients_stats)

    def compute_loss(self, logits, query_labels, local_protos=None, global_protos=None):
        loss = F.cross_entropy(logits, query_labels)
        if (
            self.option["prototype_loss_weight"] > 0
            and isinstance(local_protos, dict)
            and isinstance(global_protos, dict)
        ):
            cnt = 0
            loss_proto = 0
            for cls in local_protos.keys():
                if cls in global_protos:
                    loss_proto = loss_proto + (self.option["prototype_loss_weight"] * F.mse_loss(
                        local_protos[cls], global_protos[cls]
                    ))
                    cnt += 1
            if cnt > 0 and loss_proto > 0:
                loss_proto = loss_proto / cnt
                loss = loss + loss_proto

        return loss

    def eval(self, num_val_steps=None):
        model = self.model
        if model == None:
            return None
        
        model.eval()

        data_loader = self.test_loader
        loader_iter = iter(data_loader)

        res = {"preds": []}
        if num_val_steps is not None:
            num_val_steps = self.option["num_val_steps"]
        
        with torch.no_grad():
            loss_total = 0
            total_correct = 0
            total_num = 0
            for _ in range(num_val_steps):
                try:
                    data = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(data_loader)
                    data = next(loader_iter)

                input = self.prepare_input(data)

                output = model(input)

                loss = self.compute_loss(
                    output["logits"], input["query_labels"])

                loss_total = loss_total + loss.item()

                total_correct += (
                    (output["logits"].argmax(1) ==
                     input["query_labels"]).sum().item()
                )
                total_num += len(input["query_labels"])

                pred = output["logits"].argmax(1)
                pred = [input['class_ids'][i].item() for i in pred]
                gt = [input['class_ids'][i].item() for i in input["query_labels"]]
                res['preds'] += list(zip(pred, gt))

            res['test_loss'] = loss_total / total_num
            res['test_acc'] = total_correct / total_num

            return res

    def prepare_input(self, input):
        for key in input.keys():
            if isinstance(input[key], torch.Tensor):
                input[key] = input[key].to(fmodule.device)
                input[key] = input[key].squeeze(0)
        input["query_labels"] = input["query_labels"].to(dtype=torch.long)
        return input

    def aggregate(self, data, round_id):
        # aggregate model
        # if (round_id + 1) % self.option['num_round_per_aggregation'] == 0:
        models = [data[i]["model"] for i in range(len(data))]
        norm_val = sum([data[i]["model_coef"] for i in range(len(data))])
        p = [data[i]['model_coef'] / norm_val for i in range(len(data))]
        model = fmodule._model_average(models, p=p)
        self.model = model

        # aggregate class prototypes
        if self.option["prototype_loss_weight"] > 0:
            cls_total_len = {}
            for i in range(len(data)):
                for cls_id in data[i]["cls_data_len"]:
                    if cls_id not in cls_total_len:
                        cls_total_len[cls_id] = 0
                    cls_total_len[cls_id] += data[i]["cls_data_len"][cls_id]

            cls_protos = {}
            for i in range(len(data)):
                for cls_id in data[i]["cls_protos"]:
                    if cls_id not in cls_protos:
                        cls_protos[cls_id] = 0
                    cls_protos[cls_id] += data[i]["cls_protos"][cls_id] * \
                        data[i]["cls_data_len"][cls_id] / cls_total_len[cls_id]

            for cls_id in cls_protos:
                self.cls_protos[cls_id] = cls_protos[cls_id]

        # update mean, min, max
        # keys = ['train_loss', 'train_acc', 'test_loss', 'test_acc']
        keys = ['client_loss_total', 'client_loss_ce']
        if self.option["prototype_loss_weight"] > 0:
            keys.append('client_loss_proto')
        stats = {k: [] for k in keys}
        client_stats = []
        for client_res in data:
            client_res1 = {
                "name": client_res["name"],
            }
            for k in keys:
                if k in client_res:
                    stats[k].append(client_res[k])
                    client_res1[k] = client_res[k]
            client_stats.append(client_res1)

        self.latest_round_client_stats = client_stats

        stats1 = {}
        for k in keys:
            if len(stats[k]) > 0:
                vals = torch.tensor(stats[k])
                mean, _, _ = compute_basic_stats(vals)
                stats1[f'{k}_mean'] = mean
        
        if 'client_loss_proto_mean' in stats1:
            print(stats1['client_loss_total_mean'], stats1['client_loss_ce_mean'], stats1['client_loss_proto_mean'])
            # assert (stats1['client_loss_total_mean'] - stats1['client_loss_ce_mean'] - stats1['client_loss_proto_mean']) < 1e-4

        self.latest_round_stats = stats1

        return self.latest_round_stats, self.latest_round_client_stats

    def pack(self, client_id):
        pkg = {"model": copy.deepcopy(self.model), "cls_protos": {
        }, 'round_id': self.round_id}
        if self.cls_protos:
            pkg["cls_protos"] = copy.deepcopy(self.cls_protos)

        return pkg

    def unpack(self, received_pkgs):
        return received_pkgs


class Client(BasicClient):
    def __init__(self, option, name, train_data):
        super(Client, self).__init__(option, name, train_data)

        self.option = option

        self.train_data = train_data

        self.train_loader = DataLoader(
            self.train_data, batch_size=1, num_workers=self.option["num_loader_workers"]
        )
        self.train_loader_iter = None

        self.optimizer_cfg = {
            "name": option["optimizer"],
            "lr": option["learning_rate"],
            "weight_decay": option["weight_decay"],
            "momentum": option["momentum"],
        }
        self.iter_per_round = option["num_train_steps"]

        self.model_coef = 1
        if option['client_model_aggregation'] == 'entropy':
            self.model_coef = self.train_data.get_entropy()

        self.cls_data_len = self.train_data.get_cls_data_len()

    def reply(self, svr_pkg):

        model, global_protos, round_id = self.unpack(svr_pkg)

        res = self.train(model, global_protos, round_id)
        self.model = model
        
        local_protos = {}
        if self.option["prototype_loss_weight"] > 0:
            local_protos = self.compute_class_prototypes(model)

        # breakpoint()
        res1 = {
            "name": self.name,
            "model": model,
            "model_coef": self.model_coef,
            "client_loss_total": res['loss_total'],
            'client_loss_ce': res['loss_ce'],
            "cls_protos": local_protos,
            "cls_data_len": self.cls_data_len,
        }
        if 'loss_proto' in res:
            res1['client_loss_proto'] = res['loss_proto']
        
        return res1

    def unpack(self, received_pkg):
        return received_pkg["model"], received_pkg["cls_protos"], received_pkg['round_id']

    def prepare_input(self, input):
        for key in input.keys():
            if isinstance(input[key], torch.Tensor):
                input[key] = input[key].to(fmodule.device)
                input[key] = input[key].squeeze(0)
        input["query_labels"] = input["query_labels"].to(dtype=torch.long)
        return input

    def eval(self, data_loader=None, num_val_steps=None):
        model = self.model
        if model == None:
            return None
        model.eval()

        if data_loader == None:
            data_loader = self.train_loader
        loader_iter = iter(data_loader)

        res = {"preds": []}
        if num_val_steps is None:
            num_val_steps = self.option["num_val_steps"]
        
        with torch.no_grad():
            loss_total = 0
            total_correct = 0
            total_num = 0
            for _ in range(num_val_steps):
                try:
                    data = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(data_loader)
                    data = next(loader_iter)

                input = self.prepare_input(data)

                output = model(input)

                loss = self.compute_loss(
                    output["logits"], input["query_labels"])

                loss_total = loss_total + loss['loss_total'].item()

                total_correct += (
                    (output["logits"].argmax(1) ==
                     input["query_labels"]).sum().item()
                )
                total_num += len(input["query_labels"])

                pred = output["logits"].argmax(1)
                pred = [input['class_ids'][i].item() for i in pred]
                gt = [input['class_ids'][i].item() for i in input["query_labels"]]
                res['preds'] += list(zip(pred, gt))

            res['test_loss'] = loss_total / total_num
            res['test_acc'] = total_correct / total_num

            return res

    def init_lr_scheduler(self, optimizer):
        if self.option['use_lrscheduler']:
            self.lr_scheduler = LambdaLR(
                optimizer, lr_lambda=lambda lr: lr*0.95)

    def adjust_lr(self):
        if self.option['use_lrscheduler']:
            self.lr_scheduler.step()

    def train(self, model, global_protos, round_id):
        # optimizer = self.calculator.get_optimizer(
        #     model=model, **self.optimizer_cfg)
        # self.init_lr_scheduler(optimizer)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.option['learning_rate'])
        gradient_accumulation_steps = 4

        model.train()
        optimizer.zero_grad()

        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        res = {'loss_total': [], 'loss_ce': []}
        if self.option["prototype_loss_weight"] > 0:
            res['loss_proto'] = []
        
        for cur_iter in range(self.option["num_train_steps"]):
            try:
                input = next(self.train_loader_iter)
            except StopIteration:
                self.train_loader_iter = iter(self.train_loader)
                input = next(self.train_loader_iter)

            input = self.prepare_input(input)

            output = model(input)

            loss = self.compute_loss(
                output["logits"],
                input["query_labels"],
                output["cls_protos"],
                global_protos,
            )
            loss['loss_total'].backward()

            res["loss_total"].append(loss['loss_total'].item())
            res["loss_ce"].append(loss['loss_ce'].item())
            if 'loss_proto' in loss:
                res['loss_proto'].append(loss['loss_proto'].item())

            if ((cur_iter + 1) % gradient_accumulation_steps == 0) or (
                (cur_iter + 1) == self.iter_per_round
            ):
                optimizer.step()
                optimizer.zero_grad()
                # self.adjust_lr()
        
        res['loss_total'] = (sum(res['loss_total']) /
                             len(res['loss_total']))
        res['loss_ce'] = (sum(res['loss_ce']) / len(res['loss_ce']))
        if 'loss_proto' in res:
            if len(res['loss_proto']) > 0:
                res['loss_proto'] = (
                    sum(res['loss_proto']) / len(res['loss_proto']))
            else:
                res.pop('loss_proto')

        return res

    def compute_loss(self, logits, query_labels, local_protos=None, global_protos=None):
        loss = {"loss_total": 0, "loss_ce": 0}
        loss['loss_total'] = F.cross_entropy(logits, query_labels)
        loss['loss_ce'] = loss['loss_total']
        if (
            self.option["prototype_loss_weight"] > 0
            and isinstance(local_protos, dict)
            and isinstance(global_protos, dict)
        ):
            cnt = 0
            loss_proto = 0
            for cls in local_protos.keys():
                if cls in global_protos:
                    loss_proto = loss_proto + self.option["prototype_loss_weight"] * F.mse_loss(
                        local_protos[cls], global_protos[cls]
                    )
                    cnt += 1
            if loss_proto > 0:
                loss["loss_proto"] = (loss_proto / cnt)
                loss["loss_total"] = loss['loss_total'] + loss["loss_proto"]
        return loss

    def compute_class_prototypes(self, model):
        model.eval()
        with torch.no_grad():
            cls_protos = {}
            for cls in self.train_data.get_unique_classes():
                input = self.train_data.get_samples_by_cls(cls)
                input = input.to(fmodule.device)
                output = model.encode(input)
                cls_protos[cls] = torch.mean(output, dim=0)

        return cls_protos
