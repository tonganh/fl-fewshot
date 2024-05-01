from .fedbase import BasicServer, BasicClient
from utils import fmodule
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import LambdaLR
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

    def iterate(self, t):
        self.selected_clients = self.sample()
        # self.selected_clients = [0]

        data = self.communicate(self.selected_clients)

        if not self.selected_clients:
            return

        stats, clients_stats = self.aggregate(data, t)
        
        if (self.round_id + 1) % self.option['eval_interval'] == 0:
            global_loss, global_acc = self.eval()
            stats['global_loss'] = global_loss
            stats['global_acc'] = global_acc
        
        self.log(stats, clients_stats)

        self.round_id += 1

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
            proto_loss = 0
            for cls in local_protos.keys():
                if cls in global_protos:
                    proto_loss += self.option["prototype_loss_weight"] * F.mse_loss(
                        local_protos[cls], global_protos[cls]
                    )
                    cnt += 1
            if cnt > 0:
                loss += (proto_loss / cnt)
        return loss
    
    def eval(self):
        model = self.model
        model.eval()

        data_loader = self.test_loader
        loader_iter = iter(data_loader)

        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            total_num = 0
            for cur_iter in range(self.option["num_val_steps"]):
                try:
                    data = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(data_loader)
                    data = next(loader_iter)

                input = self.prepare_input(data)

                output = model(input)

                loss = self.compute_loss(output["logits"], input["query_labels"])

                total_loss += loss.item()
                total_correct += (
                    (output["logits"].argmax(1) == input["query_labels"]).sum().item()
                )
                total_num += len(input["query_labels"])

            return total_loss / total_num, total_correct / total_num
    
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
                    cls_protos[cls_id] += data[i]["cls_protos"][cls_id] * data[i]["cls_data_len"][cls_id] / cls_total_len[cls_id]

            for cls_id in cls_protos:
                self.cls_protos[cls_id] = cls_protos[cls_id]

        # update mean, min, max
        keys = ['train_loss', 'train_acc', 'test_loss', 'test_acc']
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
                mean, min_val, max_val = compute_basic_stats(vals)
                stats1[f'{k}_mean'] = mean
            # stats1[f'{k}_min'] = min_val
            # stats1[f'{k}_max'] = max_val
        
        self.latest_round_stats = stats1

        return self.latest_round_stats, self.latest_round_client_stats


    def pack(self, client_id):
        pkg = {"model": copy.deepcopy(self.model), "cls_protos": {}, 'round_id': self.round_id}
        if self.cls_protos :
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

        model, cls_protos, round_id = self.unpack(svr_pkg)

        self.train(model, cls_protos, round_id)

        cls_protos = {}
        if self.option["prototype_loss_weight"] > 0:
            cls_protos = self.compute_class_prototypes(model)

        train_loss, train_acc = self.eval(model, "train")

        return {
            "name": self.name,
            "model": model,
            "model_coef": self.model_coef,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "cls_protos": cls_protos,
            "cls_data_len": self.cls_data_len,
        }

    def unpack(self, received_pkg):
        return received_pkg["model"], received_pkg["cls_protos"], received_pkg['round_id']

    def prepare_input(self, input):
        for key in input.keys():
            if isinstance(input[key], torch.Tensor):
                input[key] = input[key].to(fmodule.device)
                input[key] = input[key].squeeze(0)
        input["query_labels"] = input["query_labels"].to(dtype=torch.long)
        return input

    def eval(self, model, mode="test"):
        model.eval()

        data_loader = self.test_loader if mode == "test" else self.train_loader
        loader_iter = iter(data_loader)

        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            total_num = 0
            for cur_iter in range(self.option["num_val_steps"]):
                try:
                    data = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(data_loader)
                    data = next(loader_iter)

                input = self.prepare_input(data)

                output = model(input)

                loss = self.compute_loss(output["logits"], input["query_labels"])

                total_loss += loss.item()
                total_correct += (
                    (output["logits"].argmax(1) == input["query_labels"]).sum().item()
                )
                total_num += len(input["query_labels"])

            return total_loss / total_num, total_correct / total_num
    
    def init_lr_scheduler(self, optimizer):
        if self.option['use_lrscheduler']:
            self.lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda lr: lr*0.95)
    
    def adjust_lr(self):
        if self.option['use_lrscheduler']:
            self.lr_scheduler.step()

    def train(self, model, global_protos, round_id):
        optimizer = self.calculator.get_optimizer(model=model, **self.optimizer_cfg)
        gradient_accumulation_steps = 4
        
        self.init_lr_scheduler(optimizer)

        model.train()
        optimizer.zero_grad()

        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)

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
            loss.backward()

            if ((cur_iter + 1) % gradient_accumulation_steps == 0) or (
                (cur_iter + 1) == self.iter_per_round
            ):
                optimizer.step()
                optimizer.zero_grad()
                self.adjust_lr()

    def compute_loss(self, logits, query_labels, local_protos=None, global_protos=None):
        loss = F.cross_entropy(logits, query_labels)
        if (
            self.option["prototype_loss_weight"] > 0
            and isinstance(local_protos, dict)
            and isinstance(global_protos, dict)
        ):
            cnt = 0
            proto_loss = 0
            for cls in local_protos.keys():
                if cls in global_protos:
                    proto_loss += self.option["prototype_loss_weight"] * F.mse_loss(
                        local_protos[cls], global_protos[cls]
                    )
                    cnt += 1
            if cnt > 0:
                loss += (proto_loss / cnt)
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
