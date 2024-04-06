from .fedbase import BasicServer, BasicClient
from utils import fmodule
import copy
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch


class Server(BasicServer):
    def __init__(self, option, model, clients):
        super(Server, self).__init__(option, model, clients)
        self.cls_protos = {}

    def iterate(self, t):
        self.selected_clients = self.sample()
        # self.selected_clients = [0]

        data = self.communicate(self.selected_clients)

        if not self.selected_clients:
            return

        self.aggregate(data)

    def aggregate(self, data):
        # aggregate model
        models = [data[i]["model"] for i in range(len(data))]
        p = [1 / len(data) for i in range(len(data))]
        model = fmodule._model_average(models, p=p)
        self.model = model

        # aggregate
        cls_protos = {}
        num_clients = {}
        for i in range(len(data)):
            local_protos = data[i]["cls_protos"]
            if local_protos is not None:
                for cls_id in local_protos:
                    if cls_id not in cls_protos:
                        cls_protos[cls_id] = 0
                        num_clients[cls_id] = 0
                    cls_protos[cls_id] += local_protos[cls_id]
                    num_clients[cls_id] += 1

        for cls_id in cls_protos:
            cls_protos[cls_id] /= num_clients[cls_id]
            self.cls_protos[cls_id] = cls_protos[cls_id]

    def pack(self, client_id):
        pkg = {"model": copy.deepcopy(self.model), "cls_protos": None}
        if self.cls_protos is not None:
            pkg["cls_protos"] = copy.deepcopy(self.cls_protos)
        return pkg

    def unpack(self, received_pkgs):
        return received_pkgs


class Client(BasicClient):
    def __init__(self, option, name, train_data, valid_data):
        super(Client, self).__init__(option, name, train_data, valid_data)

        self.option = option

        self.train_data = train_data
        self.test_data = valid_data

        self.train_loader = DataLoader(
            self.train_data, batch_size=1, num_workers=self.option["num_loader_workers"]
        )
        self.test_loader = DataLoader(
            self.test_data, batch_size=1, num_workers=self.option["num_loader_workers"]
        )
        self.train_loader_iter = None
        self.test_loader_iter = None

        self.optimizer_cfg = {
            "name": option["optimizer"],
            "lr": option["learning_rate"],
            "weight_decay": option["weight_decay"],
            "momentum": option["momentum"],
        }
        self.iter_per_round = option["num_train_steps"]

    def reply(self, svr_pkg):

        model, cls_protos = self.unpack(svr_pkg)

        self.train(model, cls_protos)

        cls_protos = {}
        if self.option["prototype_loss_weight"] > 0:
            cls_protos = self.compute_class_prototypes(model)

        train_loss, train_acc = self.eval(model, "train")
        test_loss, test_acc = self.eval(model, "test")

        return {
            "model": model,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "cls_protos": cls_protos,
        }

    def unpack(self, received_pkg):
        return received_pkg["model"], received_pkg["cls_protos"]

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

    def train(self, model, global_protos):
        optimizer = self.calculator.get_optimizer(model=model, **self.optimizer_cfg)
        gradient_accumulation_steps = 4
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

    def compute_loss(self, logits, query_labels, local_protos=None, global_protos=None):
        loss = F.cross_entropy(logits, query_labels)
        if (
            self.option["prototype_loss_weight"] > 0
            and isinstance(local_protos, torch.Tensor)
            and isinstance(global_protos, torch.Tensor)
        ):
            for cls in local_protos.keys():
                if cls in global_protos:
                    loss += self.option["prototype_loss_weight"] * F.mse_loss(
                        local_protos[cls], global_protos[cls]
                    )
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
