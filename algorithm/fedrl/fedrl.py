from algorithm.fedbase import BasicServer, BasicClient
from fedrl.fedrl_utils.ddpg_agent.ddpg import DDPG_Agent
import torch


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super(Server, self).__init__(option, model, clients, test_data)
        K = self.clients_per_round
        self.ddpg_agent = DDPG_Agent(state_dim= K * 3, action_dim= K * 3, hidden_dim=256)

    def unpack(self, packages_received_from_clients):
        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        n_samples = [cp["n_sample"] for cp in packages_received_from_clients]
        n_epochs = [cp["n_epochs"] for cp in packages_received_from_clients]
        return models, train_losses, n_samples, n_epochs

    def iterate(self, t):
        self.selected_clients = self.sample()
        models, train_losses, n_samples, n_epochs = self.communicate(self.selected_clients)
        if not self.selected_clients:
            return

        observation = {"losses": train_losses, "n_samples": n_samples, "n_epochs": n_epochs}
        priority = self.ddpg_agent.get_action(observation, time_step=t).tolist()
        self.model = self.aggregate(models, p=priority)
        return
    
    def run(self):
        super().run()
        self.ddpg_agent.dump_buffer()
        return


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)

    def pack(self, model, loss, n_sample, n_epochs):
        return {
            "model" : model,
            "train_loss": loss,
            "n_sample" : n_sample,
            "n_epochs" : n_epochs
        }

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        loss = self.train_loss(model)
        self.train(model)
        cpkg = self.pack(model, loss, self.datavol, self.epochs)
        return cpkg
