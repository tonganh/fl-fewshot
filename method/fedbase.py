import numpy as np
from utils import fmodule, fmodule
import copy
from torch.utils.data import DataLoader
from utils.fmodule import device,lossfunc,optim

class BaseServer():
    def __init__(self, option, model, clients, dtest = None):
        # basic setting
        self.task = option['task']
        self.name = option['method']
        self.model = model
        self.trans_model = copy.deepcopy(self.model).to(fmodule.device)
        self.dtest = fmodule.XYDataset(dtest['x'], dtest['y']) if dtest else None
        self.eval_interval = option['eval_interval']
        # clients settings
        self.clients = clients
        self.num_clients = len(self.clients)
        self.client_vols = [c.datavol for c in self.clients]
        self.data_vol = sum(self.client_vols)
        # hyper-parameters during training process
        self.num_rounds = option['num_rounds']
        self.decay_rate = option['learning_rate_decay']
        self.clients_per_round = max(int(self.num_clients * option['proportion']), 1)
        self.lr_scheduler_type = option['lr_scheduler']
        # sampling and aggregating methods
        self.sample_option = option['sample']
        self.agg_option = option['aggregate']
        self.lr=option['learning_rate']
        # names of additional parameters
        self.paras_name=[]
        self.option = option

    def run(self):
        accs = []
        c_accs = []
        local_train_losses = []
        train_losses = []
        test_losses = []
        std_c_accs = []
        mean_c_accs = []
        valid_accs = []
        test_accs = []
        for round in range(self.num_rounds+1):
            print("--------------Round {}--------------".format(round))
            # train
            local_train_loss = self.iterate(round)
            # decay learning rate
            self.global_lr_scheduler(round)
            local_train_losses.append(local_train_loss)
            if self.eval_interval>0 and (round==0 or round%self.eval_interval==0 or round== self.num_rounds):
                # train
                _, train_loss = self.test_on_clients(round, dataflag='train')
                # validate
                accs, _ = self.test_on_clients(round, dataflag='valid')
                # test
                test_acc, test_loss = self.test_on_dtest()
                # record
                train_losses.append(1.0*sum([ck * closs for ck, closs in zip(self.client_vols, train_loss)])/self.data_vol)
                c_accs.append(accs)
                test_losses.append(test_loss)
                test_accs.append(test_acc)
                valid_accs.append(1.0*sum([ck * acc for ck, acc in zip(self.client_vols, accs)])/self.data_vol)
                mean_c_accs.append(np.mean(accs))
                std_c_accs.append(np.std(accs))
                # print
                temp = "{:<30s}{:.4f}"
                print(temp.format("Training Loss:", train_losses[-1]))
                print(temp.format("Testing Loss:", test_losses[-1]))
                print(temp.format("Testing Accuracy:", test_accs[-1]))
                print(temp.format("Validating Accuracy:",valid_accs[-1]))
                print(temp.format("Mean of Client Accuracy:", mean_c_accs[-1]))
                print(temp.format("Std of Client Accuracy:", std_c_accs[-1]))
        # create_outdict
        outdict={
            "meta":self.option,
            "acc_dist":accs,
            "mean_curve":mean_c_accs,
            "var_curve":std_c_accs,
            "train_losses":train_losses,
            "test_accs":test_accs,
            "test_losses":test_losses,
            "valid_accs":valid_accs,
            "client_accs":{}
            }
        for cid in range(self.num_clients):
            outdict['client_accs'][self.clients[cid].name]=[c_accs[i][cid] for i in range(len(c_accs))]
        return outdict

    def communicate(self, cid):
        # setting client(cid)'s model with latest parameters
        self.trans_model.load_state_dict(self.model.state_dict())
        self.clients[cid].setModel(self.trans_model)
        # waiting for the reply of the update and loss
        return self.clients[cid].reply()

    def iterate(self, t):
        ws, losses = [], []
        loss_avg = -1
        # sample clients
        selected_clients = self.sample()
        # training
        for cid in selected_clients:
            w, loss = self.communicate(cid)
            ws.append(w)
            losses.append(loss)
        # aggregate
        w_new = self.aggregate(ws, p=[1.0*self.client_vols[id]/self.data_vol for id in selected_clients])
        self.model.load_state_dict(w_new)
        # output info
        nks = [self.client_vols[cid] for cid in selected_clients]
        p = [1.0*nk/sum(nks) for nk in nks]
        loss_avg = sum([loss*pk for loss,pk in zip(losses, p)])
        return loss_avg

    def global_lr_scheduler(self, round):
        if self.lr_scheduler_type == -1:
            return
        elif self.lr_scheduler_type == 0:
            """eta_{round+1} = DecayRate * eta_{round}"""
            self.lr*=self.decay_rate
            for c in self.clients:
                c.set_learning_rate(self.lr)
        elif self.lr_scheduler_type == 1:
            """eta_{t+1} = DecayRate * eta_{t}"""
            self.lr = float(self.lr * np.power(self.decay_rate, self.option['num_epochs']))

    def sample(self, replacement=False):
        cids = [i for i in range(self.num_clients)]
        selected_cids = []
        if self.sample_option == 'uniform': # original sample
            selected_cids = list(np.random.choice(cids, self.clients_per_round, replace=False))
        elif self.sample_option =='md': # default
            selected_cids = list(np.random.choice(cids, self.clients_per_round, replace=False, p=[nk / self.data_vol for nk in self.client_vols]))
            # selected_cids = list(np.random.choice(cids, self.clients_per_round, replace=True, p=[nk/self.data_vol for nk in self.client_vols]))
        # client dropout
        selected_cids = [cid for cid in selected_cids if self.clients[cid].is_available()]
        return selected_cids

    def aggregate(self, ws, p=[]):
        if not ws: return self.model.state_dict()
        """
        pk = nk/n where n=self.data_vol
        K = |S_t|
        N = |S|
        --------------------------------------------------------------------------------------------
         weighted_scale                 |uniform (default)          |weighted_com (original fedavg)
        ============================================================================================
        N/K * Σpk * wk                 |1/K * Σwk                  |(1-Σpk) * w_old + Σpk * wk
        """
        if self.agg_option == 'weighted_scale':
            K = len(ws)
            N = self.num_clients
            q = [1.0*pk*N/K for pk in p]
            return fmodule.modeldict_weighted_average(ws, q)
        elif self.agg_option == 'uniform':
            return fmodule.modeldict_weighted_average(ws)
        elif self.agg_option == 'weighted_com':
            return fmodule.modeldict_add(fmodule.modeldict_scale(self.model.state_dict(), 1 - sum(p)), fmodule.modeldict_weighted_average(ws, p))

    def test_on_clients(self, round, dataflag='valid'):
        """ Validate accuracies and losses """
        accs, losses = [], []
        for c in self.clients:
            c.setModel(self.model)
            acc, loss = c.test(dataflag)
            accs.append(acc)
            losses.append(loss)
        return accs, losses

    def test_on_dtest(self):
        if self.dtest:
            return fmodule.test(self.model, self.dtest)

    def clients_drop(self):
        for c in self.clients:
            c.set_available()
        return

class BaseClient():
    def __init__(self,  option, name = '', data_train_dict = {'x':[],'y':[]}, data_val_dict={'x':[],'y':[]}, partition = 0.8, drop_rate = -1):
        self.name = name
        self.frequency = 0
        # client's benchmark
        if partition==0:
            self.train_data = fmodule.XYDataset(data_train_dict['x'], data_train_dict['y'])
            self.valid_data = fmodule.XYDataset(data_val_dict['x'], data_val_dict['y'])
        else:
            data_x = data_train_dict['x'] + data_val_dict['x']
            data_y = data_train_dict['y'] + data_val_dict['y']
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)
            if partition==1:
                self.train_data = fmodule.XYDataset(data_x, data_y)
                self.valid_data = self.train_data
            else:
                k = int(len(data_x) * partition)
                self.train_data = fmodule.XYDataset(data_x[:k], data_y[:k])
                self.valid_data = fmodule.XYDataset(data_x[k:], data_y[k:])
        self.datavol = len(self.train_data)
        self.drop_rate = drop_rate if drop_rate>0.01 else 0
        # hyper-parameters for training
        self.epochs = option['num_epochs']
        self.learning_rate = option['learning_rate']
        self.batch_size = option['batch_size']
        self.momentum = option['momentum']
        self.weight_decay = option['weight_decay']
        self.model = None

    def setModel(self, model):
        self.model = model

    def set_learning_rate(self, lr = 0):
        self.learning_rate = lr if lr else self.learning_rate

    def is_available(self):
        if self.drop_rate==0: return True
        else: return (np.random.rand() < self.drop_rate)

    def train(self):
        self.model.train()
        if self.batch_size == -1:
            # full gradient descent
            self.batch_size = len(self.train_data)
        ldr_train = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        optimizer = optim(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay = self.weight_decay)
        epoch_loss = []
        for iter in range(self.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(device), labels.to(device)
                self.model.zero_grad()
                outputs = self.model(images)
                loss = lossfunc(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item() / len(labels))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return sum(epoch_loss) / len(epoch_loss)

    def test(self, dataflag='valid'):
        if dataflag == 'valid':
            return fmodule.test(self.model, self.valid_data)
        elif dataflag == 'train':
            return fmodule.test(self.model, self.train_data)

    def train_loss(self):
        _, loss = fmodule.test(self.model, self.train_data)
        return loss

    def valid_loss(self):
        _, loss = fmodule.test(self.model, self.valid_data)
        return loss

    def reply(self):
        self.train()
        return copy.deepcopy(self.model.state_dict()), self.train_loss()