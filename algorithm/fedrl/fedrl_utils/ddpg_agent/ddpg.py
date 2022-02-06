from pathlib import Path
from ddpg_agent.utils import *
from ddpg_agent.networks import *
from ddpg_agent.policy import *
from ddpg_agent.buffer import *
import torch
import torch.nn as nn
import torch.optim as optim
from ddpg_agent.policy import NormalizedActions
import pickle

class DDPG_Agent(nn.Module):
    def __init__(
        self,
        state_dim=3,
        action_dim=1,
        hidden_dim=256,
        init_w=1e-3,
        value_lr=1e-3,
        policy_lr=1e-3,
        replay_buffer_size=1000000,
        max_steps=16*50,
        max_frames=12000,
        batch_size=4,
        beta=0.45,
        log_dir="./log/epochs",
        gamma = 0.99,
        soft_tau = 2e-2,
    ):
        super(DDPG_Agent, self).__init__()
        self.gamma = gamma
        self.soft_tau = soft_tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Init State dim", state_dim)      # K x 3
        print("Init Action dim", action_dim)    # K x 3

        self.value_net = ValueNetwork(num_input=state_dim + action_dim, hidden_size=hidden_dim).to(self.device).double()
        self.policy_net = PolicyNetwork(num_inputs=state_dim, num_outputs=action_dim, hidden_size=hidden_dim).to(self.device).double()

        self.target_value_net = ValueNetwork(num_input=state_dim + action_dim, hidden_size=hidden_dim).to(self.device).double()
        self.target_policy_net = PolicyNetwork(num_inputs=state_dim, num_outputs=action_dim, hidden_size=hidden_dim).to(self.device).double()


        model_path = "../models"
        if Path(f"{model_path}/policy_net.pth").exists():
            self.policy_net.load_state_dict(torch.load(f"{model_path}/policy_net.pth"))
        if Path(f"{model_path}/value_net.pth").exists():
            self.value_net.load_state_dict(torch.load(f"{model_path}/value_net.pth"))
        if Path(f"{model_path}/target_policy_net.pth").exists():
            self.target_policy_net.load_state_dict(torch.load(f"{model_path}/target_policy_net.pth"))
        if Path(f"{model_path}/target_value_net.pth").exists():
            self.target_value_net.load_state_dict(torch.load(f"{model_path}/target_value_net.pth"))


        # store all the (s, a, s', r) during the transition process
        self.memory = Memory()
        # replay buffer used for main training
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.value_criterion = nn.MSELoss()
        self.step = 0
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.log_dir = log_dir

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)


    def get_action(self, observation, time_step):
        done = observation['done']
        losses = observation['losses']
        n_samples = observation['n_samples']
        n_epochs = observation['n_epochs']

        # reach to maximum step for each episode or get the done for this iteration
        state = get_state(losses, n_samples, n_epochs)
        state = torch.DoubleTensor(state).unsqueeze(0).to(self.device)  # current state
        if time_step > 0:
            self.memory.update(r=get_reward())

        action = self.policy_net.get_action(state)
        self.memory.act(state, action)

        # if self.step < self.max_steps:
        if self.memory.get_last_record() is None:
            self.step += 1
            return action

        s, a, r, s_next = self.memory.get_last_record()
        self.replay_buffer.push(s, a, r, s_next, done)
        self.step += 1

        return action


    def dump_buffer(self, buffer_path, run_name):
        with open(f"{buffer_path}/{run_name}.exp", "ab") as fp:
            pickle.dump(self.replay_buffer.buffer, fp)