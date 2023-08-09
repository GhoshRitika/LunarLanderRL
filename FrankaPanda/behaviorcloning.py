import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import csv
import pandas as pd
import ast
import numpy as np
from torch.distributions import Normal
from PPO import PPOMemory
from torch.utils.data import DataLoader, TensorDataset
from diagonal_gaussian import DiagGaussian

class BCNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=128, fc2_dims=128, chkpt_dir='tmp/bc'):
        super(BCNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, 'behavioral_cloning')
        self.bc=nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            DiagGaussian(fc2_dims, n_actions))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        if T.cuda.is_available():
            device = T.device('cuda')
            print('GPU is available')
        else:
            device = T.device('cpu')
            print('GPU is not available, using CPU')
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.bc(state)
        continuous_actions = dist.sample()
        return continuous_actions

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class BC_Agent:
    def __init__(self, n_actions, input_dims, alpha=0.001, batch_size=64, n_epochs=10):
        self.n_epochs = n_epochs
        self.input_dims = input_dims
        self.bc = BCNetwork(n_actions, input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def save_models(self):
        print('... saving models ...')
        self.bc.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.bc.load_checkpoint()

    def train_behavioral_cloning(self, observations_csv, actions_csv):
        observations_df = pd.read_csv(observations_csv)
        actions_df = pd.read_csv(actions_csv)

        observations = observations_df['observation'].apply(ast.literal_eval)
        actions = actions_df['action'].apply(ast.literal_eval)

        observations = T.tensor(list(observations), dtype=T.float32, requires_grad=True).to(self.bc.device)
        actions = T.tensor(list(actions), dtype=T.float32, requires_grad=True).to(self.bc.device)

        dataloader = DataLoader(TensorDataset(observations, actions), batch_size=64, shuffle=True)

        criterion = nn.MSELoss()

        for epoch in range(self.n_epochs):
            running_loss = 0.0
            for inputs, targets in dataloader:
                self.bc.optimizer.zero_grad()
                inputs = inputs.to(self.bc.device).requires_grad_()
                targets = targets.to(self.bc.device)

                with T.set_grad_enabled(True):
                    outputs = self.bc(inputs)
                    if isinstance(outputs, Normal):
                        outputs = outputs.rsample()

                loss = criterion(outputs, targets)
                loss.backward()
                self.bc.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {running_loss/len(dataloader):.6f}")

        return self.bc
