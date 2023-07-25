import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from diagonal_gaussian import DiagGaussian

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = [] #the log probs
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        #batch size chunks of shuffled indices
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class ActorCriticNetwork(nn.Module):
    """Policy and Value networks."""
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=128, fc2_dims=128, fc3_dims=128, chkpt_dir='tmp/actorcritic'):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_critic')
        # self.fc1 = nn.Linear(input_dims, fc1_dims)
        # self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        # # self.fc3 = nn.Linear(fc2_dims, fc3_dims).float()
        # self.dist = DiagGaussian(fc2_dims, n_actions)
        # for p in self.dist.fc_mean.parameters():
        #     nn.init.constant_(p, 0.)

        # self.vf_fc1 = nn.Linear(input_dims, fc1_dims)
        # self.vf_fc2 = nn.Linear(fc1_dims, fc2_dims)
        # # self.vf_fc3 = nn.Linear(fc2_dims, fc3_dims).float()
        # self.vf_out = nn.Linear(fc2_dims, 1)
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            DiagGaussian(fc2_dims, n_actions))
        self.critic = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1))
        # self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        """Forward."""
        # ob = T.cat(x, axis=1)
        # ob=x
        # net = ob
        # net = F.relu(self.fc1(net))
        # net = F.relu(self.fc2(net))
        # # net = F.relu(self.fc3(net))
        # pi = self.dist(net)

        # net = ob
        # net = F.relu(self.vf_fc1(net))
        # net = F.relu(self.vf_fc2(net))
        # # net = F.relu(self.vf_fc3(net))
        # vf = self.vf_out(net)
        pi=self.actor(x)
        # for p in self.dist.fc_mean.parameters():
        #     nn.init.constant_(p, 0.)
        vf = self.critic(x)

        return pi, vf

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    """Base agent class"""
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, N=2048, n_epochs=10, lr_decay_rate=1./3.16227766017,
                 lr_decay_freq=20000000, rollout_length=128, nenv=1, chkpt_dir='tmp/ppo'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir=chkpt_dir
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_freq = lr_decay_freq
        self.rollout_length = rollout_length
        self.nenv= nenv
        self.actorcritic = ActorCriticNetwork(n_actions, input_dims, alpha, chkpt_dir=self.chkpt_dir)
        self.opt = optim.Adam(self.actorcritic.parameters())
        self.pi_lr = self.opt.param_groups[0]['lr']
        # self.critic = CriticNetwork(input_dims, alpha, chkpt_dir=self.chkpt_dir)
        self.memory = PPOMemory(batch_size)
        self.mse = nn.MSELoss(reduction='none')
        self.t = 0
       
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actorcritic.save_checkpoint()
        # self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actorcritic.load_checkpoint()
        # self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.actorcritic.device)
        dist, value = self.actorcritic(state)
        action = dist.sample()
        probs =  dist.log_prob(action)
        probs = probs.detach().cpu().numpy()
        action = action.detach().cpu().numpy()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        lr_frac = self.lr_decay_rate ** (self.t // self.lr_decay_freq)
        for g in self.opt.param_groups:
            g['lr'] = self.pi_lr * lr_frac
        # self.t += self.rollout_length * self.nenv
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()
        
            values = vals_arr
            atargs = np.zeros(len(reward_arr), dtype=np.float32)
            vtargs = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0 #adavantages at each timestep starts out as 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                atargs[t] = a_t
            #Compute vtargs by adding atargs to the value predictions (values).
            vtargs[:-1] = atargs[:-1] + values[1:] * (1 - dones_arr[:-1])
            #The last value target (vtargs[-1]) is simply the last value prediction (values[-1]).
            vtargs[-1] = values[-1]
            atargs = T.tensor(atargs, dtype=T.float32).to(self.actorcritic.device)
            vtargs = T.tensor(vtargs, dtype=T.float32).to(self.actorcritic.device)
            values = T.tensor(values, dtype=T.float32).to(self.actorcritic.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actorcritic.device)
                old_probs = T.tensor(old_prob_arr[batch], dtype=T.float).to(self.actorcritic.device)
                actions = T.tensor(action_arr[batch], dtype=T.float).to(self.actorcritic.device)

                dist, critic_value = self.actorcritic(states)

                critic_value = T.squeeze(critic_value)
                new_probs = dist.log_prob(actions)

                ratio = T.exp(new_probs - old_probs).t()
                ploss1 = ratio*atargs[batch]
                ploss2 = T.clamp(ratio, 1.0-self.policy_clip, 1.0+self.policy_clip)*atargs[batch]
                pi_loss = -T.min(ploss1, ploss2).mean()
                vloss1 = 0.5*self.mse(critic_value, vtargs[batch])
                value_clipped = values[batch]+ (critic_value-values[batch]).clamp(-self.policy_clip, self.policy_clip)
                vloss2 = 0.5*self.mse(value_clipped, vtargs[batch])
                value_loss = T.max(vloss1, vloss2).mean()
                # returns = atargs[batch] + values[batch]
                # critic_loss = (returns-critic_value)**2
                # value_loss = critic_loss.mean()
                total_loss = (pi_loss + value_loss)
                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()
        self.memory.clear_memory()

