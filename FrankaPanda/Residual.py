import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from diagonal_gaussian import DiagGaussian
from utils import normalize_action

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
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
        # print("storing memory", action.shape)
        self.actions.append(action[0])
        self.probs.append(probs[0])
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
    def __init__(self, n_actions, input_dims, fc1_dims=128, fc2_dims=128, chkpt_dir='tmp/actorcritic_continuous'):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = input_dims+n_actions
        print((self.input_dims, fc1_dims))
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, 'residual')
        self.fc1 = nn.Linear(self.input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.dist = DiagGaussian(fc2_dims, n_actions, constant_log_std=False)
        for p in self.dist.fc_mean.parameters():
            nn.init.constant_(p, 0.)

        self.vf_fc1 = nn.Linear(self.input_dims, fc1_dims)
        self.vf_fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.vf_out = nn.Linear(fc2_dims, 1)

    def forward(self, x):
        """Forward."""
        ob = T.cat(x, dim=1)
        net = ob
        net = F.relu(self.fc1(net))
        net = F.relu(self.fc2(net))
        pi, mean, std = self.dist(net)

        net = ob
        net = F.relu(self.vf_fc1(net))
        net = F.relu(self.vf_fc2(net))
        vf = self.vf_out(net)

        return pi, vf, mean, std

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self, device='cpu'):
        self.load_state_dict(T.load(self.checkpoint_file, map_location=device))

class Agent:
    """Base agent class"""
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
            policy_clip=0.2, batch_size=64, N=2048, n_epochs=10, lr_decay_rate=1./3.16227766017,
                 lr_decay_freq=20000000, rollout_length=128, nenv=1, max_grad_norm=0.5, 
                 ent_coef=0.01, vf_coef=0.5,policy_training_start=10000,
                 lambda_training_start=100000,lambda_lr=1e-4,
                 lambda_init=100.,l2_reg=True, reward_threshold=-0.05, chkpt_dir='tmp/ppo'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.chkpt_dir=chkpt_dir
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_freq = lr_decay_freq
        self.rollout_length = rollout_length
        self.nenv= nenv
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.actorcritic = ActorCriticNetwork(n_actions, input_dims, chkpt_dir=self.chkpt_dir).to(self.device)
        self.opt = optim.Adam(self.actorcritic.parameters())
        self.pi_lr = self.opt.param_groups[0]['lr']
        # self.critic = CriticNetwork(input_dims, alpha, chkpt_dir=self.chkpt_dir)
        self.max_grad_norm = max_grad_norm
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.policy_training_start = policy_training_start
        self.lambda_training_start = lambda_training_start
        self.lambda_lr = lambda_lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_freq = lr_decay_freq
        self.l2_reg = l2_reg
        self.reward_threshold = reward_threshold

        if lambda_init < 10:
            lambda_init = np.log(np.exp(lambda_init) - 1)
        log_lambda_tensor = T.tensor(np.array([lambda_init]), dtype=T.float32).to(self.device)
        # print(log_lambda_tensor)
        self.log_lambda_ = nn.Parameter(
                            log_lambda_tensor)
        # print(self.log_lambda_)
        self.opt_l = optim.Adam([self.log_lambda_], lr=lambda_lr)

        self.memory = PPOMemory(batch_size)
        self.mse = nn.MSELoss(reduction='none')
        self.huber = nn.SmoothL1Loss()
        self.t = 0

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actorcritic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actorcritic.load_checkpoint()

# tmp/residual_dense_reward_tanh doesnt exceed any limits, works OK
    def choose_action(self, observation):
            dist, value, mean, std = self.actorcritic(observation)
            action = dist.sample()
            probs =  dist.log_prob(action).detach().cpu().numpy()
            value_ass = T.squeeze(value).item()
            action_detached = action.detach().cpu().numpy()
            mean_detached = mean.detach().cpu().numpy()
            std_detached = std.detach().cpu().numpy()
            action_ass = np.tanh(normalize_action(np.asarray(action_detached), (mean_detached-std_detached), (mean_detached+std_detached)))
            # print(action_ass)
            return action_ass, probs, value_ass

## filename=f"tmp/residual_dense_reward_just_tanh" Barely does anything
    # def choose_action(self, observation):
    #     dist, value, mean, std = self.actorcritic(observation)
    #     action = dist.sample()
    #     probs =  dist.log_prob(action).detach().cpu().numpy()
    #     value_ass = T.squeeze(value).item()
    #     action_detached = action.detach().cpu().numpy()
    #     mean_detached = mean.detach().cpu().numpy()
    #     std_detached = std.detach().cpu().numpy()
    #     action_ass = np.tanh(action_detached)
    #     print(action_ass)
    #     return action_ass, probs, value_ass

## tmp/residual_dense_reward NOT A FAN, exceed actions limits
    # def choose_action(self, observation):
    #     dist, value, mean, std = self.actorcritic(observation)
    #     action = dist.sample()
    #     probs =  dist.log_prob(action).detach().cpu().numpy()
    #     value_ass = T.squeeze(value).item()
    #     action_detached = action.detach().cpu().numpy()
    #     mean_detached = mean.detach().cpu().numpy()
    #     std_detached = std.detach().cpu().numpy()
    #     action_ass = normalize_action(np.asarray(action_detached), (mean_detached-std_detached), (mean_detached+std_detached))
    #     print(action_ass)
    #     return action_ass, probs, value_ass

    def learn(self):
        lr_frac = self.lr_decay_rate ** (self.t // self.lr_decay_freq)
        for g in self.opt.param_groups:
            g['lr'] = self.pi_lr * lr_frac
        for g in self.opt_l.param_groups:
            g['lr'] = self.lambda_lr * lr_frac
        self.t = self.t + (self.rollout_length * self.nenv)
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()
            #Calculating PPO losses
            values_arr = vals_arr
            atargs_arr = np.zeros(len(reward_arr), dtype=np.float32)
            vtargs_arr = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t = a_t + discount*(reward_arr[k] + self.gamma*values_arr[k+1]*\
                            (1-int(dones_arr[k])) - values_arr[k])
                    discount = discount* self.gamma*self.gae_lambda
                atargs_arr[t] = a_t
            #Compute vtargs by adding atargs to the value predictions (values).
            vtargs_arr[:-1] = atargs_arr[:-1] + values_arr[1:] * (1 - dones_arr[:-1])
            #The last value target (vtargs[-1]) is simply the last value prediction (values[-1]).
            vtargs_arr[-1] = values_arr[-1]
            atargs = T.tensor(atargs_arr, dtype=T.float32).to(self.device)
            vtargs = T.tensor(vtargs_arr, dtype=T.float32).to(self.device)
            values = T.tensor(values_arr, dtype=T.float32).to(self.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float32).to(self.device)
                old_probs = T.tensor(old_prob_arr[batch], dtype=T.float32).to(self.device)
                actions = T.tensor(action_arr[batch], dtype=T.float32).to(self.device)

                dist, critic_value_, _, _ = self.actorcritic((states, actions))
                critic_value = T.squeeze(critic_value_)
                # if self.t < self.policy_training_start:
                #     pi_loss = T.Tensor([0.0]).to(self.device)
                # else:
                new_probs = dist.log_prob(actions)
                ratio = T.exp(new_probs - old_probs)
                # Add a new dimension to atargs to match the last dimension of ratio
                atargs_batch = atargs[batch].unsqueeze(-1)
                ploss1 = ratio*atargs_batch
                ploss2 = T.clamp(ratio, 1.0-self.policy_clip, 1.0+self.policy_clip)*atargs_batch
                pi_loss = -T.min(ploss1, ploss2).mean()
                vloss1 = 0.5*self.mse(critic_value, vtargs[batch])
                value_clipped = values[batch]+ (critic_value-values[batch]).clamp(-self.policy_clip, self.policy_clip)
                vloss2 = 0.5*self.mse(value_clipped, vtargs[batch])
                value_loss = T.max(vloss1, vloss2).mean()

                # compute entropy loss
                # if self.t < self.policy_training_start:
                #     ent_loss = T.Tensor([0.0]).to(self.device)
                # else:
                ent_loss = dist.entropy().mean()
                # compute residual regularizer
                # if self.t < self.policy_training_start:
                #     reg_loss = T.Tensor([0.0]).to(self.device)
                # else:
                if self.l2_reg:
                    reg_loss = dist.rsample().pow(2).sum(dim=-1).mean()
                else:  # huber loss
                    ac_norm = T.norm(dist.rsample(), dim=-1)
                    reg_loss = self.huber(ac_norm, T.zeros_like(ac_norm, dtype=T.float32))

                ###############################
                # Constrained loss added here.
                ###############################

                # soft plus on lambda to constrain it to be positive.
                lambda_ = F.softplus(self.log_lambda_)
                # print(self.log_lambda_)
                # print(self.log_lambda_)
                # if self.t < max(self.policy_training_start, self.lambda_training_start):
                #   loss_lambda = T.Tensor([0.0]).to(self.device)
                # else:
                rewards = T.tensor(reward_arr[batch], dtype=T.float32).to(self.device)
                dones = T.tensor(dones_arr[batch], dtype=T.float32).to(self.device)
                neps = (1.0 - dones).sum()
                # print(neps)
                loss_lambda = (lambda_ * (rewards.sum()
                                            - self.reward_threshold * neps)
                                / rewards.size()[0])
                # Detach the lambda_ tensor
                detached_lambda = lambda_.detach()
                # Compute pi_loss using detached_lambda
                new_pi_loss = (reg_loss + detached_lambda * pi_loss) / (1.0 + detached_lambda)
                # print(loss_lambda)
                # if self.t >= self.policy_training_start:
                # pi_loss = (reg_loss + lambda_ * pi_loss) / (1.0 + lambda_)
                total_loss = (new_pi_loss + self.vf_coef * value_loss
                                - self.ent_coef * ent_loss)
                # print(total_loss)
                # total_loss = (pi_loss + value_loss)
                # if self.t >= max(self.policy_training_start,
                #                  self.lambda_training_start):
                self.opt_l.zero_grad()
                loss_lambda.backward(retain_graph=True)
                self.opt_l.step()
                T.autograd.set_detect_anomaly(True)
                self.opt.zero_grad()
                total_loss.backward()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.actorcritic.parameters(),
                                             self.max_grad_norm)
                self.opt.step()

        self.memory.clear_memory()
        return total_loss, loss_lambda

