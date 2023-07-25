import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import pandas as pd
import ast
import numpy as np
from ppo_torch import PPOMemory
from torch.distributions import Normal
from diagonal_gaussian import DiagGaussian
import logger
import time


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
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.dist = DiagGaussian(fc3_dims, n_actions)
        for p in self.dist.fc_mean.parameters():
            nn.init.constant_(p, 0.)

        self.vf_fc1 = nn.Linear(*input_dims, fc1_dims)
        self.vf_fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.vf_fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.vf_out = nn.Linear(fc3_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        """Forward."""
        ob = T.cat(x, axis=1)
        net = ob
        net = F.relu(self.fc1(net))
        net = F.relu(self.fc2(net))
        net = F.relu(self.fc3(net))
        pi = self.dist(net)

        net = ob
        net = F.relu(self.vf_fc1(net))
        net = F.relu(self.vf_fc2(net))
        net = F.relu(self.vf_fc3(net))
        vf = self.vf_out(net)

        return pi, vf

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class ResidualPPOActor(object):
    """Actor."""
    def __init__(self, pi, policy_training_start):
        """Init."""
        self.pi = pi
        self.policy_training_start = policy_training_start
        self.t = 0

    def __call__(self, ob, state_in=None, mask=None):
        """Produce decision from model."""
        if self.t < self.policy_training_start:
            outs = self.pi(ob, state_in, mask, deterministic=True)
            if not T.allclose(outs.action, T.zeros_like(outs.action)):
                raise ValueError("Pi should be initialized to output zero "
                                 "actions so that an acurate value function "
                                 "can be learned for the base policy.")
        else:
            outs = self.pi(ob, state_in, mask)
        residual_norm = T.mean(T.sum(T.abs(outs.action), dim=1))
        logger.add_scalar('actor/l1_residual_norm', residual_norm, self.t,
                          time.time())
        self.t += outs.action.shape[0]
        data = {'action': outs.action,
                'value': outs.value,
                'logp': outs.dist.log_prob(outs.action)}
        if outs.state_out:
            data['state'] = outs.state_out
        return data

    def state_dict(self):
        """State dict."""
        return {'t': self.t}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.t = state_dict['t']

class ConstrainedResidualPPO:
    """Constrained Residual PPO algorithm."""
    def __init__(self,n_actions, input_dims, nenv=1,
                 optimizer=T.optim.Adam,
                 lambda_lr=1e-4,
                 lambda_init=100.,
                 lr_decay_rate=1./3.16227766017,
                 lr_decay_freq=20000000,
                 l2_reg=True,
                 reward_threshold=-0.05,
                 rollout_length=128,
                 batch_size=32,
                 gamma=0.99,
                 lambda_=0.95,
                 norm_advantages=False,
                 epochs_per_rollout=10,
                 max_grad_norm=None,
                 ent_coef=0.01,
                 vf_coef=0.5,
                 clip_param=0.2,
                 base_actor_cls=None,
                 policy_training_start=10000,
                 lambda_training_start=100000,
                 eval_num_episodes=1,
                 record_num_episodes=1,
                 chkpt_dir='tmp/residual'):
        """Init."""
        self.chkpt_dir=chkpt_dir
        # self.ckptr = Checkpointer(os.path.join(chkpt_dir, 'ckpts'))
        self.clip_param = clip_param
        self.lambda_lr = lambda_lr
        self.pi = ActorCriticNetwork(n_actions, input_dims, alpha=lambda_lr)
        self.opt = optimizer(self.pi.parameters())
        self.eval_num_episodes = eval_num_episodes
        self.record_num_episodes = record_num_episodes
        self.epochs_per_rollout = epochs_per_rollout
        self.rollout_length=rollout_length
        # self.max_grad_norm = max_grad_norm
        # self.ent_coef = ent_coef
        # self.vf_coef = vf_coef
        
        # # self.base_actor_cls = base_actor_cls
        self.policy_training_start = policy_training_start
        self.lambda_training_start = lambda_training_start
        self.lambda_lr = lambda_lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_freq = lr_decay_freq
        self.l2_reg = l2_reg
        # self.reward_threshold = reward_threshold
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        if lambda_init < 10:
            lambda_init = np.log(np.exp(lambda_init) - 1)
        self.log_lambda_ = nn.Parameter(
                            T.Tensor([lambda_init]).to(self.device))
        self.opt_l = optimizer([self.log_lambda_], lr=lambda_lr)

        self.mse = nn.MSELoss(reduction='none')
        self.huber = nn.SmoothL1Loss()
        self.memory = PPOMemory(batch_size)
        self.t = 0

    def loss(self, batch):
        """Compute loss."""

        loss = {}
        # compute policy loss
        if self.t < self.policy_training_start:
            pi_loss = T.Tensor([0.0]).to(self.device)
        else:
            dist, value = self.pi(batch['obs'])
            logp = dist.log_prob(batch['action'])
            ratio = T.exp(logp - batch['logp'])
            ploss1 = ratio * batch['atarg']
            ploss2 = T.clamp(ratio, 1.0-self.clip_param,
                                 1.0+self.clip_param) * batch['atarg']
            pi_loss = -T.min(ploss1, ploss2).mean()
        loss['pi'] = pi_loss

        # compute value loss
        vloss1 = 0.5 * self.mse(value, batch['vtarg'])
        vpred_clipped = batch['vpred'] + (
            value - batch['vpred']).clamp(-self.clip_param,
                                               self.clip_param)
        vloss2 = 0.5 * self.mse(vpred_clipped, batch['vtarg'])
        vf_loss = T.max(vloss1, vloss2).mean()
        loss['value'] = vf_loss

        return loss






