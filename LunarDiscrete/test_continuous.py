import gym
import numpy as np
import torch as T
import os
from datetime import datetime
from constrained_residualPPO import Agent
from utils import norm_action, normalize_action, add_actions
from Joystick import LunarLanderJoystickActor
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v2", render_mode="human")

    N = 128 #2048
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    n_games = 100

    filename=f"tmp/ppo202308011755"

    assistive_agent = Agent(n_actions=env.action_space.shape[0], input_dims=env.observation_space.shape[0],
                batch_size=64, policy_clip=0.2,
                ent_coef=0.01, n_epochs=10,
                l2_reg=False, lambda_init=20.0,
                lambda_lr=0.003,
                lambda_training_start=2000000,
                max_grad_norm=0.5,
                nenv=1, policy_training_start= 100000,
                reward_threshold=-155.0, vf_coef=0.5,
                gamma=0.99, gae_lambda=0.95,
                lr_decay_rate=0.31622776601, lr_decay_freq=20000000,
                chkpt_dir=filename)
    assistive_agent.load_models()
    
    player_agent = LunarLanderJoystickActor(env)

    for ep in range(10):
        ob = env.reset()
        ob=ob[0]
        env.render()
        done = False
        reward = 0.0

        while not done:
            action_ass, prob, val = assistive_agent.choose_action(ob)
            # print("action:",action_ass)
            ob, r, done, info,_ = env.step(action_ass)
            reward += r
            print(r)
        print("reward:", reward)
        
    env.close()
