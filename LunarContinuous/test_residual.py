""" Test file for Residual Policy Gradient Agent """
import gym
import numpy as np
import torch as T
import os
from datetime import datetime
from Residual import Agent
from utils import norm_action, normalize_action, add_actions
from joystick import LunarLanderJoystickActor
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v2")

    N = 128
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    n_games = 100

    filename=f"tmp/residual_final_reg"

    assistive_agent = Agent(n_actions=env.action_space.shape[0], input_dims=env.observation_space.shape[0],
                   gamma=0.99, gae_lambda=0.95,
                  lr_decay_rate=0.31622776601, lr_decay_freq=20000000,
                     chkpt_dir=filename)
    assistive_agent.load_models()

    player_agent = LunarLanderJoystickActor(env)
    score_history = []
    for ep in range(10):
        ob = env.reset()
        env.render()
        done = False
        reward = 0.0
        n_steps=0
        action_h = np.empty((0, env.action_space.shape[0]), dtype=float)
        action_r = np.empty((0, env.action_space.shape[0]), dtype=float)
        actions = np.empty((0, env.action_space.shape[0]), dtype=float)
        while not done:
            n_steps += 1
            action_ = player_agent(ob)
            action_player = norm_action(env, np.asarray(action_))
            action_player = T.tensor(np.array([action_player]), dtype=T.float32).to(assistive_agent.device)
            ob_ = T.tensor(np.array([ob]), dtype=T.float32).to(assistive_agent.device)
            observation = (ob_, action_player)
            action_ass, prob, val = assistive_agent.choose_action(observation)
            action = add_actions(env, action_ass, np.asarray(action_))
            ob, r, done, info = env.step(action[0])
            reward += r
            action_h = np.vstack((action_h, action_))
            action_r = np.vstack((action_r, action_ass[0]))
            actions = np.vstack((actions, action[0]))
        y = [i+1 for i in range(n_steps)]
        plt.plot(y, action_h[:,0], label='action_h[0]', color='darkblue')
        plt.plot(y, action_r[:,0], label='action_r[0]', color='darkgreen')
        plt.plot(y, actions[:,0], label='action[0]', color='crimson')
        plt.plot(y, action_h[:,1], label='action_h[1]', color='lightblue')
        plt.plot(y, action_r[:,1], label='action_r[1]', color='greenyellow')
        plt.plot(y, actions[:,1], label='action[1]', color='salmon')

        plt.xlabel('Iterations')
        plt.ylabel('Action Values')
        plt.legend()
        plt.title('Comparison of Actions')
        plt.savefig(f"plots/results_{ep}")
        plt.clf()
        score_history.append(reward)
        print(reward)
        
    env.close()
    x = [i+1 for i in range(len(score_history))]
    plt.plot(x, score_history, color='darkblue')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Score of Lunar Lander')
    plt.savefig(f"plots/score{ep}")
    plt.clf()

