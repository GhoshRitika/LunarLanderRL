import gym
import numpy as np
import torch as T
import os
from datetime import datetime
from continuous_ppo import Agent
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

    filename=f"tmp/residual202308021104"
    # filename=f"tmp/ppo202308011529"
    # filename=f"tmp/ppo20230726134213"

    assistive_agent = Agent(n_actions=env.action_space.shape[0], input_dims=env.observation_space.shape[0],
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
        n_steps=0
        action_h = np.empty((0, env.action_space.shape[0]), dtype=float)
        action_r = np.empty((0, env.action_space.shape[0]), dtype=float)
        actions = np.empty((0, env.action_space.shape[0]), dtype=float)
        while not done:
            n_steps += 1
            action_ = player_agent(ob)
            action_player = norm_action(env, np.asarray(action_))
            # action_player = norm_action(env, action_.cpu().numpy())
            action_player = T.tensor(np.array([action_player]), dtype=T.float32).to(assistive_agent.device)
            ob_ = T.tensor(np.array([ob]), dtype=T.float32).to(assistive_agent.device)
            observation = (ob_, action_player)
            # print(observation)
            action_ass, prob, val = assistive_agent.choose_action(observation)
            # print(action_ass)
            # action_ass = normalize_action(np.asarray(action_ass), -3, 3)
            action = add_actions(env, action_ass, np.asarray(action_))
            # action = add_actions(env, np.asarray(action_ass), action_.cpu().numpy())
            ob, r, done, info, _ = env.step(action[0])
            reward += r
            # print("Assisstve acton", action_ass)
            # print("Player action", action_)
            # print("Total action", action)
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
        plt.savefig(f"plots/results{ep}")
        plt.clf()
        print(reward)
        
    env.close()

    # y = [i+1 for i in range(n_steps)]
    # plt.plot(y, action_h[:,0], label='action_h[0]', color='blue')
    # plt.plot(y, action_r[:,0], label='action_r[0]', color='green')
    # plt.plot(y, actions[:,0], label='action[0]', color='red')
    # plt.plot(y, action_h[:,1], label='action_h[1]', color='blue')
    # plt.plot(y, action_r[:,1], label='action_r[1]', color='green')
    # plt.plot(y, actions[:,1], label='action[1]', color='red')

    # plt.xlabel('Iterations')
    # plt.ylabel('Action Values')
    # plt.legend()
    # plt.title('Comparison of Actions')
    # plt.show()
