import gymnasium as gym
import numpy as np
import torch as T
from datetime import datetime
from Residual import Agent
from utils import norm_action, normalize_action, add_actions
from joystck import FrankaPandaJoystickActor
import matplotlib.pyplot as plt
import panda_gym

if __name__ == '__main__':
    env = gym.make('PandaReachDense-v3', render_mode="human", goal_random=False)
    N = 128 #2048
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    n_games = 100
    goals = np.array([[-0.09425813, -0.05269751, 0.07325654], 
                      [0.00141661, 0.00123507, 0.14757843],
                      [-0.00135891, -0.08630279, 0.04110381],
                      [ 0.07980205, -0.01122603, 0.11293837],
                      [-0.13518204,  0.13425152, 0.08225469],
                      [0.07082976, 0.13334234, 0.083761],
                      [-0.11354388, 0.13814254, 0.05822545],
                      [0.01594417, 0.07936273, 0.22025128],
                      [ 0.09835389, -0.07078907,  0.17652114],
                      [-0.00091254,  0.05386124,  0.28241307]])
    # filename=f"tmp/residual_dense_reward_just_tanh"
    filename=f"tmp/residual_dense_reward_tanh"
    # filename=f"tmp/residual_dense_reward"

    assistive_agent = Agent(n_actions=env.action_space.shape[0], input_dims=env.observation_space["observation"].shape[0],
                   gamma=0.99, gae_lambda=0.95,
                  lr_decay_rate=0.31622776601, lr_decay_freq=20000000,
                     chkpt_dir=filename)
    assistive_agent.load_models()

    player_agent = FrankaPandaJoystickActor(env)
    max_iters = 1000
    for ep in range(10):
        goal = goals[ep]
        print(goal)
        ob, info = env.reset(goal_val = np.array(goal))
        env.render()
        done = False
        reward = 0.0
        n_steps=0
        iters = 0
        print("Initial", ob["observation"][0:3])
        print("desired", ob["desired_goal"][0:3])
        action_h = np.empty((0, env.action_space.shape[0]), dtype=float)
        action_r = np.empty((0, env.action_space.shape[0]), dtype=float)
        actions = np.empty((0, env.action_space.shape[0]), dtype=float)
        while not done:
            action_ = player_agent(ob["observation"])
            action_player = norm_action(env, np.asarray(action_))
            action_player = T.tensor(np.array([action_player]), dtype=T.float32).to(assistive_agent.device)
            ob_ = T.tensor(np.array([ob["observation"]]), dtype=T.float32).to(assistive_agent.device)
            observation = (ob_, action_player)
            action_ass, prob, val = assistive_agent.choose_action(observation)
            action = add_actions(env, action_ass, np.asarray(action_))
            ob, r, done, _, info = env.step(action[0])
            reward += r
            n_steps += 1
            iters += 1
            action_h = np.vstack((action_h, action_))
            action_r = np.vstack((action_r, action_ass[0]))
            actions = np.vstack((actions, action[0]))
        print("score", reward)
        y = [i+1 for i in range(n_steps)]
        plt.plot(y, action_h[:,0], label='action_h[0]', color='darkblue')
        plt.plot(y, action_h[:,1], label='action_h[1]', color='lightblue')
        plt.plot(y, action_h[:,2], label='action_h[2]', color='aqua')
        plt.xlabel('Iterations')
        plt.ylabel('Action Values')
        plt.legend()
        plt.title('Human of Actions')
        plt.savefig(f"plots/results_human{ep}")
        plt.clf()

        plt.plot(y, action_r[:,0], label='action_r[0]', color='darkgreen')
        plt.plot(y, action_r[:,1], label='action_r[1]', color='greenyellow')
        plt.plot(y, action_r[:,2], label='action_r[2]', color='yellow')
        plt.xlabel('Iterations')
        plt.ylabel('Action Values')
        plt.legend()
        plt.title('Assisstant of Actions')
        plt.savefig(f"plots/results_assistant{ep}")
        plt.clf()

        plt.plot(y, actions[:,0], label='action[0]', color='crimson')
        plt.plot(y, actions[:,1], label='action[1]', color='salmon')
        plt.plot(y, actions[:,2], label='action[2]', color='orange')
        plt.xlabel('Iterations')
        plt.ylabel('Action Values')
        plt.legend()
        plt.title('Actions')
        plt.savefig(f"plots/results{ep}")
        plt.clf()

    env.close()

