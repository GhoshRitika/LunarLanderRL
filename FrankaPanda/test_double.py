import gymnasium as gym
import numpy as np
import torch as T
from datetime import datetime
from Residual import Agent
from utils import norm_action, normalize_action, add_actions
from joystick import FrankaPandaJoystickActor
import matplotlib.pyplot as plt
import panda_gym

if __name__ == '__main__':
    env = gym.make('PandaReachDoubleDense-v3', render_mode="human", goal_random=False)
    N = 128 #2048
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    n_games = 100
    goals1 = np.array([[0.01763173, -0.00903483,  0.2312084 ], 
                      [0.1034271,  0.12923324, 0.03021851],
                      [0.00144255, 0.11346565, 0.20616764],
                      [0.13413355, -0.12223893,  0.25686872],
                      [0.04563733, -0.02232733,  0.0599547],
                      [0.02705727, 0.00031858, 0.05606366],
                      [0.02102337, 0.01389123, 0.07119013],
                      [-0.06336048,  0.12956764,  0.13416193],
                      [0.13908045, 0.06844391, 0.22935514],
                      [ 0.0972967,  -0.12839457,  0.21350014]])
    goals2 = np.array([[0.0767793,  0.03106248, 0.1473312], 
                      [ 0.0058055,  -0.09255553,  0.18336102],
                      [-0.03291276, 0.02742109,  0.02320863],
                      [-0.02892975,  0.09941293,  0.18178284],
                      [0.12782557, -0.14034954,  0.28461438],
                      [0.0074744,  0.11386087, 0.12034672],
                      [0.10534766, 0.05495432, 0.26455942],
                      [-0.03348396,  0.12122231,  0.01641202],
                      [0.07289395, -0.11697372,  0.26891217],
                      [0.02272988, 0.11841866, 0.22751188]])
    filename=f"tmp/residual_dense_reward_tanh"

    assistive_agent = Agent(n_actions=env.action_space.shape[0], input_dims=env.observation_space["observation"].shape[0],
                   gamma=0.99, gae_lambda=0.95,
                  lr_decay_rate=0.31622776601, lr_decay_freq=20000000,
                     chkpt_dir=filename)
    assistive_agent.load_models()

    player_agent = FrankaPandaJoystickActor(env)
    max_iters = 1000
    for ep in range(10):
        goal1 = goals1[ep]
        goal2 = goals2[ep]
        ob, info = env.reset(goal_val1 = np.array(goal1), goal_val2 = np.array(goal2))
        env.render()
        done = False
        reward = 0.0
        n_steps=0
        iters = 0
        print("Initial", ob["observation"][0:3])
        print("desired", ob["desired_goal1"][0:3])
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
