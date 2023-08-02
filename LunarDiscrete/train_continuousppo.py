import gym
import numpy as np
import torch as T
import os
from datetime import datetime
from constrained_residualPPO import Agent
from utils import plot_learning_curve, norm_action, add_actions, normalize_action
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v2")
    N = 128
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    n_games = 1000

    # timestamp = datetime.now().strftime("%Y%m%d%H%M")
    timestamp = f"{N}+{batch_size}+{n_epochs}"
    figure_file = f"plots/lunarlandercontinuous{timestamp}.png"
    filename=f"tmp/ppo{timestamp}"

    if not os.path.exists(filename):
        os.makedirs(filename)
        print(f"Directory created at {filename}")
    else:
        print(f"Directory already exists at {filename}")

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

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        observation=observation[0]
        done = False
        score = 0
        while not done:
            action_ass, prob, val = assistive_agent.choose_action(observation)
            # print("action:",action_ass)
            observation_, reward, done, info,_ = env.step(action_ass)
            n_steps += 1
            score += reward
            assistive_agent.remember(observation, action_ass, prob, val, reward, done)
            if n_steps % N == 0:
                assistive_agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        # assistive_agent.save_models()
        if avg_score > best_score:
            best_score = avg_score
            assistive_agent.save_models()
        else:
            print(f"Avg {avg_score} not better than best {best_score}")

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)

    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)