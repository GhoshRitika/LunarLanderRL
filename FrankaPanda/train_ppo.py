import gymnasium as gym
import numpy as np
import torch as T
import os
from datetime import datetime
from PPO import Agent
from utils import plot_learning_curve
import panda_gym


if __name__ == '__main__':
    # vectorize the environnement
    env = gym.make('PandaReach-v3')
    N = 128
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    n_games = 100

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    figure_file = f"plots/FrankaReach{timestamp}.png"
    filename=f"tmp/ppo{timestamp}"

    if not os.path.exists(filename):
        os.makedirs(filename)
        print(f"Directory created at {filename}")
    else:
        print(f"Directory already exists at {filename}")

    assistive_agent = Agent(n_actions=env.action_space.shape[0], input_dims=env.observation_space["observation"].shape[0],
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
    loss_total = []
    loss_l = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    max_iters = 10000

    for i in range(n_games):
        observation, info = env.reset()
        done = False
        # terminated = False
        # truncated = False
        score = 0
        iters = 0 
        while not done:
            # print(observation)
            action_ass, prob, val = assistive_agent.choose_action(observation["observation"])
            # print("action:",action_ass)
            observation_, reward, done, _, info = env.step(action_ass)
            # if terminated or truncated:
            #     done = True
            # if done: print("Done")
            n_steps += 1
            iters +=1
            score += reward
            assistive_agent.remember(observation["observation"], action_ass, prob, val, reward, done)
            if n_steps % N == 0:
                total = assistive_agent.learn()
                # loss_total.append(total.detach().cpu().numpy())
                # assistive_agent.t += N
                learn_iters += 1
            observation = observation_
            if iters>max_iters:
                done = True
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