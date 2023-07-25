import gym
import numpy as np
import os
from datetime import datetime
from continuous_ppo import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v2")
    # print(env.action_space.shape)
    N = 128 #2048
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    n_games = 1000

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    figure_file = f"plots/lunarlandercontnuous{timestamp}.png"
    filename=f"tmp/ppo{timestamp}"

    if not os.path.exists(filename):
        os.makedirs(filename)
        print(f"Directory created at {filename}")
    else:
        print(f"Directory already exists at {filename}")
    agent = Agent(n_actions=env.action_space.shape[0], input_dims=env.observation_space.shape[0],
                   gamma=0.99, gae_lambda=0.95,
                  lr_decay_rate=0.31622776601, lr_decay_freq=20000000,
                     chkpt_dir=filename)

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        print(i)
        observation = env.reset()
        observation=observation[0]
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info,_ = env.step(action)
            n_steps += 1
            agent.t += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)