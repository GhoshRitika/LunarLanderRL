import gymnasium as gym
import panda_gym
import numpy as np
import torch as T
import os
from datetime import datetime
from Residual import Agent
from behaviorcloning import BC_Agent
from utils import plot_learning_curve, norm_action, add_actions, normalize_action
import matplotlib.pyplot as plt
from joystck import PIDController


if __name__ == '__main__':
    env = gym.make('PandaReachDense-v3')
    print("DENSE")
    N = 128 #2048
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    n_games = 10000

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    # figure_file = f"plots/FrankaReach_residual{timestamp}.png"
    # filename=f"tmp/residual{timestamp}"
    figure_file = f"plots/FrankaReach_residual_dense.png"
    filename=f"tmp/residual_dense_reward"

    if not os.path.exists(filename):
        os.makedirs(filename)
        print(f"Directory created at {filename}")
    else:
        print(f"Directory already exists at {filename}")

    user_input = input("Press 'y' to update previous model")
    if user_input.lower() == 'y':
        assistive_agent = Agent(n_actions=env.action_space.shape[0], input_dims=env.observation_space["observation"].shape[0],
                    gamma=0.99, gae_lambda=0.95,
                    lr_decay_rate=0.31622776601, lr_decay_freq=20000000,
                        chkpt_dir=f"tmp/ppoGPU")
        assistive_agent.load_models()
    else:
        assistive_agent = Agent(n_actions=env.action_space.shape[0], input_dims=env.observation_space["observation"].shape[0],
                    batch_size=256, policy_clip=0.2,
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

    player_agent = BC_Agent(n_actions=env.action_space.shape[0], 
                    input_dims=env.observation_space["observation"].shape)
    player_agent.load_models()
    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    max_iters = 10000

    # Define PID gains
    kp = 0.05  # Proportional gain
    ki = 0.001  # Integral gain
    kd = 0.3  # Derivative gain

    for i in range(n_games):
        observation, info = env.reset()
        done = False
        # terminated = False
        # truncated = False
        score = 0
        iters = 0
        pid_controller = PIDController(kp, ki, kd)
        while not done:
            current_position = observation["observation"][0:3]
            desired_position = observation["desired_goal"][0:3]

            error = desired_position - current_position

            # Compute control action using the PID controller
            action_ = pid_controller.compute_action(error)

            # ob_ = T.tensor(np.array([observation["observation"]]), dtype=T.float32).to(player_agent.bc.device)
            # action_ = player_agent.bc(ob_)
            # # action_player = norm_action(env, np.asarray(action_))
            # action_player = norm_action(env, action_.cpu().numpy())
            ob_ = T.tensor(np.array([observation["observation"]]), dtype=T.float32).to(assistive_agent.device)
            action_player = T.tensor(np.array([action_]), dtype=T.float32).to(assistive_agent.device)
            ob = (ob_, action_player)

            action_ass, prob, val = assistive_agent.choose_action(ob)
            # action_ass = normalize_action(np.asarray(action_ass), -3, 3)
            # action = add_actions(env, np.asarray(action_ass), np.asarray(action_))
            action = add_actions(env, action_ass, action_)

            observation_, reward, done, _, info = env.step(action[0])
            # if terminated or truncated:
            #     done = True
            n_steps += 1
            iters +=1
            score += reward
            #check if the combined action is saved or just the assistive action is
            assistive_agent.remember(observation["observation"], action_ass, prob, val, reward, done)
            if n_steps % N == 0:
                total, l_loss = assistive_agent.learn()
                learn_iters += 1
            observation = observation_
            if iters>max_iters:
                done = True
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        assistive_agent.save_models()
        if avg_score > best_score:
            best_score = avg_score
            # assistive_agent.save_models()
        else:
            print(f"Avg {avg_score} not better than best {best_score}")

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]

    plot_learning_curve(x, score_history, figure_file)