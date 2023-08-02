import gym
import numpy as np
import torch as T
import os
from datetime import datetime
from continuous_ppo import Agent
from behaviorcloning import BC_Agent
from utils import plot_learning_curve, norm_action, add_actions, normalize_action
import matplotlib.pyplot as plt


if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v2")
    # print(env.action_space.shape)
    # print(env.observation_space.shape)
    N = 128 #2048
    batch_size = 64
    n_epochs = 10
    alpha = 0.0003
    n_games = 10000

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    figure_file = f"plots/lunarlandercontnuous_residual{timestamp}.png"
    filename=f"tmp/residual{timestamp}"

    if not os.path.exists(filename):
        os.makedirs(filename)
        print(f"Directory created at {filename}")
    else:
        print(f"Directory already exists at {filename}")

    user_input = input("Press 'y' to update previous model")
    if user_input.lower() == 'y':
        assistive_agent = Agent(n_actions=env.action_space.shape[0], input_dims=env.observation_space.shape[0],
                    gamma=0.99, gae_lambda=0.95,
                    lr_decay_rate=0.31622776601, lr_decay_freq=20000000,
                        chkpt_dir=f"tmp/ppoGPU")
        assistive_agent.load_models()
    else:
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

    player_agent = BC_Agent(n_actions=env.action_space.shape[0], 
                    input_dims=env.observation_space.shape)
    player_agent.load_models()
    best_score = env.reward_range[0]
    score_history = []
    loss_total = []
    loss_l = []
    action_h = np.empty((0, env.action_space.shape[0]), dtype=float)
    action_r = np.empty((0, env.action_space.shape[0]), dtype=float)
    actions = np.empty((0, env.action_space.shape[0]), dtype=float)

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    max_iters = 10000

    for i in range(n_games):
        # print(i)
        observation = env.reset()
        observation=observation[0]
        done = False
        score = 0
        iters = 0
        while not done:
            ob_ = T.tensor(np.array([observation]), dtype=T.float32).to(player_agent.bc.device)
            action_ = player_agent.bc(ob_)
            # action_player = norm_action(env, np.asarray(action_))
            action_player = norm_action(env, action_.cpu().numpy())

            action_player = T.tensor(action_player, dtype=T.float32).to(player_agent.bc.device)
            ob = (ob_, action_player)

            action_ass, prob, val = assistive_agent.choose_action(ob)
            # action_ass = normalize_action(np.asarray(action_ass), -3, 3)
            # action = add_actions(env, np.asarray(action_ass), np.asarray(action_))
            action = add_actions(env, action_ass, action_.cpu().numpy())

            observation_, reward, done, info,_ = env.step(action[0])
            n_steps += 1
            iters +=1
            # assistive_agent.t += 1
            score += reward
            # print(action_.cpu().numpy()[0])
            action_h = np.vstack((action_h, action_.cpu().numpy()[0]))
            action_r = np.vstack((action_r, action_ass[0]))
            actions = np.vstack((actions, action[0]))
            #check if the combined action is saved or just the assistive action is
            assistive_agent.remember(observation, action_ass, prob, val, reward, done)
            if n_steps % N == 0:
                total, l_loss = assistive_agent.learn()
                # print(l_loss)
                loss_total.append(total.detach().cpu().numpy())
                loss_l.append(l_loss.detach().cpu().numpy())
                # assistive_agent.t += N
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
    # y = [i+1 for i in range(learn_iters)]

    # plt.plot(y, loss_total, label='total loss', color='darkblue')
    # plt.plot(y, loss_l, label='lambda loss', color='darkgreen')
    # plt.xlabel('Learning Iterations')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.title('Comparison of losses')
    # plt.savefig(f"plots/losses")

    plot_learning_curve(x, score_history, figure_file)
