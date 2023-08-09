import gymnasium as gym
import torch as T
import numpy as np
from PPO import Agent
import matplotlib.pyplot as plt
import panda_gym

if __name__ == '__main__':
    env = gym.make('PandaReach-v3', render_mode="human")
    max_iters = 1000

    filename=f"tmp/ppo202308082059"

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
    assistive_agent.load_models()
    
    for ep in range(10):
        ob, info = env.reset()
        env.render()
        done = False
        terminated = False
        truncated = False
        reward = 0.0
        iters = 0
        actions = np.empty((0, env.action_space.shape[0]), dtype=float)
        print("Initial", ob["observation"][0:3])
        print("desired", ob["desired_goal"][0:3])
        while not done:
            action_ass, prob, val = assistive_agent.choose_action(ob["observation"])
            ob, r, terminated, truncated, info = env.step(action_ass)
            if terminated or truncated:
                done = True
            reward += r
            iters += 1
            actions = np.vstack((actions, action_ass))
            # print(actions.shape, iters)
            if iters>max_iters:
                done = True
        y = [i+1 for i in range(iters)]
        plt.clf()
        plt.plot(y, actions[:,0], label='action[0]', color='lightblue')
        plt.plot(y, actions[:,1], label='action[1]', color='lightgreen')
        plt.plot(y, actions[:,2], label='action[2]', color='salmon')
        plt.xlabel('Iterations')
        plt.ylabel('Action Values')
        plt.legend()
        plt.title('Comparison of Actions')
        plt.savefig(f"plots/results{ep}")
        print("reward:", reward)

    env.close()
