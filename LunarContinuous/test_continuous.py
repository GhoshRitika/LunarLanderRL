import gym
import torch as T
import numpy as np
from constrained_residualPPO import Agent
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v2", render_mode="human")
    # print(env.state_dict())

    max_iters = 1000

    filename=f"tmp/ppoContinuous_avg120"

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
    assistive_agent.load_models()
    
    for ep in range(10):
        ob = env.reset()
        ob=ob[0]
        env.render()
        done = False
        reward = 0.0
        iters = 0
        actions = np.empty((0, env.action_space.shape[0]), dtype=float)
        while not done:
            action_ass, prob, val = assistive_agent.choose_action(ob)
            ob, r, done, info,_ = env.step(action_ass)
            reward += r
            iters += 1
            actions = np.vstack((actions, action_ass))
            # print(actions.shape, iters)
            if iters>max_iters:
                done = True
        y = [i+1 for i in range(iters)]
        plt.clf()
        plt.plot(y, actions[:,0], label='action[0]', color='crimson')
        plt.plot(y, actions[:,1], label='action[1]', color='salmon')
        plt.xlabel('Iterations')
        plt.ylabel('Action Values')
        plt.legend()
        plt.title('Comparison of Actions')
        plt.savefig(f"plots/results{ep}")
        print("reward:", reward)

    env.close()
