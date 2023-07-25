import gym
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from behaviorcloning import BC_Agent

def train_rl_with_residual_policy(env, residual_policy, human_policy, loss_theta, loss_lambda,
                                  alpha_theta, alpha_lambda, rollout_length, num_episodes):
    # Initialize policy parameters
    theta = residual_policy.parameters()
    lambda_ = human_policy.parameters()

    # Initialize optimizers
    optimizer_theta = optim.SGD(theta, lr=alpha_theta)
    optimizer_lambda = optim.SGD(lambda_, lr=alpha_lambda)

    for episode in range(num_episodes):
        state = env.reset()  # Assuming 'env' is the RL environment
        total_reward = 0.0

        for t in range(rollout_length):
            # Sample action from the residual policy
            action_h = human_policy(state)

            # Sample action from the base policy
            action_r = residual_policy(state, action_h)

            # Execute action in the environment
            next_state, reward, done, _ = env.step(action_h + action_r)
            total_reward += reward

            # Estimate the losses using the collected trajectory
            # Assuming `loss_theta` and `loss_lambda` take state, action_h, action_r, reward, and next_state as inputs
            loss_theta_value = loss_theta(state, action_h, action_r, reward, next_state)
            loss_lambda_value = loss_lambda(state, action_h, action_r, reward, next_state)

            # Update the policies using gradient descent/ascent
            optimizer_theta.zero_grad()
            loss_theta_value.backward()
            optimizer_theta.step()

            optimizer_lambda.zero_grad()
            (-loss_lambda_value).backward()  # Using gradient ascent for lambda
            optimizer_lambda.step()

            state = next_state

            if done:
                break

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    # After training, you can use the learned policies for inference
    residual_policy.eval()
    human_policy.eval()

# Example usage:
# base_policy and residual_policy are instances of PyTorch nn.Module representing the base and residual policies, respectively.
# loss_theta and loss_lambda are the loss functions for updating the base and residual policies, respectively.
# alpha_theta and alpha_lambda are the learning rates for updating the base and residual policies, respectively.
# rollout_length is the number of steps in each episode.
# num_episodes is the total number of episodes for training.
env = gym.make("LunarLanderContinuous-v2", render_mode="human")
bc_agent = BC_Agent(n_actions=env.action_space.shape[0], 
                    input_dims=env.observation_space.shape)
bc_agent.load_models()
train_rl_with_residual_policy(residual_policy, bc_agent, loss_theta, loss_lambda,
                              alpha_theta=0.01, alpha_lambda=0.01,
                              rollout_length=128, num_episodes=1000)