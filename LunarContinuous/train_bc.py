""" Train the behavioral cloning model and test it on the LunarLanderContinuous-v2 environment."""
import gym
import numpy as np
import torch as T
from behaviorcloning import BC_Agent
from collect_data import collect_expert_demonstrations
from utils import plot_learning_curve
from joystick import LunarLanderJoystickActor

if __name__ == '__main__':
    # Create the LunarLander environment
    env = gym.make("LunarLanderContinuous-v2", render_mode="human")
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.001
    agent = BC_Agent(n_actions=env.action_space.shape[0], 
                    input_dims=env.observation_space.shape, batch_size=batch_size,
                    n_epochs=n_epochs)
    # Create the joystick actor
    actor = LunarLanderJoystickActor(env)
    # Train the behavioral cloning model
    model = agent.train_behavioral_cloning('observations.csv', 'actions.csv')
    agent.save_models()

    n_games = 10
    for i in range(n_games):
        observation = env.reset()
        observation = observation[0]
        env.render()  # Render the environment (optional)
        done = False
        score = 0

        while not done:
            obs_tensor = T.tensor(observation, dtype=T.float32).to(agent.bc.device)
            action = agent.bc(obs_tensor)
            action = action.detach().cpu().numpy()
            observation, reward, done, info, _ = env.step(action)
            score += reward

        print('Game:', i+1, 'Score:', score)

    env.close()

