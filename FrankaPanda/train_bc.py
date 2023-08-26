""" Train the behavioral cloning model """
import torch as T
from behaviorcloning import BC_Agent
from joystick import FrankaPandaJoystickActor
import gymnasium as gym
import panda_gym

if __name__ == '__main__':
    env = gym.make('PandaReach-v3', render_mode="human")
    N = 20
    batch_size = 256
    n_epochs = 10
    alpha = 0.001
    print(env.observation_space["observation"].shape)
    agent = BC_Agent(n_actions=env.action_space.shape[0], 
                    input_dims=env.observation_space["observation"].shape, batch_size=batch_size,
                    n_epochs=n_epochs)
    # Create the joystick actor
    actor = FrankaPandaJoystickActor(env)
    # Train the behavioral cloning model
    model = agent.train_behavioral_cloning('observations.csv', 'actions.csv')
    agent.save_models()
    max_iters = 10000
    n_games = 10
    for i in range(n_games):
        observation, info = env.reset()
        env.render()  # Render the environment (optional)
        score = 0
        terminated= False
        truncated = False
        iters = 0
        while not terminated or not truncated:
            obs_tensor = T.tensor(observation["observation"], dtype=T.float32).to(agent.bc.device)
            action = agent.bc(obs_tensor)
            action = action.detach().cpu().numpy()
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            iters += 1
            if iters>max_iters:
                terminated= True
                truncated = True
        print('Game:', i+1, 'Score:', score)

    env.close()

