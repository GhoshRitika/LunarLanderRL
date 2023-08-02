import gym
from continuous_ppo import Agent

if __name__ == '__main__':
    env = gym.make("LunarLanderContinuous-v2", render_mode="human")
    filename= 'tmp/ppo20230726134213'
    agent = Agent(n_actions=env.action_space.shape[0], input_dims=env.observation_space.shape[0],
                     chkpt_dir=filename)
    agent.load_models()  # Load the saved models

    n_games = 10

    for i in range(n_games):
        observation = env.reset()
        observation = observation[0]
        env.render()  # Render the environment (optional)
        done = False
        score = 0

        while not done:
            action, _, _ = agent.choose_action(observation)
            observation, reward, done, info, _ = env.step(action)
            score += reward

        print('Game:', i+1, 'Score:', score)

    env.close()