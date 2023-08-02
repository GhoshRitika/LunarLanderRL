import pandas as pd
import gym
from Joystick import LunarLanderJoystickActor

def collect_expert_demonstrations(env, actor, num_episodes, observations_csv, actions_csv):
    observations_data = {'observation': []}
    actions_data = {'action': []}
    user_input = input("Press 'y' to append data to existing files or any other key to create new files: ")
    if user_input.lower() == 'y':
        mode = 'a'
        flag=False
    else:
        mode = 'w'
        flag = True
    for episode in range(num_episodes):
        print("Episode number: ", episode)

        obs = env.reset()
        obs = obs[0]
        env.render()
        done = False
        reward = 0.0
        while not done:
            action = actor(obs)
            observations_data['observation'].append(obs.tolist())
            actions_data['action'].append(action.tolist())
            obs, r, done, info, _ = env.step(action)
            reward += r
        print(reward)

    observations_df = pd.DataFrame(observations_data)
    actions_df = pd.DataFrame(actions_data)

    observations_df.to_csv(observations_csv, index=False, mode=mode, header=flag)
    actions_df.to_csv(actions_csv, index=False, mode=mode, header=flag)

    print(f"Observations data saved to {observations_csv}")
    print(f"Actions data saved to {actions_csv}")

if __name__ == '__main__':
    # Create the LunarLander environment
    env = gym.make("LunarLanderContinuous-v2", render_mode="human")

    # Create the joystick actor
    actor = LunarLanderJoystickActor(env)
    # Define the number of expert episodes and the CSV filename
    num_episodes = 20
    csv_filename = 'expert_demonstrations.csv'

    collect_expert_demonstrations(env, actor, num_episodes, 'observations.csv', 'actions.csv')

