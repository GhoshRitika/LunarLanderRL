import csv
import pandas as pd
import gym
from Joystick import LunarLanderJoystickActor

# def collect_expert_demonstrations(env, actor, num_episodes, csv_filename):
#     with open(csv_filename, 'w', newline='') as csvfile:
#         fieldnames = ['observation', 'action']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()

#         for episode in range(num_episodes):
#             obs = env.reset()
#             obs=obs[0]
#             env.render()
#             done = False
#             while not done:
#                 action = actor(obs)
#                 # obs_str = ', '.join(str(val) for val in obs)
#                 # action_str = ', '.join(str(val) for val in action)
#                 # writer.writerow({'observation': obs_str, 'action': action_str})
#                 writer.writerow({'observation': obs.tolist(), 'action': action.tolist()})
#                 obs, _, done, info, _ = env.step(action)
#         print(f"Expert data saved to {csv_filename}")

def collect_expert_demonstrations(env, actor, num_episodes, observations_csv, actions_csv):
    observations_data = {'observation': []}
    actions_data = {'action': []}

    for episode in range(num_episodes):
        obs = env.reset()
        obs = obs[0]
        env.render()
        done = False
        while not done:
            action = actor(obs)
            observations_data['observation'].append(obs.tolist())
            actions_data['action'].append(action.tolist())
            obs, _, done, info, _ = env.step(action)

    observations_df = pd.DataFrame(observations_data)
    actions_df = pd.DataFrame(actions_data)

    observations_df.to_csv(observations_csv, index=False)
    actions_df.to_csv(actions_csv, index=False)

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

    # Collect expert demonstrations using the joystick actor
    # collect_expert_demonstrations(env, actor, num_episodes, csv_filename)
    collect_expert_demonstrations(env, actor, num_episodes, 'observations.csv', 'actions.csv')

#     num_games = 10  # Number of games to play
#     expert_data_filename = 'expert_data.csv'  # Filename for the expert data CSV file

    # for game in range(num_episodes):
    #     ob = env.reset()
    #     env.render()
    #     done = False
    #     reward = 0.0

    #     while not done:
    #         ob, r, done, _ = env.step(actor(ob))
    #         reward += r

    #     print(f"Game {game+1}: Reward = {reward}")

    # env.close()

#     # Save the expert data to a CSV file
#     with open(expert_data_filename, 'w', newline='') as csvfile:
#         fieldnames = ['observation', 'action']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()

#         for data in actor.expert_data:
#             writer.writerow({'observation': data[0], 'action': data[1]})

#     print(f"Expert data saved to {expert_data_filename}")
