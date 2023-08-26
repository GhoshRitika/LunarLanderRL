""" Collects expert demonstrations using a PID controller."""
import pandas as pd
import gym
from joystick import FrankaPandaJoystickActor, PIDController 

def collect_expert_demonstrations(env, actor, num_episodes, observations_csv, actions_csv):
    """ Collects expert demonstrations using a PID controller."""
    observations_data = {'observation': []}
    actions_data = {'action': []}
    user_input = input("Press 'y' to append data to existing files or any other key to create new files: ")
    if user_input.lower() == 'y':
        mode = 'a'
        flag=False
    else:
        mode = 'w'
        flag = True
    kp = 0.05  # Proportional gain
    ki = 0.001  # Integral gain
    kd = 0.3  # Derivative gain

    max_iters = 10000

    for episode in range(num_episodes):
        print("Episode number: ", episode)
        observation, info = env.reset()
        score = 0
        terminated = False
        truncated = False
        iters = 0
        print("Initial", observation["observation"][0:3])
        print("desired", observation["desired_goal"][0:3])

        pid_controller = PIDController(kp, ki, kd)

        while not terminated:
            current_position = observation["observation"][0:3]
            desired_position = observation["desired_goal"][0:3]

            error = desired_position - current_position

            # Compute control action using the PID controller
            action = pid_controller.compute_action(error)
            observations_data['observation'].append(observation["observation"].tolist())
            actions_data['action'].append(action.tolist())
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            iters += 1
            if iters > max_iters:
                terminated = True
                truncated = True
        print(f"score: {score}")

    env.close()
    observations_df = pd.DataFrame(observations_data)
    actions_df = pd.DataFrame(actions_data)

    observations_df.to_csv(observations_csv, index=False, mode=mode, header=flag)
    actions_df.to_csv(actions_csv, index=False, mode=mode, header=flag)

    print(f"Observations data saved to {observations_csv}")
    print(f"Actions data saved to {actions_csv}")

if __name__ == '__main__':
    import gymnasium as gym
    import panda_gym

    env = gym.make('PandaReach-v3', render_mode="human")

    # Create the joystick actor
    actor = FrankaPandaJoystickActor(env)
    # Define the number of expert episodes and the CSV filename
    num_episodes = 1000
    collect_expert_demonstrations(env, actor, num_episodes, 'observations.csv', 'actions.csv')

