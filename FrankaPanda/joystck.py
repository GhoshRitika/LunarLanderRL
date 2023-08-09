"""Franka Panda Joystick Agent."""
import pygame
import numpy as np
import time

#####################################
# Change these to match your joystick
R_UP_AXIS = 4 # Z
R_SIDE_AXIS = 3 # X
L_SIDE_AXIS = 0 # Y
#####################################


class FrankaPandaJoystickActor(object):
    """Joystick Controller for Franka Arm."""

    def __init__(self, env, fps=60):
        """Init."""
        # if env.num_envs > 1:
        #     raise ValueError("Only one env can be controlled with the joystick.")
        self.env = env
        self.human_agent_action = np.array([[0., 0., 0.]], dtype=np.float32)
        pygame.joystick.init()
        joysticks = [pygame.joystick.Joystick(x)
                     for x in range(pygame.joystick.get_count())]
        if len(joysticks) != 1:
            raise ValueError("There must be exactly 1 joystick connected."
                             f"Found {len(joysticks)}")
        self.joy = joysticks[0]
        self.joy.init()
        pygame.init()
        self.t = None
        self.fps = fps

    def _get_human_action(self):
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                if event.axis == R_SIDE_AXIS:
                    self.human_agent_action[0, 0] = -1.0 * event.value
                elif event.axis == R_UP_AXIS:
                    self.human_agent_action[0, 2] = -1.0 * event.value
                elif event.axis == L_SIDE_AXIS:
                    self.human_agent_action[0, 1] = -1.0 * event.value
        if abs(self.human_agent_action[0, 0]) < 0.1:
            self.human_agent_action[0, 0] = 0.0
        if abs(self.human_agent_action[0, 1]) < 0.1:
            self.human_agent_action[0, 1] = 0.0
        if abs(self.human_agent_action[0, 2]) < 0.1:
            self.human_agent_action[0, 2] = 0.0
        return self.human_agent_action

    def __call__(self, ob):
        """Act."""
        self.env.render()
        action = self._get_human_action()
        if self.t and (time.time() - self.t) < 1. / self.fps:
            st = 1. / self.fps - (time.time() - self.t)
            if st > 0.:
                time.sleep(st)
        self.t = time.time()
        # print(action)
        return action[0]

    def reset(self):
        self.human_agent_action[:] = 0.

if __name__ == '__main__':
    import gymnasium as gym
    # from stable_baselines.common.cmd_util import make_vec_env
    import panda_gym

    env = gym.make('PandaReach-v3', render_mode="human")
    # env = make_vec_env(
    #     'PandaReach-v3',
    #     "robotics",
    #     1, 0, flatten_dict_observations=False, env_kwargs={"render": True})
    actor = FrankaPandaJoystickActor(env)
    print(env.action_space)
    print(env.observation_space)
    max_iters = 100
    for _ in range(10):
        observation, info = env.reset()
        score = 0
        terminated= False
        truncated = False
        iters = 0
        print("Initial", observation["observation"][0:3])
        print("desired", observation["desired_goal"][0:3])
        while not terminated or not truncated:
            # print(observation)
            # action = env.action_space.sample() # random action
            # current_position = observation["observation"][0:3]
            # desired_position = observation["desired_goal"][0:3]
            # action = 0.05*(desired_position - current_position)
            action = actor(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            iters += 1
            if iters>max_iters:
                terminated= True
                truncated = True
            # print("d", np.linalg.norm(observation["observation"][0:3] - observation["desired_goal"][0:3], axis=-1))
        print(f"score: {score}")

    env.close()