"""Franka Panda Joystick Agent."""
import pygame
import numpy as np
import time
import matplotlib.pyplot as plt

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

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute_action(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        action = self.kp * error + self.ki * self.integral + self.kd * derivative
        return action

if __name__ == '__main__':
    import gymnasium as gym
    # from stable_baselines.common.cmd_util import make_vec_env
    import panda_gym
    goal_his = np.array([[-0.09425813, -0.05269751, 0.07325654], 
                      [0.00141661, 0.00123507, 0.14757843],
                      [-0.00135891, -0.08630279, 0.04110381],
                      [ 0.07980205, -0.01122603, 0.11293837],
                      [-0.13518204,  0.13425152, 0.08225469],
                      [0.07082976, 0.13334234, 0.083761],
                      [-0.11354388, 0.13814254, 0.05822545],
                      [0.01594417, 0.07936273, 0.22025128],
                      [ 0.09835389, -0.07078907,  0.17652114],
                      [-0.00091254,  0.05386124,  0.28241307]])
    score_his = [-234.2966771274805, -150.35052871331573, -212.95436526089907, -149.89449425041676,
                 -194.23260818049312, -130.69264344871044, -144.02622505649924, -35.998916912823915, -20.194139402359724, -162.23665458709002]
    score_results = [-1260.0, -15.0, -340.0, -792.0, -1280.0, -109.0, -438.0, -61.0, -170.0, -286.0]
    goals = []
    env = gym.make('PandaReachDoubleDense-v3', render_mode="human", goal_random=True)
    actor = FrankaPandaJoystickActor(env)
    max_iters = 1000
    score_history = []
    for _ in range(10):
        observation, info = env.reset() #goal_val=np.array([-0.10652074, 0.00213265, 0.19745056])
        score = 0
        terminated= False
        truncated = False
        iters = 0
        print("Initial", observation["observation"][0:3])
        print("desired", observation["desired_goal1"][0:3])
        print("desired", observation["desired_goal2"][0:3])
        goals.append(observation["desired_goal1"][0:3])
        while not terminated or not truncated:
            action = actor(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            score += reward
            iters += 1
            if iters>max_iters:
                terminated= True
                truncated = True
        print(f"score: {score}")
        score_history.append(score)

    env.close()
    x = [i+1 for i in range(len(score_history))]
    plt.plot(x, score_history, color='darkblue')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Score of Lunar Lander')
    plt.savefig(f"plots/scores2")
    plt.clf()

    # # Define PID gains
    # kp = 0.05  # Proportional gain
    # ki = 0.001  # Integral gain
    # kd = 0.3  # Derivative gain

    # env = gym.make('PandaReachDense-v3', render_mode="human", goal_random=False)
    # # env = gym.make('PandaReachDouble-v3', render_mode="human", goal_random=False)
    # # actor = FrankaPandaJoystickActor(env)
    # max_iters = 10000

    # for _ in range(10):
    #     observation, info = env.reset()
    #     score = 0
    #     terminated = False
    #     truncated = False
    #     iters = 0
    #     print("Initial", observation["observation"][0:3])
    #     print("desired", observation["desired_goal"][0:3])

    #     pid_controller = PIDController(kp, ki, kd)

    #     while not terminated:
    #         current_position = observation["observation"][0:3]
    #         desired_position = observation["desired_goal"][0:3]

    #         error = desired_position - current_position

    #         # Compute control action using the PID controller
    #         action = pid_controller.compute_action(error)

    #         observation, reward, terminated, truncated, info = env.step(action)
    #         score += reward
    #         iters += 1
    #         if iters > max_iters:
    #             terminated = True
    #             truncated = True
    #         # print(iters)
    #     print(f"score: {score}")

    # env.close()