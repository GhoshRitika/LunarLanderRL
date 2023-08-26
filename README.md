# Shared Autonomy with Reinforcement Learning
Author: Ritika Ghosh

https://github.com/GhoshRitika/LunarLanderRL/assets/60728026/4acf7e26-f334-4344-9ce5-b0fe06e996d8

## **Description**
This project allows for human robot collaboration with the help of an assistive agent which minimally adjusts the human actions to improve the task performance without any prior knowledge or restrictive assumptions about the environment, goal space or human policy. This is an adaptation of model free
constrained residual policy using proximal policy optimization for shared control. It has been tested on Lunar Lander and Franka Reach environments.

## **Setup Guidelines**
The following are the requirements to manually setup the project else `pip install -r requirements.txt`:
1. Connect to xbox 360 controller.
2. Make sure all existing packages are most recent and up-to-date
```
sudo apt updatels
sudo apt upgrade
```
2. Install dependencies:
```
pip install torch torchvision
pip install tensorflow-gpu==1.15.0
pip install gin-config
pip install PyOpenGL 
pip install pygame PyOpenGL_accelerate
pip install pandas
```
3. Install Lunar Lander environment, OpenAI Gym:
```
pip install gym
pip install gym[box2d]
```
4. Install Franka Reach environment, panda-gym: 
```
git clone https://github.com/GhoshRitika/panda-gym
pip install -e panda-gym
```
5. Download [this package](https://github.com/GhoshRitika/LunarLanderRL) into the current directory: `git clone git@github.com:GhoshRitika/LunarLanderRL.git`


## **Contents**
This project is divided into 2 sections for each of the training environments.
1. `LunarContinuous`: This section contains the code for joystick control as well as training and testing the assistive agent for the Lunar Lander Continuous environment.
2. `FrankaPanda`: This section contains the code for joystick control as well as training and testing the assistive agent for the Franka Reach environment.

## **Package Structure**
### Lunar Lander: 
**Joystick User Guide:**
The Lunar Lander environment allows control over its 3 thrusters, the main thruster along with its 2 lateral thrusters.
    - Right Joystick Vertical Motion -> Main Thruster
    - Right Joystick Horizontal Motion -> Lateral Thrusters
In order to play without assistance of the agent:
```
cd LunarContinuous
python3 joystick.py
```
**Train the Human Surrogate:**
Collect data for the human surrogate run `python3 collect_data.py` and to train the behavioral cloning model
run `python3 train_bc.py`. If you want to visualize the human surrogate policy performance run `python3 test_bc.py`.

**Train the Residual Policy:**
After training the human agent, the Residual policy can be trained by running `python3 train_ppo.py`. The resulting policy can be tested by running `python3 test_residual.py`

### Franka Panda:
**Joystick User Guide:**
This project uses the IK controller to control the x, y & z position of the end effector. 
    - Right Joystick Vertical Motion -> Z axis
    - Right Joystick Horizontal Motion -> X axis
    - Left Joystick Horizontal Motion -> Y axis
In order to play without assistance of the agent:
```
cd FrankaPanda
python3 joystick.py
```
**Train the Human Surrogate:**
Collect data for the human surrogate run `python3 collect_data.py` and to train the behavioral cloning model
run `python3 train_bc.py`. If you want to visualize the human surrogate policy performance run `python3 test_bc.py`.
NOTE: For the sake of getting expert data to train on a PID control loop was used as the human surrage in this case.

**Train the Residual Policy:**
After training the human agent, the Residual policy can be trained by running `python3 train_residual.py`. The resulting policy can be tested by running `python3 test_residual.py`.

**Assistive Agent with Multiple Goals:**
To train the the Residual policy with two goals, where one goal has greater penalties, run `python3 train_residual_double.py`. The resulting policy can be tested by running `python3 test_double.py`

### For more detailed information check out [this link](https://ghoshritika.github.io/SharedAutonomy.html)