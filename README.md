# Shared Autonomy with Reinforcement Learnign
Author: Ritika Ghosh

https://github.com/GhoshRitika/LunarLanderRL/assets/60728026/944e5f9f-7647-4cbb-b00b-01b157ff75bc

## **Description**


## **Setup Guidelines**
The following packages are meant to be used with ROS 2 Humble.
1. Make sure ROS packages are most recent and up-to-date
```
sudo apt update
sudo apt upgrade
```
2. Install Moveit!: `sudo apt install ros-humble-moveit`
3. Create a new ROS2 workspace and enter it
```
mkdir -p copycatws/src 
cd copycatws
```
4. Install dependencies:
```
sudo apt-get install cmake gcc g++ libpopt-dev
pip install media pipe
sudo apt-get install python3-opencv
```
- Download, build, and install PCAN-USB driver for Linux: [libpcan](http://www.peak-system.com/fileadmin/media/linux/index.htm#download)
    ```
    tar -xzvf peak-linux-driver-x.x.tar.gz
    cd peak-linux-driver-x.x
    make NET=NO
    sudo make install
    sudo modprobe pcan
    ```
- Download, build, and install PCAN-Basic API for Linux: [libpcanbasic](https://www.peak-system.com/Software-APIs.305.0.html?&L=1)
    ```
    tar -xzvf PCAN_Basic_Linux-x.x.x.tar.gz
    cd PCAN_Basic_Linux-x.x.x/pcanbasic
    make
    sudo make install
    ```
- Download, build, and install Grasping Library for Linux, "libBHand": [Grasping_Library_for_Linux](http://wiki.wonikrobotics.com/AllegroHandWiki/index.php/Grasping_Library_for_Linux)
5. Download [this package](https://github.com/GhoshRitika/Allegro_hand) into the src directory: `git clone git@github.com:GhoshRitika/Allegro_hand.git`
6. Download my [gesture recognition](https://github.com/GhoshRitika/go1-gesture-command) fork in the same src directory: `git clone git@github.com:GhoshRitika/go1-gesture-command.git`

## **Contents**
Refer to README of individual packages for more detailed instructions and information.
1. 
2. 

## **Package Structure**
### Lunar Lander: 
**Joystick User Guide:**

**Train the Human Surrogate:**
**Train the Residual Policy:**
**Playing with the Assistive Agent:**


### Franka Panda:
**Joystick User Guide:**
**Train the Human Surrogate:**
**Train the Residual Policy:**
**Playing with the Assistive Agent:**

### For more detailed information: [this package](https://github.com/GhoshRitika/Allegro_hand)