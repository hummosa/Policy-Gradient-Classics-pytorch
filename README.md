# Policy-Gradient-Classics-pytorch

This repository is an implementation of the reinforcement learning algorithm DDPG as described by [Lillicrap et. al](https://arxiv.org/abs/1509.02971).

#### Unity [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment:

The environment simulates an arm with two joints. The goal of the task is to keep the tip of the arm inside a spherical target region, rotating around the arm. The environment provides a reward of +0.1 for each time step the agent maintains its tip within the spherical target region. The state of the environment is represented in a vector of 33 values representing the position and velocities of the arm segments.  

The environment is continuous rather than episodic but can easily be sampled to a desired episode length without information loss as it’s a generally stationary environment. The action space is continuous, which constrain our choice of algorithm to solve it. The environment is considered solved when achieving a score of >30 on 100 consecutive episodes over all 20 agents. This project uses the 20 -agent version of the environment.  


To run the enviornment locally a few required packages are necessary.

1) Udacity provides [this github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) with many of the dependencies required to run the environment. Follow the instructions in the readme.md file under thea heading 'dependencies' to clone a copy and install the required packages. 
     * Note also that it will be required to install ipykernel (pip install ipykernel) to execute all the steps.

2) Installing Unity enviroments is an option, but it will be easier to download only the required executable provided by Udacity. [The Windows 64bit compabtible version](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip) is included here for reference, but other versions can be found on the Udacity github repository.

3) To run the enviroment locally additional Unity packages are required as detailed [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation-Windows.md). 
    * **Note:** it is also important to run 'pip install unityagents' which is not mentioned in the instructions.
    
Training and demo:
The file continuous_control_ddpg.py is the main python file used to load the environment, instantiate and train and agent. The function play_round can be used to demonstrate a learned agent.

Weight files are also included for a fully trained agent.
Here are the average scores over training episodes:

![Training curve](https://raw.githubusercontent.com/hummosa/Policy-Gradient-Classics-pytorch/master/DDPG_training_scores.png)
    :scale: 50 %
