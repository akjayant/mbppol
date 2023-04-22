# MBPPO-Lagrangian
This repository contains code for the paper "[Model-based Safe Deep Reinforcement Learning via a Constrained Proximal Policy Optimization Algorithm](https://arxiv.org/abs/2210.07573)" accepted at NeurIPS 2022. Read paper [here](https://proceedings.neurips.cc/paper_files/paper/2022/file/9a8eb202c060b7d81f5889631cbcd47e-Paper-Conference.pdf).



    1) Requirements - 
        a) Python 3.7+
        b) PyTorch==1.10.0 and cuda11.3
        c) numpy==1.21.4
        d) gym==0.15.7 
        e) Hardware : Cuda supported GPU with atleast 4GB memory
    2) Install mujoco200 using https://roboti.us/download/mujoco200_linux.zip 
    3) Install Safety Gym using https://github.com/openai/safety-gym
    4) For reproducing results (upto same extent because of seed randomness) -
        a) Take backup of  /…/safety-gym/safety_gym/envs/suite.py 
        b) Copy ./src/env_suite_file/suite.py to above path. This removes "Vases" and increases "Hazards" from 10 to 15.
        c) Change ‘num_steps’ = 750’ in ‘DEFAULT’ dict of class Engine in  /…/safety-gym/safety_gym/envs/engine.py 
        d) Run for 8 random seeds :
            i) cd src
            ii) python3  mbppo_lagrangian.py –exp_name=”experiment_name” –seed=0 –env=”<environment_name>” –beta=0.02

	Where environment names are [Safexp-PointGoal2-v0,Safexp-CarGoal2-v0]

    5) Use https://github.com/openai/safety-starter-agents/blob/master/scripts/plot.py for plotting -  
	a) python plot.py –logdir=’<path to data>’’ --value=<plot_choice>
	
	Where plot_choice are ‘AverageEpRet’ for reward performance, ‘AverageEpCost’ for cost performance.  

If you are using this in your work, please cite using :


@inproceedings{NEURIPS2022_9a8eb202,
 author = {Jayant, Ashish K and Bhatnagar, Shalabh},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {24432--24445},
 publisher = {Curran Associates, Inc.},
 title = {Model-based Safe Deep Reinforcement Learning via a Constrained Proximal Policy Optimization Algorithm},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/9a8eb202c060b7d81f5889631cbcd47e-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
}
