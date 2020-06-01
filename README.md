# Implementation Practice: Proximal Policy Optimization

see this [[README writing tutorial](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)]

> Project description paragraph

## PPO Structure

Proximal Policy Optimization is a default method in reinforcement learning. TRPO (Trust Region Policy Optimization) improves the VPG (Vanilla Policy Gradient) method by optimizing the policy update scale, PPO is an improved version of TRPO, which is easier to implement and tune.

The original paper: [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf) 

The algorithm updates the loss computed from collected policy rollouts, then train actor-network and critic-network together.

The loss function in PPO constructed with three parts: policy loss, critic loss, entropy loss. Where policy loss shows how policy is close to the optimal state. The critical loss and entropy loss indicate critic network precision and policy randomness.
$$
\begin{split}
&L_{policy}(\pi) = Advantage_{\pi}(s) \cdot \frac{\pi_{new}(s)}{\pi_{old}(s)} \\
&L_{critic}(\pi) = MSE(critic(s), reward(s)) \\
&L_{entro}(\pi) = entropy(\pi)
\end{split}
$$
Where $\pi_{new}(s) / \pi_{old}(s)$ also notated as $ratio_{\pi}$

People store policy rollouts from last iteration to calculate $Advantage_{\pi}$, $ratio_{\pi}$ and $L_{critic}(\pi)$ in current iteration.

### PPO Code Structure

PPO use a Actor-Critic style algorithm and do policy update as vanilla policy gradient method.

![algorithm](.\readme_data\image\algorithm.png)

Based on precious instruction, the PPO code should be like this:

![code_structure](.\readme_data\image\code_structure.png)

## Getting Started

> Instruction of how to download and test this code base

Make sure your machine is using cuda, then clone the code base to local:

> ```python
> git clone https://github.com/ziangqin-stu/impl_ppo.git
> ```

Install necessary python packages:

> ```python
> pip install -r requirements.txt
> ```

Run a simple experiment (cart pole) to check every thing going well:

> ```python
> python ppo.py --param_id=0
> ```

## Run Experiments

### Command Line Interface

Reach training settings and hyperparameters separately using command-line arguments:

* Default arguments are provided in `training_param.csv`, you can specify the parameter set by selecting parameter set id with command-line argument `--param_id`.

* Command-line interface can control all parameters. Default argument set provides baseline experiment settings, then customize the parameters using other command-line arguments. You can also add your baseline parameters too.

Test: run an experiment using small number of rollout worker:

> ```python
> python ppo.py --param_id=1 --envs_num=5
> ```

### Training Loggers

This implementation uses `tensorboard` as training monitoring tool. Logged data saved in `./run/` folder.

You can also check the sampled episode videos in folder `./save/video/ `. The video logging feature is controlled by two command line arguments,  their default values are:

> ```python
> --log_video=True
> --plotting_iters=20
> ```

Where `log_video` specifies whether saving videos during training, `plotting_iters` specifies the interval (by iteration number) between each two sampled videos. Videos will be saved in folders named by the training prefix and iteration number.

### Resume Model

This implementation also offers a feature of saving and loading training checkpoints. This feature helps to continue previous training and restore the running environment to debug issues that occur after many iterations.

Saving & loading are also controlled by four command line arguments,  their default values are:

> ```python
> --save_checkpoint=True
> --checkpoint_iter=100
> --use_pretrain=False
> --pretrain_file=""
> ```

Where `save_checkpoint` specifies whether saving checkpoints automatically during training, `checkpoint_iter` specifies the interval (by iteration number) between each two checkpoints. `use_pretrain` specifies whether algorithm starts with loading a checkpoint file from local folder, `pretrain_file` is the file name (not include the file path) to load. Checkpoint files are saved in `./save/medoel` folder. 

Checkpoints features are built with `torch.save()` and `torch.load()`, local files are formatted as `.tar` file.

Test:

> ```python
> python ppo.py --horizon=5 --prefix=test_pretrain
> python ppo.py --use_pretrain=True --pretrain_file=test_pretrain_iter_5.tar
> ```

You can see prompts in command line showing the loading and saving process works well.

## Training Result

### Discrete Environment:

#### CartPole-v1

One of the simplest environment in classical control. Training converges quickly.

`[missing training curve]`

### Continuous Environment:

#### InvertedPendulum-v2

The simplest environment in MuJoCo environments. Using continuous action space. The maximum episode length during training is 200 steps, the maximum episode steps of environments is 1,000. Algorithm learns quickly, training converges within 5 minutes.

<div style="display:flex;">
    <div style="display:flex; margin:auto;">
        <img src=".\readme_data\image\demo_invertedpendulum_episode.png" alt="reward curve" width="400" style="padding: 5px;"/>
        <img src=".\readme_data\image\demo_invertedpendulum_reward(time).png" alt="reward curve" width="400" style="padding: 5px;"/>
    </div>
</div>

#### Hopper-v2

A more complex environment than InvertedPendulum-v2. During training, agents first learn how to jump forward, then learn how to control their joints to both keep balance and jump far. 

Some times training gets trapped to a local minimum that agents standstill only to get the alive reward (1.0 per step). Under this situation, episode length goes up quickly, but reward only limited to the number of alive steps. This situation is shown in the first line of learning plot below.

In a typical training process, agents try different ways of hopping, some move dramatically, some move slow and stable. During their training, the reward will go up (find a way that can move better) then fall back a little (fail to move this way for more steps), then go up again (adjust policy and find another way of moving). If the reward goes horizontally for a while, the entropy of policy will go high to encourage exploration.

<div style="display:flex;">
    <div style="display:flex; margin:auto;">
        <img src=".\readme_data\image\demo_hoppe_2_reward.png" alt="reward curve" width="400" style="padding: 5px;"/>
        <img src=".\readme_data\image\demo_hoppe_2_episode.png" alt="reward curve" width="400" style="padding: 5px;"/>
    </div>
</div>

<div style="display:flex;">
    <div style="display:flex; margin:auto; align:left">
        <img src=".\readme_data\image\demo_hoppe_1_reward.png" alt="reward curve" width="400" style="padding: 5px;"/>
        <img src=".\readme_data\image\demo_hoppe_1_entropy.png" alt="reward curve" width="400" style="padding: 5px;"/>
    </div>
</div>

`[experiment not complete]`

#### Humanoid-v2

One of the most complex environment in MuJoCo environments.

`[missing experiment]`

### Image Based Environment:

#### MiniGrid-Empty-5x5-v0

The simplest environment in MiniGrid series environment, run to test code correctness. The maximum episode length during training is **15 steps**, episode steps reaches its minimum(5 steps) in about100 iterations. The entire training process last for 40 minutes.

Run a experiment with the same hyper parameter of my train: `python ppo.py --param_id=1`

<div style="display:flex;">
    <div style="display:flex; margin:auto;">
        <img src=".\readme_data\image\demo_minigrid-empty5_reward.png" alt="reward curve" width="400" style="padding: 5px;"/>
        <img src=".\readme_data\image\demo_minigrid-empty5_episode.png" alt="reward curve" width="400" style="padding: 5px;"/>
    </div>
</div>

#### MiniGrid-Empty-16x16-v0

Bigger empty environment, also works. The maximum episode length during training is 200 steps, algorithm learns quick in first 10 iterations then generally reaches minimal episode step number of 27 steps.

`[missing training curve]`

#### MiniGrid-SimpleCrossingS9N1-v0

`[missing experiment]`

#### MiniGrid-LavaGapS5-v0

`[missing experiment]`

### Some Rollout Episode Videos:

<div style="display:flex;">
    <div style="display:flex; margin:auto;">
        <img src=".\readme_data\video\demo_invertedpendulum_1.gif" alt="reward curve" width="320" style="padding: 5px;"/>
        <img src=".\readme_data\video\demo_invertedpendulum_2.gif" alt="reward curve" width="320" style="padding: 5px;"/>
    </div>       
</div>

<div style="display:flex;">
    <div style="display:flex; margin:auto;">
       <img src=".\readme_data\video\demo_minigrid-empty5.gif" alt="reward curve" width="320" style="padding: 5px;"/>
        <img src=".\readme_data\video\demo_minigrid-empty16.gif" alt="reward curve" width="320" style="padding: 5px;"/>
    </div>
</div>

<div style="display:flex;">
    <div style="display:flex; margin:auto;">
       <img src=".\readme_data\video\demo_hopper_3.gif" alt="reward curve" width="320" style="padding: 5px;"/>
        <img src=".\readme_data\video\demo_hopper_4.gif" alt="reward curve" width="320" style="padding: 5px;"/>
    </div>
</div>


<div style="display:flex;">
    <div style="display:flex; margin:auto;">
       <img src=".\readme_data\video\demo_hopper_1.gif" alt="reward curve" width="320" style="padding: 5px;"/>
        <img src=".\readme_data\video\demo_hopper_2.gif" alt="reward curve" width="320" style="padding: 5px;"/>
    </div>
</div>

## Develop Experience: Reflections (TL;DR)

During this implementation practice, I spend unbearable long time on debugging and tuning, witch is very embarrassing.  In this part, I will record some of my reflections.

### Be Focus & Devoted

You need to know how each line of your code works, How will they affect other functions. You need to code and debug it from scratch to get familiar with them. So you can locate the error quickly and accurately, then also solve it correctly.  This requires you to be very focused on your code. Make sure everything is working as expected. When you leave your development, you should guarantee there are no uncertain things in your code.

### Always Be Cautious

* Start form a small system, make sure it works then add more features, so you can quickly know what is going wrong
* Start from simplest environments then move to more complex ones, add tricks layer by layer

### Be Aware of Data Formats

Some times you need to transform your intermediate data with different python tools like `torch.Tensor`,` ndarray`, python `list` and so on. The transformation is easy an d safe but be careful of the computation over data after your change its format, different python packages treat data differently during their computation. The issue I spent a lot of time to debug is that after transfer plain python `float` number into `torch.Tensor` with an redundant dimeson, I get wrong dimension data when apply multiplication.

So please be aware of your data format if you are not confident about all your "data flow", check key interface in your code to make sure if data are working correctly if necessary. 

### Think the Correct Way of Debugging

Always remember **the purpose of debugging is to find the logic flaw in your code**. One big "myth" in my head is I always doing the test right after I modify the code, wish to find some evidence showing new code is working. This will trap you in time-wasting runs and make your mind unclear about why you do this, what have you modified, even lost the "big picture." The correct way of doing that is to finish a correct version first, then add some new features or do some reformat, then test if it also works fine as before. This way, you can go back to a safe state at any time you fell its hard to locate the issue. If some issue is related to previous code or need more code reformat, then just think comprehensively, then spend some time to finish it. If you diced to by pass some tricky issues by using tricks, it should be a temporary solution. Not solving these tricky sometime leads to endless debugging. Do keep in mind that only the correct code is your result of work. Do not write many codes that do not work properly.

You should **have a clear development log** from which you can read the progress and track development history of your developing & debugging. So you can also plan accordingly. The mistake I make at this part is although I  have a dev-log, I do not utilize it correctly. I use it as a note but not a tool. I should use it to help to clear my mind and do planning in future development. 

I am not mean to hide my weakness in implementation, but I am sure that I know how to debug before, and I am able to be efficient in many previous developments. However, I feel lost on this implementation project, and I think maybe it is because I am not writing code whose logic is more complicated than a website project for a long time. This awkward situation also reminds me to always refresh my status by getting my hands dirty.

Another thing about debugging or tuning a DL algorithm is, If the algorithm does not always learn fine, then there should be errors in your code or in hyperparameter, you need to continue to debug or tune till it always learns nicely. 

Also, in DL algorithm implementation, if your learning curve is weird in some point of view, it is more likely that something is wrong in your code,  you should think of math or algorithm error after make sure there is no serious bug.

### General Way of Debug

1. Reproduce the error
2. Try to think what cause the error
3. Run related code segment to verify 2.
4. Debug and test, if bug fixed, done; else, go to 1.

You need to follow this roughly in your debugging process. Otherwise, you are very likely to mess up the debugging. 

### Learn More Engineer Knowledge for Coding

It is critical to have solid development skills. I think I need to learn more coding skills and engineering knowledge as software developer engineers do. This will improve my code quality, also allow me to be more confident during development.