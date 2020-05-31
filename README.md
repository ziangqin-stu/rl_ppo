# Implementation Practice: Proxy Policy Optimization

see this [[README writing tutorial](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)]

> Project description paragraph

## Getting Started

> Instruction of how to download and test this code base

Make sure your machine is using cuda, then clone the code base to local:

`git clone https://github.com/ziangqin-stu/impl_ppo.git `

Install necessary python packages:

` pip install -r requirements.txt`

Run a simple experiment (cart pole) to check every thing going well:

`python ppo.py --param_id=0`

## Code Running Guide

### Command Line Interface

You can reach each training settings and hyperparameters separately using command line arguments in this implementation. Several default argument set also provided in `training_param.csv`, you can select different training setting set by specify the parameter set id in `training_param.csv`, using command line argument `--param_id`.

Like other implementation, you can use the command line interface to control the program. You can select different default argument set to reach a baseline experiment setting, i.e. , select different environments and apply baseline corresponding hyperparameter to it. Then you can customize the parameters using other command line arguments to run a slightly different experiment. You can also add your baseline parameters too.

In `training_param.csv`, each training setting is a line, you can also add your training settings in that file and do the rest tests. The purpose of designing this feature is to simplify the tuning process when manually tuning is needed. 

Run an experiment using small number of rollout worker:

`python ppo.py --param_id=1 --envs_num=15`

### Training Loggers

This implementation use `tensorboard` as training monitoring tool. Logged data are saved in `./run/` folder.

You can also check the sampled episode videos in folder `./save/video/ `. The video logging feature is controlled by the command line interface,  default values of controlling arguments are:

> ```python
> --log_video=True
> --plotting_iters=20
> ```

Where `log_video` specifies whether saving videos during training, `plotting_iters` specifies the interval (by iteration number) between each two sampled videos. 

Each video is saved in a folder named by the training prefix and iteration number when sampling happens.

### Resume Model Feature

This implementation also offers a feature of save and load training check points. 

There are basically two way of using this feature:

1. save checkpoints as a pre-trained model to local, continue training process by loading checkpoint from local, nonmatter your previous training process stopped intentionally or unintentionally
2. debug when an issue only occurs after some training iterations: load the checkpoint saved just before the issue occurs and debug, so you can save time of waiting that bug occurs again.

Saving & loading are also controlled by the command line interface, default values of controlling arguments are:

> ```python
> --save_checkpoint=True
> --checkpoint_iter=100
> --use_pretrain=False
> --pretrain_file=""
> ```

Where `save_checkpoint` specifies whether saving checkpoints automatically during training, `checkpoint_iter` specifies the interval (by iteration number) between each two checkpoints. 

`use_pretrain` specifies whether algorithm starts with loading a checkpoint from local file, `pretrain_file` is the file name (not include the file path) to load.

Checkpoints features are built with `torch.save()` and `torch.load()`, local files are formatted as `.tar` file.

Try:

`python ppo.py --checkpoint_iter=5`

`python ppo.py --use_pretrain=True --pretrain_file=FILE_NAME`

You can see prompts in command line showing the loading and saving process works well.

## Project Structure

### PPO Code Structure

### Applied Tricks

## Training Result

### Discrete Environment:

#### CartPole-v1

MountainCar-v0

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

A more complex environment than InvertedPendulum-v2. Humanoid-v2

#### Humanoid-v2

One of the most complex environment in MuJoCo environments.

### Image Observation Environment:

#### MiniGrid-Empty-5x5-v0

The simplest environment in MiniGrid series environment, run to test code correctness. The maximum episode length during training is 15 steps, episode steps reaches its minimum(5 steps) in about100 iterations. The entire training process last for 40 minutes.

Run a experiment with the same hyper parameter of my train: `python ppo.py --param_id=1`

<div style="display:flex;">
    <div style="display:flex; margin:auto;">
        <img src=".\readme_data\image\demo_minigrid-empty5_reward.png" alt="reward curve" width="400" style="padding: 5px;"/>
        <img src=".\readme_data\image\demo_minigrid-empty5_episode.png" alt="reward curve" width="400" style="padding: 5px;"/>
    </div>
</div>

#### MiniGrid-Empty-16x16-v0

Bigger empty environment, also works. The maximum episode length during training is 200 steps, algorithm learns quick in first 10 iterations then generally reaches minimal episode step number of 27 steps.

#### MiniGrid-SimpleCrossingS9N1-v0

#### MiniGrid-LavaGapS5-v0

### Rollout Episode Videos:

<div style="display:flex;">
    <div style="display:flex; margin:auto;">
        <video src=".\readme_data\video\demo_invertedpendulum.mp4" controls width="300" autoplay=FALSE type="video/webm" style="padding: 10px;"></video>
        <video source src=".\readme_data\video\demo_hopper.mp4" controls width="300" autoplay=FALSE type="video/webm" style="padding: 10px;"></video>
        <video source src=".\readme_data\video\demo_minigrid-empty5.mp4" controls width="300" autoplay=FALSE type="video/webm" style="padding: 10px;"></video>
    </div>
</div>



## Develop Experience: Reflections

During this implementation practice, I spend unbearable long time on debugging and tuning, witch makes me feel very embarrassing.  In this part, I will record some of my reflections.

### Be Focus & Devoted

You need to know how each line of your code works, How will they affect other functions. You need to code and debug it from scratch to get familiar with them. So you can locate the error quickly and accurately, then also solve it correctly.  This requires you to be very focused on your code. Make sure everything is working as expected. When you leave your development, you should guarantee there are no uncertain things in your code.

### Always Be Cautious

* Start form a small system, make sure it works then add more features, so you can quickly know what is going wrong
* Start from simplest environments then move to more complex ones, add tricks layer by layer

### Be Aware of Data Formats

Some times you need to transform your intermediate data with different python tools like `torch.Tensor`,` ndarray`, python `list` and so on. The transformation is easy an d safe but be careful of the computation over data after your change its format, different python packages treat data differently during their computation. The issue I spent a lot of time to debug is that after transfer plain python `float` number into `torch.Tensor` with an redundant dimeson, I get wrong dimension data when apply multiplication.

So please be aware of your data format if you are not confident about all your "data flow", check key interface in your code to make sure if data are working correctly if necessary. 

### Think the Correct Way of Debugging

I wasted a lot of time on debugging the logic of my code. But it ...

Also, remember **the purpose of debugging is to find the logic flaw in your code**. One big "myth" in my head is I always doing the test right after I modify the code, wish to find some evidence of new code is working. This will trap you in time-wasting runs and make your mind unclear about why you do this, even lost the "big picture." The correct way of doing that is to finish a correct version first, then add some new features or do some reformat, then test if it also works fine as before. If some issue is related to previous code or need more code reformat, then just think comprehensively then spend some time to finish it. If you diced to by pass that issue by using some tricks, it should be a temporary solution. Not solving this will cause endless debug. Do keep in mind that only the correct code is your result of work. Do not write many codes that do not work properly.

You should **have a clear development log** from which you can read the progress and track development history of your developing & debugging. So you can also plan accordingly. The mistake I make at this part is although I  have a reasonable dev-log, I do not utilize it seriously. I use it as a note but not a tool. I should use it to help my think in future development. 

I am not mean to hide my incompetence on implementation, but I am sure that I know how to do it before, and I was efficient on many previous tasks. However, I feel lost on this implementation project, and I think maybe it is because I am not writing code whose logic is slightly more complicated than a website project for a long time. This awkward situation also reminds me to **always refresh my status by getting my hands dirty**.

Another thing about debugging or tuning a DL algorithm is, If the learning sign is not strong or not always shows, then that is not a good code or hyper parameter, you need to continue debug or tune till it always learns nicely. 

Also, in DL algorithm implementation, if your learning curve is weird in some point of view, it is more likely that something is wrong in your code, instead of the math or algorithm is wrong.

### General Way of Debug



### Learn More Engineer Knowledge for Coding

It is critical to have solid development skills! I think I need to learn more coding skills and engineering knowledge as software developer engineers do. This will improve your code quality, also allows you to be more confident during your development. Also, pay some attention to books that teach you how to think when developing. Implementation skills or even developing skills are critical to everyone who are willing to do some AI.