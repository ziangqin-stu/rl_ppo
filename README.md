# Implementation Practice: Proxy Policy Optimization

see this [[README writing tutorial](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)]

> Project description paragraph

## Getting Started

> Instruction of how to download and test this code base

### Prerequisites

> Requirements before going to package installation

### Installing

> Step-by-step instruction of how to install from scratch, end with a tiny system test

### Running Test

> Running a simple(interesting) test for showing it works

```python
python ppo.py --param_id=0
```



## Code Running Guide

### Command Line Interface

You can reach each training settings and hyperparameters separately using command line arguments in this implementation. Several default argument set also provided in `training_param.csv`, you can select different training setting set by specify the parameter set id in `training_param.csv`, using command line argument `--param_id`.

Like other python code, you can use the command line interface to control the program. The difference in this implementation is you can select different default argument set to reach a baseline experiment setting, i.e. , select different environments and apply baseline corresponding hyperparameter to it. Then you can customize the parameters using other command line arguments to run a slightly different experiment.

The "baseline argument sets" are stored in `training_param.csv` as mentioned above, each training setting is a line, you can also add your training settings in that file and do the rest tests. The purpose of designing this feature is to accelerate the tuning process and help to keep a clear mind by saving time spent on typing in parameters.

Try:

```python
python ppo.py --param_id=1 --envs_num=5
```

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

~~~python
python ppo.py --checkpoint_iter=5
~~~

```python
python ppo.py --use_pretrain=True --pretrain_file=FILE_NAME
```

You can see prompts in command line to see the loading and saving process working well.

## Project Structure

### PPO Code Structure

### Applied Tricks

## Training Result

### Discrete Environment:

#### CartPole-v1

MountainCar-v0

### Continuous Environment:

InvertedPendulum-v2

Hopper-v2

Humanoid-v2

### Image Observation Environment:

#### MiniGrid-Empty-5x5-v0

The simplest environment in MiniGrid series environment, run to test code correctness. The maximum episode length during training is 15steps, training converges within 100 iterations.

#### MiniGrid-SimpleCrossingS9N1-v0

#### MiniGrid-LavaGapS5-v0

### Some Episode Videos:

## Develop Experience: Take Homes

### Always Be Cautious

* Start form a small system, make sure it works then add more features, so you can quickly know what is going wrong
* Start from simplest environments then move to more complex ones, add tricks layer by layer

### Be Aware of Data Formats

Some times you need to transform your intermediate data with different python tools like `torch.Tensor`,` ndarray`, python `list` and so on. The transformation is easy an d safe but be careful of the computation over data after your change its format, different python packages treat data differently during their computation. The issue I spent a lot of time to debug is that after transfer plain python `float` number into `torch.Tensor` with an redundant dimeson, I get wrong dimension data when apply multiplication.

So please be aware of your data format if you are not confident about all your "data flow", check key interface in your code to make sure if data are working correctly if necessary.

### Think the Correct Way of Debugging

I wasted a lot of time on debugging the logic of my code. But it ...

record your debugging route, avoid lost in long logic chains ...

### Learn More Engineer Knowledge for Coding

It is critical for an AI developer to have solid develop skills! I think I need to learn more coding skills and engineering knowledge as software developer engineers do. This will improve your code quality, also allows you to be more confident during your development. 