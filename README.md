# Implementation Practice: Proxy Policy Optimization

see this [[README tutorial](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)]

> Project description paragraph

## Getting Started

> Instruction of how to download and test this code base

### Prerequisites

> Requirements before going to package installation

### Installing

> Step-by-step instruction of how to install from scratch, end with a tiny system test

### Running Test

> Running a simple(interesting) test for showing it works

## Code Running Guide

### Command Line Interface

You can reach each training settings and hyperparameters separately using command line arguments in this implementation. Several default argument set also provided in `training_param.csv`, you can select different training setting set by ...

### Loggers

This implementation use `tensorboard` as training monitoring tool. Logged data are saved in `./run/` folder.

You can also check the sampled episode videos in folder `./save/video/ `. The video logging feature is controlled by the command line interface,  default values of controlling arguments are:

> ```python
> --log_video=True
> --plotting_iters=20
> ```

Where `log_video` specifies whether saving videos during training, `plotting_iters` specifies the interval (by iteration number) between each two sampled videos. 

Each video is saved in a folder named by the training prefix and iteration number when sampling happens.

### Resume Model

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

> ~~~python
> python ppo.py --checkpoint_iter=5
> ~~~
>
> ```python
> python ppo.py --use_pretrain=True --pretrain_file=FILE_NAME
> ```

You can see prompts in command line to see the loading and saving process working well.

## Project Structure

