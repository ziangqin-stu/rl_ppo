import copy
import csv

from train import train
from utils import ParamDict
from pyvirtualdisplay import Display
import argparse

# ================
# Argument Binding
# ================
# >> training base settings
parser = argparse.ArgumentParser(description="Specific Hyper-Parameters for PPO training. ")
parser.add_argument('--env_name', default='Hopper-v2',
                    help='training gym environment name, check data.py for supporting environments.')
parser.add_argument('--prefix', default='test_Hopper_demo', help='experiment prefix for savings.')
parser.add_argument('--save_path', default='./save', help='folder name of results saving.')
parser.add_argument('--use_pretrain', '-p', type=bool, default=False, help='specify if using pretrained model.')
parser.add_argument('--pretrain_file', default=None, type=str, help='folder name of pretrained model.')
parser.add_argument('--parallel', default=True, type=bool, help='specify if use ray parallelization')
# >> loggers
parser.add_argument('--log_video', default=True, type=bool, help='specify if save episode video during training')
parser.add_argument('--plotting_iters', default=20, type=int, help='video saving interval')
# >> algorithm training settings
parser.add_argument('--iter_num', default=1000, type=int, help='training iter length')
parser.add_argument('--seed', default=123, help='training seed (experimental)')
parser.add_argument('--reducing_entro_loss', default=True, type=bool,
                    help='specify if apply entropy coefficient discount during training')
# >> algorithm detailed settings
parser.add_argument('--learning_rate', default=1e-5, help='optimizer learning rate')
parser.add_argument('--hidden_dim', default=512, help='fully connected network hidden dimension')
parser.add_argument('--envs_num', default=50, type=int, help='(parallel) interacting environment number')
parser.add_argument('--horizon', default=400, type=int, help='max episode steps during training')
parser.add_argument('--batch_size', default=64, help='sampling size during policy update')
parser.add_argument('--epochs_num', default=50, help='policy update number in each iteration')
parser.add_argument('--critic_coef', default=0.5, help='critic loss coefficient')
parser.add_argument('--entropy_coef', default=1e-2, help='entropy loss coefficient')
# >> stable parameters
parser.add_argument('--clip_param', default=0.2, help='PPO clip parameter')
parser.add_argument('--discount', default=0.99, help='episode reward discount')
parser.add_argument('--lambd', default=0.99, help='GAE lambda')
# >> parse arguments
args = parser.parse_args()


# =======================
# Run Training Experiment
# =======================
def cmd_run(params):
    training_param = copy.deepcopy(params)
    del training_param['policy_params']
    print("=========================================================")
    print("Start PPO Training: env={}, #update={}".format(params.env_name, params.iter_num))
    print("    -------------------------------------------------")
    print("    Training-Params: {}".format(training_param))
    print("    -------------------------------------------------")
    print("    Policy-Params: {}".format(params.policy_params))
    print("=========================================================")
    display = Display(backend='xvfb')
    display.start()
    train(params)
    display.popen.kill()
    print(">=================================<")
    print("Training Finished!: alg=PPO, env={}, #update={}".format(params.env_name, params.iter_num))
    print(">=================================<")


if __name__ == "__main__":
    policy_params = ParamDict(
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        envs_num=args.envs_num,
        horizon=args.horizon,
        batch_size=args.batch_size,
        epochs_num=args.epochs_num,
        critic_coef=args.critic_coef,
        entropy_coef=args.entropy_coef,
        discount=args.discount,
        lambd=args.lambd,
        clip_param=args.clip_param
    )
    params = ParamDict(
        policy_params=policy_params,
        env_name=args.env_name,
        prefix=args.prefix,
        save_path=args.save_path,
        use_pretrain=args.use_pretrain,
        pretrain_file=args.pretrain_file,
        parallel=args.parallel,
        log_video=args.log_video,
        plotting_iters=args.plotting_iters,
        iter_num=args.iter_num,
        seed=args.seed,
        reducing_entro_loss=args.reducing_entro_loss
    )
    cmd_run(params)
