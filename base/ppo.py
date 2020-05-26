import copy
import csv

from train import train
from utils import ParamDict
from pyvirtualdisplay import Display
import argparse

# ================
# Argument Binding
# ================
parser = argparse.ArgumentParser(description="Specific Hyper-Parameters for PPO training. ")
# >> select fundamental parameters from local file
parser.add_argument('--param_id', default=0, type=int, help='index of parameter load from local csv file')
# >> training base settings
parser.add_argument('--env_name', help='training gym environment name, check data.py for supporting environments.')
parser.add_argument('--prefix', help='experiment prefix for savings.')
parser.add_argument('--save_path', help='folder name of results saving.')
parser.add_argument('--use_pretrain', '-p', type=bool, help='specify if using pretrained model.')
parser.add_argument('--pretrain_file', type=str, help='folder name of pretrained model.')
parser.add_argument('--parallel', type=bool, help='specify if use ray parallelization')
# >> loggers
parser.add_argument('--log_video', type=bool, help='specify if save episode video during training')
parser.add_argument('--plotting_iters', type=int, help='video saving interval')
# >> algorithm training settings
parser.add_argument('--iter_num', type=int, help='training iter length')
parser.add_argument('--seed', type=int, help='training seed (experimental)')
parser.add_argument('--reducing_entro_loss', type=bool,
                    help='specify if apply entropy coefficient discount during training')
# >> algorithm detailed settings
parser.add_argument('--learning_rate', type=float, help='optimizer learning rate')
parser.add_argument('--hidden_dim', type=int, help='fully connected network hidden dimension')
parser.add_argument('--envs_num', type=int, help='(parallel) interacting environment number')
parser.add_argument('--horizon', type=int, help='max episode steps during training')
parser.add_argument('--batch_size', type=int, help='sampling size during policy update')
parser.add_argument('--epochs_num', type=int, help='policy update number in each iteration')
parser.add_argument('--critic_coef', type=float, help='critic loss coefficient')
parser.add_argument('--entropy_coef', type=float, help='entropy loss coefficient')
# >> stable parameters
parser.add_argument('--clip_param', help='PPO clip parameter')
parser.add_argument('--discount', help='episode reward discount')
parser.add_argument('--lambd', help='GAE lambda')
# >> parse arguments
args = parser.parse_args()


# =======================
# Read Training Parameters from File / Update by Inputs
# =======================
def load_params(index):
    f = open('./train_param.csv', 'r')
    with f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
        file_param = rows[index]
        policy_params = ParamDict(
            hidden_dim=args.hidden_dim if args.hidden_dim is not None else int(file_param['hidden_dim']),
            learning_rate=args.learning_rate if args.learning_rate is not None else float(file_param['learning_rate']),
            envs_num=args.envs_num if args.envs_num is not None else int(file_param['envs_num']),
            horizon=args.horizon if args.horizon is not None else int(file_param['horizon']),
            batch_size=args.batch_size if args.batch_size is not None else int(file_param['batch_size']),
            epochs_num=args.epochs_num if args.epochs_num is not None else int(file_param['epochs_num']),
            critic_coef=args.critic_coef if args.critic_coef is not None else float(file_param['critic_coef']),
            entropy_coef=args.entropy_coef if args.entropy_coef is not None else float(file_param['entropy_coef']),
            discount=args.discount if args.discount is not None else float(file_param['discount']),
            lambd=args.lambd if args.lambd is not None else float(file_param['lambd']),
            clip_param=args.clip_param if args.clip_param is not None else float(file_param['clip_param'])
        )
        params = ParamDict(
            policy_params=policy_params,
            env_name=args.env_name if args.env_name is not None else file_param['env_name'],
            prefix=args.prefix if args.prefix is not None else file_param['prefix'],
            save_path=args.save_path if args.save_path is not None else file_param['save_path'],
            use_pretrain=args.use_pretrain if args.use_pretrain is not None else bool(file_param['use_pretrain']),
            pretrain_file=args.pretrain_file if args.pretrain_file is not None else file_param['pretrain_file'],
            parallel=args.parallel if args.parallel is not None else bool(file_param['parallel']),
            log_video=args.log_video if args.log_video is not None else bool(file_param['log_video']),
            plotting_iters=args.plotting_iters if args.plotting_iters is not None else int(
                file_param['plotting_iters']),
            iter_num=args.iter_num if args.iter_num is not None else int(file_param['iter_num']),
            seed=args.seed if args.seed is not None else int(file_param['seed']),
            reducing_entro_loss=args.reducing_entro_loss if args.reducing_entro_loss is not None else bool(
                file_param['reducing_entro_loss'])
        )
    return params, policy_params


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
    params, policy_params = load_params(args.param_id)
    cmd_run(params)
