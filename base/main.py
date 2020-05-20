from train import train
from utils import ParamDict
from xvfbwrapper import Xvfb
from pyvirtualdisplay import Display


def debug_run():
    policy_params = ParamDict(
        hidden_dim=64,  # dimension of the hidden state in actor network
        learning_rate=1e-5,  # learning rate of policy update
        discount=0.99,  # discount factor
        lambd=0.95,
        entropy_coef=0.1,  # hyper-parameter to vary the contribution of entropy loss
        critic_coef=0.5,  # Coefficient of critic loss when weighted against actor loss
        clip_param=0.2,
        envs_num=50,
        horizon=200,
        batch_size=64,  # batch size for policy update
        epochs_num=10,  # number of epochs per policy update
    )
    params = ParamDict(
        policy_params=policy_params,
        iter_num=5e4,  # number of training policy iterations
        plotting_iters=500,  # interval for logging graphs and policy rollouts
        seed=123,
        env_name='InvertedPendulum-v2',
        save_path='./save',
        prefix='dev_InvertedPendulum_1'
    )

    print(">=================================<")
    print("Start Training: alg=PPO, env={}, #update={}".format(params.env_name, params.iter_num))
    print("    ---------------------------------")
    print("    Params: {}".format(params))
    print("    ---------------------------------")
    print("    Policy-Params: {}".format(params.policy_params))
    print(">=================================<")
    # vdisplay = Xvfb(width=1280, height=740)
    display = Display(backend='xvfb')
    display.start()
    train(params)
    # vdisplay.stop()
    display.popen.kill()
    print(">=================================<")
    print("Training Finished!: alg=PPO, env={}, #update={}".format(params.env_name, params.iter_num))
    print(">=================================<")


if __name__ == "__main__":
    debug_run()
