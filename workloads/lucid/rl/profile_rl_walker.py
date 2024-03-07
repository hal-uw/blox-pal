import gym
import os
import sys
import time
import torch
import argparse
import torch.backends.cudnn as cudnn

sys.path.append(os.path.abspath(
    os.path.join(
        os.path.join(
            os.path.join(os.path.dirname(__file__), os.path.pardir),
            os.path.pardir),
        os.path.pardir)
))

from stable_baselines3 import PPO, A2C, TD3
from stable_baselines3.common.env_util import make_vec_env


parser = argparse.ArgumentParser(
    description="PyTorch DP Synthetic Benchmark", formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--warmup_time', type=int, default=30, help='Warmup time to run the code')
parser.add_argument('--total_time', type=int, default=100, help='Total time to run the code')

args = parser.parse_args()

# args.total_time = settings.total_time

# warmup_epoch = 200
# benchmark_epoch = 1000


def benchmark_rl2(model_name, batch_size):
    t_start = time.time()

    cudnn.benchmark = True 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Environments & Model
    env = make_vec_env("BipedalWalker-v3", n_envs=1)
    if model_name == 'PPO':
        model = PPO("MlpPolicy", env, verbose=0, batch_size=batch_size, device=device)
    elif model_name == 'TD3':
        model = TD3("MlpPolicy", env, verbose=0, batch_size=batch_size, device=device)
    
    # Warm-up
    # model.learn(total_timesteps=warmup_epoch)
    # t_warmend = time.time()

    # Benchmark
    print(f'==> Training {model_name} model with {batch_size} batchsize..')
    warmup_iter_num = -1
    iter_num = 0
    while True:
        if time.time() - t_start >= args.warmup_time and warmup_iter_num == -1:
            t_warmend = time.time()
            warmup_iter_num = iter_num
        if time.time() - t_start >= args.total_time:
            t_end = time.time()
            t_pass = t_end - t_warmend
            break
        model.learn(total_timesteps=1)
        iter_num += 1
    img_sec = (iter_num - warmup_iter_num) * batch_size / t_pass
    print(f'speed: {img_sec}')


if __name__ == '__main__':
    model_name = 'TD3'
    batch_size = args.batch_size
    benchmark_rl2(model_name, batch_size)
