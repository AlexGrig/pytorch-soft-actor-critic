import argparse
from pathlib import Path
import datetime
import time
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v4",
                    help='Mujoco Gym environment (default: HalfCheetah-v4)')
parser.add_argument('--checkpoint', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

args = parser.parse_args()

def get_state(obs):
    """
    Function which alters the observation received from the environment. Output from this function is written
    to replay_buffer.
    """
    
    #state = np.array(obs)
    #state = state.transpose((2, 0, 1))
    #state = torch.from_numpy(state)
    #return state.unsqueeze(0)
    if isinstance(obs, tuple): # seems to work on reset in gymansium
        tt0 = obs[0]
    else:
        tt0 = obs
    #tt1 = wrap.obs_fit_shape_to_pytorch(tt0, extra_batch_dim=False)
    #return torch.from_numpy(tt1)
    return tt0

def make_env(env_id='HalfCheetah-v4',record_video=False, log_dir = None, seed=12345):
    #env_id = "PongNoFrameskip-v4"
    
    env = gym.make(args.env_name, render_mode='rgb_array')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if record_video:
        env.metadata["render_fps"] = 10
        env.metadata["fps"] = 10
        env = RecordVideo(env, video_folder=log_dir, episode_trigger=lambda episode_no: True, 
                   step_trigger = None, video_length = 0, name_prefix = 'test-video', disable_logger = False)
        # Note! to be able to use video recorder like this I had to modify external code in two places:
        # 1) File: python3.9/site-packages/moviepy/video/VideoClip.py
        #    I commented one decorator of the method `write_videofile` because it could not get default 
        #       and explicitely transferred fps param.:
        #    @requires_duration
        #    #@use_clip_fps_by_default
        #    @convert_masks_to_RGB
        #    def write_videofile(self, filename, fps=None, codec=None, ...
        # 2) File site-packages/gymnasium/wrappers/monitoring/video_recorder.py  
        #    In method `close` I changed the call to `write_videofile` function to be able to explicitely
        #    transfer `fps` parameter.
        #    # AlexGrig ->
        #    # clip.write_videofile(self.path, logger=moviepy_logger
        #    clip.write_videofile(self.path, fps=self.frames_per_sec, logger=moviepy_logger)
        #    # AlexGrig <-
        # Only after this manipulations vedeo recording started to work.     
    print(env_id)
    return env

test_log_dir = Path(args.checkpoint).parent

#test_log_dir = 'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
#                                                  args.policy, "autotune" if args.automatic_entropy_tuning else "")

env = make_env(env_id='HalfCheetah-v4', record_video=True, log_dir=test_log_dir, seed=args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent.load_checkpoint(args.checkpoint, evaluate=True)
writer = SummaryWriter(test_log_dir)

state = get_state(env.reset(seed=args.seed))
episode_reward = 0
episode_length = 0
done = False
episode_start_time = time.time()
while not done:
    action = agent.select_action(state, evaluate=True)

    next_state, reward, terminated, truncated, _ = env.step(action)
    next_state = get_state(next_state)
    done = (terminated or truncated)

    episode_reward += reward
    episode_length +=1
    
    state = next_state

    if done: # episode ends
        env.close()
        print(f'Test episode finished.')
        print(f'Episode reward: {episode_reward}')
        print(f'Episode length: {episode_length}')
        print(f'Episode time: {time.time() - episode_start_time}')
        break
            

