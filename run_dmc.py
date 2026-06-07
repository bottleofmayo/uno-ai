import glob
import os
import re
import argparse

import torch

import rlcard
from rlcard.agents.dmc_agent import DMCTrainer
from rlcard.utils import get_device, set_seed, tournament, Logger, plot_curve

def _checkpoint_snapshots(log_dir, num_players):
    snapshots = {}
    for path in glob.glob(os.path.join(log_dir, '*.pth')):
        basename = os.path.basename(path)
        match = re.fullmatch(r'(\d+)_(\d+)\.pth', basename)
        if not match:
            continue
        position = int(match.group(1))
        frames = int(match.group(2))
        snapshots.setdefault(frames, {})[position] = path

    valid_snapshots = {}
    for frames, files in snapshots.items():
        if len(files) == num_players:
            valid_snapshots[frames] = [files[pos] for pos in sorted(files)]
    return dict(sorted(valid_snapshots.items()))


def _load_dmc_agents(paths):
    agents = []
    for path in paths:
        agent = torch.load(path, map_location='cpu')
        if hasattr(agent, 'net'):
            agent.device = 'cpu'
            agent.net.to('cpu')
        agents.append(agent)
    return agents


def train(args):

    device = get_device()
    set_seed(args.seed)

    if args.log_dir:
        args.log_dir = args.log_dir.rstrip('/\\')
        args.savedir = os.path.dirname(args.log_dir) or '.'
        args.xpid = os.path.basename(args.log_dir)
    else:
        args.log_dir = os.path.join(args.savedir, args.xpid)

    if args.total_frames is None:
        args.total_frames = args.num_episodes * args.frames_per_episode

    env = rlcard.make(args.env, config={'seed': args.seed})

    trainer = DMCTrainer(
        env,
        cuda=args.cuda,
        load_model=args.load_model,
        xpid=args.xpid,
        savedir=args.savedir,
        save_interval=args.save_interval,
        num_actor_devices=args.num_actor_devices,
        num_actors=args.num_actors,
        training_device=args.training_device,
        total_frames=args.total_frames,
        exp_epsilon=args.exp_epsilon,
        batch_size=args.batch_size,
        unroll_length=args.unroll_length,
        num_buffers=args.num_buffers,
        num_threads=args.num_threads,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        momentum=args.momentum,
        epsilon=args.epsilon,
    )

    print(
        f"Training DMC: env={args.env}, xpid={args.xpid}, savedir={args.savedir}, "
        f"seed={args.seed}, num_episodes={args.num_episodes}, total_frames={args.total_frames}"
    )
    trainer.start()

    with Logger(args.log_dir) as logger:
        logger.log(
            f"Evaluating DMC snapshots in {args.log_dir} for {args.num_eval_games} games."
        )

        snapshots = _checkpoint_snapshots(args.log_dir, env.num_players)
        if not snapshots:
            logger.log('No DMC snapshots found for evaluation.')
        else:
            for frames, paths in snapshots.items():
                eval_env = rlcard.make(args.env, config={'seed': args.seed})
                agents = _load_dmc_agents(paths)
                eval_env.set_agents(agents)
                reward = tournament(eval_env, args.num_eval_games)[0]
                episode = int(frames / args.frames_per_episode) if args.frames_per_episode else frames
                logger.log_performance(episode, reward)

            final_paths = list(snapshots.values())[-1]
            final_agents = _load_dmc_agents(final_paths)
            save_path = os.path.join(args.log_dir, 'model.pth')
            torch.save(final_agents, save_path)
            print('Model saved in', save_path)

    csv_path, fig_path = logger.csv_path, logger.fig_path
    plot_curve(csv_path, fig_path, 'dmc')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DMC example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='uno',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy'
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--load_model',
        action='store_true',
        help='Load an existing model',
    )
    parser.add_argument(
        '--xpid',
        default='uno',
        help='Experiment id (default: uno)',
    )
    parser.add_argument(
        '--savedir',
        default='experiments/uno_dmc_result',
        help='Root dir where experiment data will be saved'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='',
        help='Alias for savedir/xpid, if provided the experiment directory is created directly',
    )
    parser.add_argument(
        '--save_interval',
        default=30,
        type=int,
        help='Time interval (in minutes) at which to save the model',
    )
    parser.add_argument(
        '--num_actor_devices',
        default=1,
        type=int,
        help='The number of devices used for simulation',
    )
    parser.add_argument(
        '--num_actors',
        default=2,
        type=int,
        help='The number of actors for each simulation device',
    )
    parser.add_argument(
        '--training_device',
        default="0",
        type=str,
        help='The index of the GPU used for training models',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for environment and training',
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=10000,
        help='Approximate number of episodes to train for',
    )
    parser.add_argument(
        '--frames_per_episode',
        type=int,
        default=100,
        help='Estimated number of environment frames per episode',
    )
    parser.add_argument(
        '--total_frames',
        type=int,
        default=None,
        help='Total environment frames to train for; overrides num_episodes * frames_per_episode',
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
        help='Evaluation interval in episodes (for compatibility)',
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
        help='Number of evaluation games to run (for compatibility)',
    )
    parser.add_argument(
        '--exp_epsilon',
        type=float,
        default=0.01,
        help='Exploration epsilon for DMC actor models',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Learner batch size',
    )
    parser.add_argument(
        '--unroll_length',
        type=int,
        default=100,
        help='Unroll length for children actors',
    )
    parser.add_argument(
        '--num_buffers',
        type=int,
        default=50,
        help='Number of shared-memory buffers',
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=4,
        help='Number of learner threads',
    )
    parser.add_argument(
        '--max_grad_norm',
        type=int,
        default=40,
        help='Gradient clipping max norm',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='Learning rate for the optimizer',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSProp smoothing constant',
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0,
        help='RMSProp momentum',
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.00001,
        help='RMSProp epsilon',
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)
