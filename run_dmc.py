import os
import argparse

import torch

from dmc_device import apply_dmc_device_patches, resolve_training_device
from dmc_eval import apply_dmc_eval_patch, clone_agents_for_eval

apply_dmc_device_patches()
apply_dmc_eval_patch()

import rlcard
from rlcard.agents.dmc_agent import DMCTrainer
from rlcard.utils import set_seed, Logger, plot_curve


def train(args):

    args.training_device = resolve_training_device(args.cuda, args.training_device)
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
        f"seed={args.seed}, num_episodes={args.num_episodes}, total_frames={args.total_frames}, "
        f"evaluate_every={args.evaluate_every}"
    )

    with Logger(args.log_dir) as logger:
        trainer._eval_config = {
            "env_name": args.env,
            "seed": args.seed,
            "evaluate_every": args.evaluate_every,
            "frames_per_episode": args.frames_per_episode,
            "num_eval_games": args.num_eval_games,
            "logger": logger,
            "_next_eval_episode": args.evaluate_every,
        }
        logger.log(
            f"Evaluating every {args.evaluate_every} episodes "
            f"({args.num_eval_games} games per evaluation)."
        )

        learner_model = trainer.start()

        if learner_model is not None:
            save_path = os.path.join(args.log_dir, "model.pth")
            torch.save(clone_agents_for_eval(learner_model), save_path)
            print("Model saved in", save_path)

        csv_path, fig_path = logger.csv_path, logger.fig_path

    plot_curve(csv_path, fig_path, "dmc")

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
        help=(
            'Learner device: GPU index for CUDA, "mps" for Apple Silicon, '
            '"cpu" for CPU-only, or "0" to auto-pick MPS/CUDA/CPU'
        ),
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
        help='Evaluate every N episodes during training',
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