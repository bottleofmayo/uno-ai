import os
import argparse

import rlcard
from rlcard.agents import (
    DQNAgent,
    RandomAgent,
)
from rlcard.utils import (
    set_seed,
    tournament,
)
from dmc_device import resolve_torch_device

def move_agent_networks_to_device(agent, device):
    if hasattr(agent, "net"):
        agent.net.to(device)
    if hasattr(agent, "policy_network"):
        agent.policy_network.to(device)
    if hasattr(agent, "q_estimator") and hasattr(agent.q_estimator, "qnet"):
        agent.q_estimator.qnet.to(device)
        agent.q_estimator.device = device
    if hasattr(agent, "target_estimator") and hasattr(agent.target_estimator, "qnet"):
        agent.target_estimator.qnet.to(device)
        agent.target_estimator.device = device
    if hasattr(agent, "_rl_agent"):
        move_agent_networks_to_device(agent._rl_agent, device)


def load_model(model_path, env=None, position=None, device="cpu"):
    if os.path.isfile(model_path):  # Torch model
        import torch
        loaded = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(loaded, list):
            if position is None:
                raise ValueError(f"Model at {model_path} contains multiple agents; position is required")
            agent = loaded[position]
        else:
            agent = loaded
        if hasattr(agent, "net"):
            agent.device = device
            agent.net.to(device)
        elif hasattr(agent, "set_device"):
            agent.set_device(device)
            move_agent_networks_to_device(agent, device)
    elif os.path.isdir(model_path):  # CFR model
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, model_path)
        agent.load()
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]

    return agent

def evaluate(args):

    device = resolve_torch_device(args.cuda)

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Load models
    agents = []
    for position, model_path in enumerate(args.models):
        agents.append(load_model(model_path, env, position, device))
    env.set_agents(agents)

    # Evaluate
    rewards = tournament(env, args.num_games)
    for position, reward in enumerate(rewards):
        print(position, args.models[position], reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
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
            'gin-rummy',
        ],
    )
    parser.add_argument(
        '--models',
        nargs='*',
        default=[
            #'experiments/uno_dqn_result/model.pth',
            #'experiments/uno_nfsp_result/model.pth',
            #'uno-rule-v1',
            #'random',
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_games',
        type=int,
        default=10000,
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    evaluate(args)