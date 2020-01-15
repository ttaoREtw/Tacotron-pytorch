import yaml
import torch
import random
import argparse
import numpy as np


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='Tacotron')
    parser.add_argument('--config', type=str, help='Path to experiment config file')
    parser.add_argument('--log-dir', default='log/', type=str, help='Logging path', required=False)
    parser.add_argument('--checkpoint-dir', type=str, help='Checkpoint/Result path', required=True)
    parser.add_argument('--checkpoint-path', type=str, help='Restore model from checkpoint path if given', required=False)
    parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducible results.', required=False)
    parser.add_argument('--cpu', action='store_true', help='Disable GPU training')
    parser.add_argument('--no-msg', action='store_true', help='Hide all messages')
    args = parser.parse_args()

    args.gpu = not args.cpu
    args.verbose = not args.no_msg

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Train
    from src.solver import Trainer as Solver
    solver = Solver(config, args)
    solver.load_data()
    solver.build_model()
    solver.exec()

