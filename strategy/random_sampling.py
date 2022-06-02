import argparse
import csv
import random
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--inp_path',
        required=True,
    )
    parser.add_argument(
        '--out_path',
        required=True,
    )
    parser.add_argument(
        '--n_sample',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
    )
    return parser.parse_args()


def random_sampling(items, n_sample, seed=None):
    random.seed(seed)
    random.shuffle(items)
    return items[:n_sample]


if __name__ == '__main__':
    args = parse_args()

    # Read from input file
    with open(args.inp_path) as f:
        items = sum(csv.reader(f), [])

    # Sample using strategy
    picked_lines = random_sampling(items, args.n_sample, args.seed)

    # Write to output file
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, 'w') as f:
        csv.writer(f).writerows([[x] for x in picked_lines])
