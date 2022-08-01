import argparse
import csv
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
    return parser.parse_args()


def time_sampling(inp_path, n_sample):
    # Read from input file
    with open(inp_path) as f:
        items = sum(csv.reader(f), [])

    N = len(items)
    return [items[i * (N - 1) // (n_sample - 1)] for i in range(n_sample)]


if __name__ == '__main__':
    args = parse_args()

    # Sample using strategy
    picked_lines = time_sampling(args.inp_path, args.n_sample, args.seed)

    # Write to output file
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, 'w') as f:
        csv.writer(f).writerows([[x] for x in picked_lines])
