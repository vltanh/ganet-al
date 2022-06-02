import csv
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_list_path')
    parser.add_argument('--used_list_path')
    parser.add_argument('--new_list_path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    old_list = list(csv.reader(open(args.old_list_path), delimiter=' '))

    used_list = list(csv.reader(open(args.used_list_path), delimiter=' '))
    used_list = [x[0] for x in used_list]

    new_list = list(filter(lambda x: x[0] not in used_list, old_list))

    with open(args.new_list_path, 'w') as f:
        csv.writer(f, delimiter=' ').writerows(new_list)
