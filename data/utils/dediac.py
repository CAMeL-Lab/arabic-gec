from camel_tools.utils.dediac import dediac_ar
import re
import argparse


def read_data(path):
    with open(path) as f:
        return [line.strip() for line in f.readlines()]

space_re = re.compile(' +')
def dediac(data, path):
    with open(path, mode='w') as f:
        for line in data:
            line = dediac_ar(line)
            line = space_re.sub(' ', line)
            f.write(line)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help='Input file path')
    parser.add_argument('--output_file', help='Output file path')
    args = parser.parse_args()

    data = read_data(args.input_file)
    dediac(data, args.output_file)

