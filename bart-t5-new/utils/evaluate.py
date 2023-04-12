from m2scorer import m2scorer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--system_output')
parser.add_argument('--m2_file')

args = parser.parse_args()


m2scorer.evaluate(args.system_output, args.m2_file, timeout=30)