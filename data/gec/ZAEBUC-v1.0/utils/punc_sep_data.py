from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET
import string
import re
import argparse

"""A simple script to dediac and punctuation tokenize
Arabic data"""

parser = argparse.ArgumentParser()
parser.add_argument( "--input_file_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data dir.."
                    )

parser.add_argument("--output_file_dir",
                    default=None,
                    type=str,
                    help="The path of the output file"
                    )

args = parser.parse_args()

puncs = string.punctuation + ''.join(list(UNICODE_PUNCT_SYMBOL_CHARSET))

output_file = open(args.output_file_dir, mode='w', encoding='utf8')

with open(args.input_file_dir, encoding='utf8') as f:
    for line in f.readlines():
        line = line.strip()
        line = re.sub(r'([' + re.escape(puncs) + '])(?!\d)', r' \1 ', line)
        line = re.sub(' +', ' ', line)
        line = line.strip()
        output_file.write(line)
        output_file.write('\n')

output_file.close()
