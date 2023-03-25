from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET
import string
import re
import argparse

"""A simple script to punctuation tokenize Arabic text"""

parser = argparse.ArgumentParser()
parser.add_argument( "--input",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data file"
                    )

parser.add_argument("--output",
                    default=None,
                    type=str,
                    help="The output data file"
                    )

args = parser.parse_args()

puncs = string.punctuation + ''.join(list(UNICODE_PUNCT_SYMBOL_CHARSET))

output_file = open(args.output, mode='w', encoding='utf8')

with open(args.input, encoding='utf8') as f:
    for line in f.readlines():
        line = line.strip()
        line = re.sub(r'([' + re.escape(puncs) + '])(?!\d)', r' \1 ', line)
        line = re.sub(' +', ' ', line)
        line = line.strip()
        output_file.write(line)
        output_file.write('\n')

output_file.close()
