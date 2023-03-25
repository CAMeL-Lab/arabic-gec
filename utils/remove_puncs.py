from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET
import string
import re
import argparse

"""A simple script to remove puncs"""

parser = argparse.ArgumentParser()
parser.add_argument( "--input",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data dir.."
                    )

parser.add_argument("--output",
                    default=None,
                    type=str,
                    help="The path of the output file"
                    )

args = parser.parse_args()

puncs = string.punctuation + ''.join(list(UNICODE_PUNCT_SYMBOL_CHARSET)) + '&amp;'

output_file = open(args.output, mode='w', encoding='utf8')

with open(args.input, encoding='utf8') as f:
    for line in f.readlines():
        line = line.strip()
        line = re.sub(r'([' + re.escape(puncs) + '])', r'', line)
        line = re.sub(' +', ' ', line)
        line = line.strip()
        output_file.write(line)
        output_file.write('\n')

output_file.close()
