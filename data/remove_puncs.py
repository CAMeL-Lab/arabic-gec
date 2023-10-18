from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET
import string
import re
import argparse

"""A simple script to remove puncs"""

parser = argparse.ArgumentParser()
parser.add_argument( "--input_file",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data dir.."
                    )

parser.add_argument("--output_file",
                    default=None,
                    type=str,
                    help="The path of the output file"
                    )

args = parser.parse_args()

puncs = string.punctuation + ''.join(list(UNICODE_PUNCT_SYMBOL_CHARSET)) + '&amp;'

output_file = open(args.output_file, mode='w', encoding='utf8')

pnx_re = re.compile(r'([' + re.escape(puncs) + '])')
space_re = re.compile(' +')

with open(args.input_file, encoding='utf8') as f:
    for line in f.readlines():
        line = line.strip()
        line = pnx_re.sub(r'', line)
        line = space_re.sub(' ', line)
        line = line.strip()
        output_file.write(line)
        output_file.write('\n')

output_file.close()
