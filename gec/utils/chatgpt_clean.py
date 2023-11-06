from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET
import string
import re
import argparse


PNX = string.punctuation + ''.join(list(UNICODE_PUNCT_SYMBOL_CHARSET))

def read_data(path):
    with open(path) as f:
        return [x.strip() for x in f.readlines()]

def write_data(data, path):
    with open(path, mode='w') as f:
        for line in data:
            f.write(line)
            f.write('\n')

def pnx_tokenize(txt):
    txt = re.sub(r'([' + re.escape(PNX) + '])(?!\d)', r' \1 ', txt)
    txt = re.sub(' +', ' ', txt)
    txt = txt.strip()
    return txt

def check_en(data):
    en_seq_lens = [(txt, len(re.findall('[A-Za-z]', txt))) for txt in data]
    max_en_seq = max(en_seq_lens, key=lambda x: x[1])
    print(f'Sequence with longest english characters ({max_en_seq[1]}):')
    print(max_en_seq[0])
    return max_en_seq[1]

def detect_en(txt, max_len=15):
    if len(re.findall(r'[A-Za-z]', txt)) > max_len:
        print('Remove English:\n\n')
        print(txt)
        print('\n\n')

def clean_data(data, max_en_len):
    clean = []
    for txt in data:
        # removing diacs
        txt = dediac_ar(txt)
        # check en
        detect_en(txt, max_len=max_en_len)
        # pnx tokenize and multi-space clean
        txt = pnx_tokenize(txt)

        clean.append(txt)
    return clean


if __name__ == '__main__':
    # qalb14_dev_input = read_data('/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac')
    # qalb15_dev_input = read_data('/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/dev/QALB-2015-L2-Dev.sent.no_ids.dediac')
    # zaebuc_dev_input = read_data('/scratch/ba63/gec/data/gec/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok.dediac')

    # print('QALB-2014')
    # qalb14_max_en_seq = check_en(qalb14_dev_input)
    # print('QALB-2015')
    # qalb15_max_en_seq = check_en(qalb15_dev_input)
    # print('ZAEBUC')
    # zaebuc_max_en_seq = check_en(zaebuc_dev_input)
    # print('\n\n')

    qalb14_test_input = read_data('/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/test/QALB-2014-L1-Test.sent.no_ids.clean.dediac')
    # qalb15_test_l1_input = read_data('/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L1-Test.sent.no_ids.dediac')
    # qalb15_test_l2_input = read_data('/scratch/ba63/gec/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2015/test/QALB-2015-L2-Test.sent.no_ids.dediac')
    # zaebuc_test_input = read_data('/scratch/ba63/gec/data/gec/ZAEBUC-v1.0/data/ar/test/test.sent.raw.pnx.tok.dediac')

    print('QALB-2014')
    qalb14_max_en_seq = check_en(qalb14_test_input)
    # print('QALB-2015 L1')
    # qalb15_max_l1_en_seq = check_en(qalb15_test_l1_input)
    # print('QALB-2015 L2')
    # qalb15_max_l2_en_seq = check_en(qalb15_test_l2_input)
    # print('ZAEBUC')
    # zaebuc_max_en_seq = check_en(zaebuc_test_input)
    # print('\n\n')

    # qalb14_dev_chatgpt = read_data('output/qalb14/chatgpt_output_0_shot_qalb14_dev.txt')
    # qalb15_dev_chatgpt = read_data('output/qalb15/chatgpt_output_0_shot_qalb15_l2_dev.txt')
    # zaebuc_dev_chatgpt = read_data('output/zaebuc/chatgpt_output_0_shot_zaebuc_dev.txt')

    qalb14_test_chatgpt = read_data('output/qalb14/chatgpt_output_3_shot_qalb14_test.txt')
    # qalb15_test_l1_chatgpt = read_data('output/qalb15/chatgpt_output_3_shot_qalb15_l1_test.txt')
    # qalb15_test_l2_chatgpt = read_data('output/qalb15/chatgpt_output_3_shot_qalb15_l2_test.txt')
    # zaebuc_test_chatgpt = read_data('output/zaebuc/chatgpt_output_3_shot_zaebuc_test.txt')



    # print('Cleaning QALB-2014..')
    # qalb14_clean = clean_data(qalb14_dev_chatgpt,
    #                           qalb14_max_en_seq)
    # print('Cleaning QALB-2015..')
    # qalb15_clean = clean_data(qalb15_dev_chatgpt,
    #                         qalb15_max_en_seq)
    # print('Cleaning ZAEBUC..')
    # zaebuc_clean = clean_data(zaebuc_dev_chatgpt,
    #                           zaebuc_max_en_seq)

    # write_data(qalb14_clean, 'clean_output/qalb14/chatgpt_output_0_shot_qalb14_dev.txt')
    # write_data(qalb15_clean, 'clean_output/qalb15/chatgpt_output_0_shot_qalb15_l2_dev.txt')
    # write_data(zaebuc_clean, 'clean_output/zaebuc/chatgpt_output_0_shot_zaebuc_dev.txt')


    # write_data(qalb14_clean, 'clean_output/qalb14/chatgpt_output_3_shot_qalb14_test.txt')

    print('Cleaning QALB-2014..')
    qalb14_clean = clean_data(qalb14_test_chatgpt,
                              qalb14_max_en_seq)
    # print('Cleaning QALB-2015 L1..')
    # qalb15_l1_clean = clean_data(qalb15_test_l1_chatgpt,
    #                           qalb15_max_l1_en_seq)
    # print('Cleaning QALB-2015 L2..')
    # qalb15_l2_clean = clean_data(qalb15_test_l2_chatgpt,
    #                           qalb15_max_l2_en_seq)

    # print('Cleaning ZAEBUC..')
    # zaebuc_clean = clean_data(zaebuc_test_chatgpt,
    #                           zaebuc_max_en_seq)

    write_data(qalb14_clean, 'clean_output/qalb14/chatgpt_output_3_shot_qalb14_test.txt')
    # write_data(qalb15_l1_clean, 'clean_output/qalb15/chatgpt_output_3_shot_qalb15_l1_test.txt')
    # write_data(qalb15_l2_clean, 'clean_output/qalb15/chatgpt_output_3_shot_qalb15_l2_test.txt')
    # write_data(zaebuc_clean, 'clean_output/zaebuc/chatgpt_output_3_shot_zaebuc_test.txt')
