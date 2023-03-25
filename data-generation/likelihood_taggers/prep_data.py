import argparse


def read_data(path):
    with open(path) as f:
        return [x.strip() for x in f.readlines()]


def read_alignment(path):
    src_tokens = []
    tgt_tokens = []
    tags = []
    delete_examples = []
    insert_examples = []
    replace_examples = []

    with open(path) as f:
        for line in f.readlines():
            line = line.split('\t')
            if len(line) > 1:
                src = line[0]
                tgt = line[1]
                tag = line[2]

                tgt_tokens.append(tgt.strip())
                tags.append(tag.strip())

            else:
                replace_tokens, replace_tags = replace_data(tgt_tokens, tags)
                insert_tokens, insert_tags = insert_data(tgt_tokens, tags)
                delete_tokens, delete_tags = delete_data(tgt_tokens, tags)

                if len(delete_tokens) != 0:
                    delete_examples.append({'tokens': delete_tokens,
                                            'tags': delete_tags})

                if len(insert_tokens) != 0:
                    insert_examples.append({'tokens': insert_tokens,
                                            'tags': insert_tags})

                if len(replace_tokens) != 0:
                    replace_examples.append({'tokens': replace_tokens,
                                            'tags': replace_tags})

                if len(insert_tokens) != 0 and len(delete_tokens) != 0:
                    assert len(replace_tokens) == len(insert_tokens) == len(delete_tokens)
                    assert len(replace_tags) == len(insert_tags) == len(delete_tags)
                    assert replace_tokens == insert_tokens == delete_tokens


                tgt_tokens, tags = [], []

    return replace_examples, delete_examples, insert_examples


def replace_data(tgt_tokens, tags):
    if len(set(tags)) == 1:
        assert list(set(tags))[0] == 'UC'
        return [], []

    replace_tokens, replace_tags = [], []

    for token, tag in zip(tgt_tokens, tags):
        if token != '':
            if 'REPLACE' in tag:
                replace_tags.append('I')
            else:
                replace_tags.append('C')

            replace_tokens.append(token)
    
    assert len(replace_tokens) == len(replace_tags)

    return replace_tokens, replace_tags


def delete_data(tgt_tokens, tags):
    if 'DELETE' not in tags:
        return [], []

    delete_tokens, delete_tags = [], []
    for i, (token, tag) in enumerate(zip(tgt_tokens, tags)):
        if token == '':
            assert tag == 'DELETE'
            if len(delete_tags) > 0:
                delete_tags[-1] = 'I'
            else:
                delete_tags.append('I')
        else:
            if len(delete_tokens) < len(delete_tags):
                delete_tokens.append(token)
            else:
                delete_tags.append('C')
                delete_tokens.append(token)

    assert len(delete_tokens) == len(delete_tags)

    return delete_tokens, delete_tags

def insert_data(tgt_tokens, tags):
    if 'INSERT_PM' not in tags and 'INSERT_XM' not in tags:
        return [], []

    insert_tokens, insert_tags = [], []

    for token, tag in zip(tgt_tokens, tags):
        if token != '':
            if 'INSERT' in tag:
                insert_tags.append('I')
            else:
                insert_tags.append('C')

            insert_tokens.append(token)
    
    assert len(insert_tokens) == len(insert_tags)

    return insert_tokens, insert_tags

def write_alignment(alignment, path):
    with open(path, mode='w') as f:
        for example in alignment:
            assert len(example['tokens']) == len(example['tags'])
            for token, tag in zip(example['tokens'], example['tags']):
                f.write(f'{token}\t{tag}')
                f.write('\n')
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alignment_path')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    replace_examples, delete_examples, insert_examples = read_alignment(args.alignment_path)

    write_alignment(replace_examples, args.output_path+'.replace.txt')
    write_alignment(delete_examples, args.output_path+'.delete.txt')
    write_alignment(insert_examples, args.output_path+'.insert.txt')

