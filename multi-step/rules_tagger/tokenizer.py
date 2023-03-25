from transformers import AutoTokenizer

class GECTokenizer:
    def __init__(self, model_path):
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize(self, text):
        if text.isdigit():
            return [text]

        return self._tokenizer.tokenize(text)

    def get_word_ids(self, text):
        return self._tokenizer(text).word_ids(0)[1:-1]

    def _postprocess_bert_like(self, tokenized:list):
        spaces_correct = []

        for token_ind, token in enumerate(tokenized):
            if token_ind == 0:
                spaces_correct.append(token)
                continue

            if token.startswith('##'):
                spaces_correct.append(token[2:])
            else:
                # adding a space helps when tokenizing 
                # many tokens on the source
                spaces_correct.append(' ' + token)
                # spaces_correct.append(token)

        assert len(spaces_correct) == len(tokenized)

        return {'orig': tokenized, 'clean': spaces_correct}

