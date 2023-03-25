
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertForTokenClassification, AutoTokenizer
from .tokenizer import GECTokenizer

class _PrepSentence:
    """A single input sentence for token classification.
    Args:
        guid (:obj:`str`): Unique id for the sentence.
        words (:obj:`list` of :obj:`str`): list of words of the sentence.
        labels (:obj:`list` of :obj:`str`): The labels for each word
            of the sentence.
    """

    def __init__(self, guid, subwords, labels):
        self.guid = guid
        self.subwords = subwords
        self.labels = labels



def _prepare_sentences(tokenizer, sentences):
    """
    Encapsulates the input sentences into PrepSentence
    objects.
    Args:
        sentences (:obj:`list` of :obj:`list` of :obj: `str): The input
            sentences.
    Returns:
    :obj:`list` of :obj:`PrepSentence`: The list of PrepSentence objects.
    """

    guid_index = 1
    prepared_sentences = []


    tokenized_sents = [tokenizer.tokenize(sent) for sent in sentences]

    for i, subwords in enumerate(tokenized_sents):
        labels = ['["Rule", {"edits": [["UC"]]}, {}]'] * len(subwords)
        prepared_sentences.append(_PrepSentence(guid=f"{guid_index}",
                                  subwords=subwords,
                                  labels=labels))
        guid_index += 1

    return prepared_sentences


class GECTokenClassificationDataset(Dataset):

    def __init__(self, sentences, tokenizer, labels, max_seq_length):
        prepared_sentences = _prepare_sentences(tokenizer, sentences)
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.
        self.pad_token_label_id = nn.CrossEntropyLoss().ignore_index
        self.features = self._featurize_input(
            prepared_sentences,
            labels,
            max_seq_length,
            tokenizer,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            pad_token_label_id=self.pad_token_label_id,
        )


    def _featurize_input(self, prepared_sentences, label_list, max_seq_length,
                        tokenizer, cls_token="[CLS]", cls_token_segment_id=0,
                        sep_token="[SEP]", pad_token=0, pad_token_segment_id=0,
                        pad_token_label_id=-100, sequence_a_segment_id=0,
                        mask_padding_with_zero=True):
        """Featurizes the input which will be fed to the fine-tuned BERT model.
        Args:
            prepared_sentences (:obj:`list` of :obj:`PrepSentence`): list of
                PrepSentence objects.
            label_list (:obj:`list` of :obj:`str`): The labels which the model
                was trained to classify.
            max_seq_length (:obj:`int`):  Maximum sequence length.
            tokenizer (:obj:`PreTrainedTokenizer`): Bert's pretrained
                tokenizer.
            cls_token (:obj:`str`): BERT's CLS token. Defaults to [CLS].
            cls_token_segment_id (:obj:`int`): BERT's CLS token segment id.
                Defaults to 0.
            sep_token (:obj:`str`): BERT's CLS token. Defaults to [SEP].
            pad_token (:obj:`int`): BERT's pading token. Defaults to 0.
            pad_token_segment_id (:obj:`int`): BERT's pading token segment id.
                Defaults to 0.
            pad_token_label_id (:obj:`int`): BERT's pading token label id.
                Defaults to -100.
            sequence_a_segment_id (:obj:`int`): BERT's segment id.
                Defaults to 0.
            mask_padding_with_zero (:obj:`bool`): Whether to masks the padding
                tokens with zero or not. Defaults to True.
        Returns:
            obj:`list` of :obj:`Dict`: list of dicts of the needed features.
        """
        label_map = {label: i for i, label in enumerate(label_list)}
        features = []

        for example in prepared_sentences:
            tokens = example.subwords

            label_ids = []

            for label in example.labels:
                label_ids.append(label_map[label])

            assert len(label_ids) == len(tokens)


            special_tokens_count = tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]


            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            features.append({'input_ids': torch.tensor(input_ids),
                            'attention_mask': torch.tensor(input_mask),
                            'token_type_ids': torch.tensor(segment_ids),
                            'label_ids': torch.tensor(label_ids)}
                            )

        return features


    def __len__(self):
        return len(self.features)


    def __getitem__(self, i):
        return self.features[i]



class GECTagger:
    """The GEC tagger object.
    Args:
        model_path (:obj:`str`): The path to the fine-tuned model.
        use_gpu (:obj:`bool`, optional): The flag to use a GPU or not.
            Defaults to True.
    """

    def __init__(self, model_path, use_gpu=True):
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.my_tokenizer = GECTokenizer(model_path)
        self.labels_map = self.model.config.id2label
        self.device = ('cuda' if use_gpu and torch.cuda.is_available()
                       else 'cpu')
        self.model.to(self.device)
        self.model.eval()


    def _align_predictions(self, predictions, label_ids):
        """Aligns the predictions of the model with the inputs and it takes
        care of getting rid of the padding token.
        Args:
            predictions (:obj:`np.ndarray`): The predictions of the model
            label_ids (:obj:`np.ndarray`): The label ids of the inputs.
                They will always be the ids of Os since we're dealing with a
                test dataset. Note that label_ids are also padded.
            sent_ids (:obj:`np.ndarray`): The sent ids of the inputs.
        Returns:
            :obj:`list` of :obj:`list` of :obj:`str`: The predicted labels for
            all the sentences in the batch
        """

        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        preds_list = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    preds_list[i].append(self.labels_map[preds[i][j]])

        # Collating the predicted labels based on the sentence ids
        # final_preds_list = [[] for _ in range(len(set(sent_ids)))]
        # for i, id in enumerate(sent_ids):
        #     final_preds_list[id].extend(preds_list[i])

        return preds_list

    def predict(self, sentences, batch_size=32):
        """Predict the word-level ged labels of a list of sentences.
        Args:
            sentences (:obj:`list` of :obj:`str`): The input
                sentences.
            batch_size (:obj:`int`): The batch size.
        Returns:
            :obj:`list` of :obj:`list` of :obj:`str`: The predicted ged
            labels for the given sentences.
        """

        if len(sentences) == 0:
            return []

        test_dataset = GECTokenClassificationDataset(sentences=sentences,
                                                     tokenizer=self.tokenizer,
                                                     labels=list(self.labels_map.values()),
                                                     max_seq_length=256)

        data_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, drop_last=False)

        label_ids = None
        preds = None

        subwords = []
        word_ids = []

        with torch.no_grad():
            for batch in data_loader:
                inputs = {'input_ids': batch['input_ids'],
                          'token_type_ids': batch['token_type_ids'],
                          'attention_mask': batch['attention_mask']}

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                label_ids = (batch['label_ids'] if label_ids is None
                             else torch.cat((label_ids, batch['label_ids'])))

                logits = self.model(**inputs)[0]

                preds = logits if preds is None else torch.cat((preds, logits),
                                                               dim=0)



        predictions = self._align_predictions(preds.cpu().numpy(),
                                              label_ids.cpu().numpy())


        return predictions

    def predict_sentence(self, sentence):
        """Predict the ged labels of a single sentence.
        Args:
            sentence (:obj:`list` of :obj:`str`): The input sentence.
        Returns:
            :obj:`list` of :obj:`str`: The predicted ged
            labels for the given sentence.
        """

        subwords, word_ids = prepare_input(self.my_tokenizer, sentence)

        predictions = self.predict([sentence])[0]

        assert len(predictions) == len(word_ids) == len(subwords['orig']) == len(subwords['clean'])

        return {'subwords_model': subwords['orig'],
                'subwords': subwords['clean'],
                'word_ids': word_ids,
                'preds': predictions
                }



def prepare_input(tokenizer, sent):
    tokenized_sent = tokenizer._postprocess_bert_like(tokenizer.tokenize(sent))
    word_ids = tokenizer.get_word_ids(sent)

    return tokenized_sent, word_ids


# if __name__ == '__main__':
#     tagger = GECTagger('/scratch/ba63/gec/models/rules_tagger/w_camelira/all')
#     sent = 'انا احب الطعام كثيييرا'
#     preds = tagger.predict_sentence(sent)
#     x = 10
#     # sent = _prepare_sentences(tagger.tokenizer, [sent])
#     # import pdb; pdb.set_trace()
#     # x = 10