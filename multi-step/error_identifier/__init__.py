import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertForTokenClassification, BertTokenizer

GED_LABELS = ['DELETE', 'MERGE-B', 'MERGE-I', 'REPLACE_M', 'REPLACE_MI',
                'REPLACE_MI+REPLACE_OH', 'REPLACE_MT', 'REPLACE_O', 'REPLACE_OA',
                'REPLACE_OA+REPLACE_OH',
                'REPLACE_OA+REPLACE_OR',
                'REPLACE_OC',
                'REPLACE_OD',
                'REPLACE_OD+REPLACE_OG',
                'REPLACE_OD+REPLACE_OH', 'REPLACE_OD+REPLACE_OM',
                'REPLACE_OD+REPLACE_OR', 'REPLACE_OH',
                'REPLACE_OH+REPLACE_OM',
                'REPLACE_OH+REPLACE_OT',
                'REPLACE_OH+REPLACE_XC', 'REPLACE_OM',
                'REPLACE_OM+REPLACE_OR', 'REPLACE_OR',
                'REPLACE_OR+REPLACE_OT',
                'REPLACE_OT',
                'REPLACE_OW',
                'REPLACE_PC',
                'REPLACE_PM',
                'REPLACE_PT',
                'REPLACE_S',
                'REPLACE_SF',
                'REPLACE_SW',
                'REPLACE_X',
                'REPLACE_XC',
                'REPLACE_XC+REPLACE_XG',
                'REPLACE_XC+REPLACE_XN',
                'REPLACE_XF',
                'REPLACE_XG',
                'REPLACE_XM',
                'REPLACE_XN',
                'REPLACE_XT',
                'SPLIT',
                'UC']

class _PrepSentence:
    """A single input sentence for token classification.
    Args:
        guid (:obj:`str`): Unique id for the sentence.
        words (:obj:`list` of :obj:`str`): list of words of the sentence.
        labels (:obj:`list` of :obj:`str`): The labels for each word
            of the sentence.
    """

    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels


def _prepare_sentences(sentences):
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

    for words in sentences:
        labels = ['UC']*len(words)
        prepared_sentences.append(_PrepSentence(guid=f"{guid_index}",
                                  words=words,
                                  labels=labels))
        guid_index += 1

    return prepared_sentences


class TokenClassificationDataset(Dataset):
    """TokenClassificationDataset PyTorch Dataset
    Args:
        sentences (:obj:`list` of :obj:`list` of :obj:`str`): The input
            sentences.
        tokenizer (:obj:`PreTrainedTokenizer`): Bert's pretrained tokenizer.
        labels (:obj:`list` of :obj:`str`): The labels which the model was
            trained to classify.
        max_seq_length (:obj:`int`):  Maximum sentence length.
    """

    def __init__(self, sentences, tokenizer, labels, max_seq_length):
        prepared_sentences = _prepare_sentences(sentences)
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

        for sent_id, sentence in enumerate(prepared_sentences):
            tokens = []
            label_ids = []

            for word, label in zip(sentence.words, sentence.labels):
                word_tokens = tokenizer.tokenize(word)
                # bert-base-multilingual-cased sometimes output "nothing ([])
                # when calling tokenize with just a space.
                if len(word_tokens) > 0:
                    tokens.append(word_tokens)
                    # Use the real label id for the first token of the word,
                    # and padding ids for the remaining tokens
                    label_ids.append([label_map[label]] +
                                     [pad_token_label_id] *
                                     (len(word_tokens) - 1))

            token_segments = []
            token_segment = []
            label_ids_segments = []
            label_ids_segment = []
            num_word_pieces = 0
            seg_seq_length = max_seq_length - 2

            # Dealing with empty sentences
            if len(tokens) == 0:
                data = self._add_special_tokens(token_segment,
                                                label_ids_segment,
                                                tokenizer,
                                                max_seq_length,
                                                cls_token,
                                                sep_token, pad_token,
                                                cls_token_segment_id,
                                                pad_token_segment_id,
                                                pad_token_label_id,
                                                sequence_a_segment_id,
                                                mask_padding_with_zero)
                # Adding sentence id
                data['sent_id'] = sent_id
                features.append(data)
            else:
                # Chunking the tokenized sentence into multiple segments
                # if it's longer than max_seq_length - 2
                for idx, word_pieces in enumerate(tokens):
                    if num_word_pieces + len(word_pieces) > seg_seq_length:
                        data = self._add_special_tokens(token_segment,
                                                        label_ids_segment,
                                                        tokenizer,
                                                        max_seq_length,
                                                        cls_token,
                                                        sep_token, pad_token,
                                                        cls_token_segment_id,
                                                        pad_token_segment_id,
                                                        pad_token_label_id,
                                                        sequence_a_segment_id,
                                                        mask_padding_with_zero)
                        # Adding sentence id
                        data['sent_id'] = sent_id
                        features.append(data)

                        token_segments.append(token_segment)
                        label_ids_segments.append(label_ids_segment)
                        token_segment = list(word_pieces)
                        label_ids_segment = list(label_ids[idx])
                        num_word_pieces = len(word_pieces)
                    else:
                        token_segment.extend(word_pieces)
                        label_ids_segment.extend(label_ids[idx])
                        num_word_pieces += len(word_pieces)

                # Adding the last segment
                if len(token_segment) > 0:
                    data = self._add_special_tokens(token_segment,
                                                    label_ids_segment,
                                                    tokenizer,
                                                    max_seq_length,
                                                    cls_token,
                                                    sep_token, pad_token,
                                                    cls_token_segment_id,
                                                    pad_token_segment_id,
                                                    pad_token_label_id,
                                                    sequence_a_segment_id,
                                                    mask_padding_with_zero)
                    # Adding sentence id
                    data['sent_id'] = sent_id
                    features.append(data)

                    token_segments.append(token_segment)
                    label_ids_segments.append(label_ids_segment)

                # DEBUG: Making sure we got all segments correctly
                # assert sum([len(_) for _ in label_ids_segments]) == \
                #        sum([len(_) for _ in label_ids])

                # assert sum([len(_) for _ in token_segments]) == \
                #        sum([len(_) for _ in tokens])

        return features

    def _add_special_tokens(self, tokens, label_ids, tokenizer, max_seq_length,
                            cls_token, sep_token, pad_token,
                            cls_token_segment_id, pad_token_segment_id,
                            pad_token_label_id, sequence_a_segment_id,
                            mask_padding_with_zero):

        _tokens = list(tokens)
        _label_ids = list(label_ids)

        _tokens += [sep_token]
        _label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(_tokens)

        _tokens = [cls_token] + _tokens
        _label_ids = [pad_token_label_id] + _label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only
        # real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        _label_ids += [pad_token_label_id] * padding_length

        return {'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(input_mask),
                'token_type_ids': torch.tensor(segment_ids),
                'label_ids': torch.tensor(_label_ids)}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

class ErrorIdentifier:
    """The Error Identifier object.
    Args:
        model_path (:obj:`str`): The path to the fine-tuned model.
        use_gpu (:obj:`bool`, optional): The flag to use a GPU or not.
            Defaults to True.
    """

    def __init__(self, model_path, use_gpu=True):
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.labels_map = self.model.config.id2label
        self.device = ('cuda' if use_gpu and torch.cuda.is_available()
                       else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def labels():
        """Get the list of word-level ged labels returned by predictions.
        Returns:
            :obj:`list` of :obj:`str`: List of word-level ged labels.
        """
        return GED_LABELS

    def _align_predictions(self, predictions, label_ids, sent_ids):
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
        final_preds_list = [[] for _ in range(len(set(sent_ids)))]
        for i, id in enumerate(sent_ids):
            final_preds_list[id].extend(preds_list[i])

        return final_preds_list

    def predict(self, sentences, batch_size=32):
        """Predict the word-level ged labels of a list of sentences.
        Args:
            sentences (:obj:`list` of :obj:`list` of :obj:`str`): The input
                sentences.
            batch_size (:obj:`int`): The batch size.
        Returns:
            :obj:`list` of :obj:`list` of :obj:`str`: The predicted ged
            labels for the given sentences.
        """

        if len(sentences) == 0:
            return []

        test_dataset = TokenClassificationDataset(sentences=sentences,
                                        tokenizer=self.tokenizer,
                                        labels=list(self.labels_map.values()),
                                        max_seq_length=256)

        data_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, drop_last=False)

        label_ids = None
        preds = None
        sent_ids = None

        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                inputs = {'input_ids': batch['input_ids'],
                          'token_type_ids': batch['token_type_ids'],
                          'attention_mask': batch['attention_mask']}

                label_ids = (batch['label_ids'] if label_ids is None
                             else torch.cat((label_ids, batch['label_ids'])))
                sent_ids = (batch['sent_id'] if sent_ids is None
                            else torch.cat((sent_ids, batch['sent_id'])))

                logits = self.model(**inputs)[0]

                preds = logits if preds is None else torch.cat((preds, logits),
                                                               dim=0)

        predictions = self._align_predictions(preds.cpu().numpy(),
                                              label_ids.cpu().numpy(),
                                              sent_ids.cpu().numpy())

        return predictions

    def predict_sentence(self, sentence):
        """Predict the ged labels of a single sentence.
        Args:
            sentence (:obj:`list` of :obj:`str`): The input sentence.
        Returns:
            :obj:`list` of :obj:`str`: The predicted ged
            labels for the given sentence.
        """

        return self.predict([sentence])[0]