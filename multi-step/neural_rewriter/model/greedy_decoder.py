import torch
import torch.nn.functional as F
import re

class BatchSampler:
    def __init__(self, model, src_vocab_char,
                 src_vocab_word, trg_vocab_char,
                 ged_tag_vocab):

        self.model = model
        self.src_vocab_char = src_vocab_char
        self.src_vocab_word = src_vocab_word
        self.trg_vocab_char = trg_vocab_char
        self.ged_tag_vocab = ged_tag_vocab

    def set_batch(self, batch):
        self.sample_batch = batch

    def get_trg_token(self, index):
        trg_token = self.sample_batch['trg_y'][index].cpu().detach().numpy()
        return self.get_str(trg_token, self.trg_vocab_char)

    def get_src_token(self, index):
        src_token = self.sample_batch['src_char'][index].cpu().detach().numpy()
        return self.get_str(src_token, self.src_vocab_char)

    def get_str(self, vectorized_token, vocab):
        token = []
        for i in vectorized_token:
            if i == vocab.sos_idx:
                continue
            elif i == vocab.eos_idx:
                break
            else:
                token.append(vocab.lookup_index(i))
        return ''.join(token)

    def get_ged_tag(self, index):
        ged_tag = self.sample_batch['ged_tags'][index].cpu().detach().numpy().tolist()
        return self.ged_tag_vocab.lookup_index(ged_tag)

    def greedy_decode(self, token, add_side_constraints=False, max_len=512):

        # vectorizing the src token on the char level and word level
        if add_side_constraints:
            sc = token[:token.rfind('>')+1]
            token = token[token.rfind('>')+1:]

        vectorized_src_token_char = [self.src_vocab_char.sos_idx]
        vectorized_src_token_word = [self.src_vocab_word.sos_idx]

        if add_side_constraints:
            vectorized_src_token_char.append(self.src_vocab_char.lookup_token(sc))
            vectorized_src_token_word.append(self.src_vocab_word.lookup_token(sc))

        for c in token:
            vectorized_src_token_char.append(self.src_vocab_char.lookup_token(c))
            vectorized_src_token_word.append(self.src_vocab_word.lookup_token(token))

        vectorized_src_token_word.append(self.src_vocab_word.eos_idx)
        vectorized_src_token_char.append(self.src_vocab_char.eos_idx)

        # getting token length
        src_token_length = [len(vectorized_src_token_char)]

        # converting the lists to tensors
        vectorized_src_token_char = torch.tensor([vectorized_src_token_char], dtype=torch.long)
        vectorized_src_token_word = torch.tensor([vectorized_src_token_word], dtype=torch.long)
        src_token_length = torch.tensor(src_token_length, dtype=torch.long)

        # passing the src sequence to the encoder
        with torch.no_grad():
            encoder_outputs, encoder_h_t = self.model.encoder(vectorized_src_token_char,
                                                              vectorized_src_token_word,
                                                              src_token_length
                                                              )

        # creating attention mask
        attention_mask = self.model.create_mask(vectorized_src_token_char, self.src_vocab_char.pad_idx)

        # initializing the first decoder_h_t to encoder_h_t
        decoder_h_t = encoder_h_t

        context_vectors = torch.zeros(1, self.model.encoder.rnn.hidden_size * 2)

        # intializing the trg sequences to the <s> token
        trg_seqs = [self.trg_vocab_char.sos_idx]

        with torch.no_grad():
            for i in range(max_len):
                y_t = torch.tensor([trg_seqs[-1]], dtype=torch.long)

                # do a single decoder step
                prediction, decoder_h_t, atten_scores, context_vectors = self.model.decoder(trg_seqs=y_t,
                                                                                            encoder_outputs=encoder_outputs,
                                                                                            decoder_h_t=decoder_h_t,
                                                                                            context_vectors=context_vectors,
                                                                                            attention_mask=attention_mask
                                                                                            )

                # getting the most probable prediciton
                max_pred = torch.argmax(prediction, dim=1).item()

                # if we reach </s> token, stop decoding
                if max_pred == self.trg_vocab_char.eos_idx:
                    break

                trg_seqs.append(max_pred)

        str_token = self.get_str(trg_seqs, self.trg_vocab_char)
        return str_token
