import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
from .attention import AdditiveAttention


class Encoder(nn.Module):
    """Encoder bi-GRU"""
    def __init__(self, input_dim, char_embed_dim,
                 encoder_hidd_dim,
                 decoder_hidd_dim,
                 num_layers,
                 morph_embeddings=None,
                 char_padding_idx=0,
                 word_padding_idx=0,
                 dropout=0):

        super(Encoder, self).__init__()
        morph_embeddings_dim = 0
        self.morph_embedding_layer = None

        self.char_embedding_layer = nn.Embedding(input_dim,
                                                 char_embed_dim,
                                                 padding_idx=char_padding_idx)

        if morph_embeddings is not None:
            self.morph_embedding_layer = nn.Embedding.from_pretrained(morph_embeddings,
                                                                      padding_idx=word_padding_idx)
            morph_embeddings_dim = morph_embeddings.shape[1]

        self.rnn = nn.GRU(input_size=char_embed_dim + morph_embeddings_dim,
                          hidden_size=encoder_hidd_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)

        self.linear_map = nn.Linear(encoder_hidd_dim * 2, decoder_hidd_dim)

    def forward(self, char_src_seqs, word_src_seqs, src_seqs_lengths):

        embedded_seqs = self.char_embedding_layer(char_src_seqs)
        # embedded_seqs shape: [batch_size, max_src_seq_len, char_embed_dim]

        # Add morph embeddings to the char embeddings if needed
        if self.morph_embedding_layer is not None:
            embedded_word_seqs_morph = self.morph_embedding_layer(word_src_seqs)
            # embedded_word_seqs_morph shape: [batch_size, max_src_seq_len, morph_embeddings_dim]

            embedded_seqs = torch.cat((embedded_seqs, embedded_word_seqs_morph), dim=2)
            # embedded_seqs shape: [batch_size, max_src_seq_len, char_embed_dim + morph_embeddings_dim]

        # packing the embedded_seqs
        packed_embedded_seqs = pack_padded_sequence(embedded_seqs, src_seqs_lengths, batch_first=True)

        output, hidd = self.rnn(packed_embedded_seqs)
        # hidd shape: [num_layers * num_dirs, batch_size, encoder_hidd_dim]

        # concatenating the forward and backward vectors for each layer
        hidd = torch.cat([hidd[0:hidd.size(0):2], hidd[1:hidd.size(0):2]], dim=2)
        # hidd shape: [num layers, batch_size, num_directions * encoder_hidd_dim]

        # mapping the encode hidd state to the decoder hidd dim space
        hidd = torch.tanh(self.linear_map(hidd))

        # unpacking the output
        output, lengths = pad_packed_sequence(output, batch_first=True)
        # output shape: [batch_size, src_seqs_length, num_dirs * encoder_hidd_dim]
        return output, hidd


class Decoder(nn.Module):
    """Decoder GRU

       Things to note:
           - The input to the decoder rnn at each time step is the
             concatenation of the embedded token and the context vector
           - The context vector will have a size of batch_size, encoder_hidd_dim * 2
           - The prediction layer input is the concatenation of
             the context vector and the h_t of the decoder
    """
    def __init__(self, input_dim, char_embed_dim,
                 decoder_hidd_dim, num_layers,
                 output_dim,
                 encoder_hidd_dim,
                 padding_idx=0,
                 dropout=0):

        super(Decoder, self).__init__()

        self.attention = AdditiveAttention(encoder_hidd_dim=encoder_hidd_dim,
                                           decoder_hidd_dim=decoder_hidd_dim)

        self.char_embedding_layer = nn.Embedding(input_dim,
                                                 char_embed_dim,
                                                 padding_idx=padding_idx)

        # the input to the rnn is the context_vector + embedded token --> embed_dim + hidd_dim
        self.rnn = nn.GRU(input_size=char_embed_dim + encoder_hidd_dim * 2,
                          hidden_size=decoder_hidd_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)

        # the input to the classifier is h_t + context_vector --> hidd_dim * 2
        self.classification_layer = nn.Linear(encoder_hidd_dim * 2
                                              + decoder_hidd_dim * num_layers
                                              + char_embed_dim,
                                              output_dim)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, trg_seqs, encoder_outputs, decoder_h_t, context_vectors,
                attention_mask):
        # trg_seqs shape: [batch_size]
        batch_size = trg_seqs.shape[0]

        trg_seqs = trg_seqs.unsqueeze(1)
        # trg_seqs shape: [batch_size, 1]

        # Step 1: embedding the target seqs
        embedded_seqs = self.char_embedding_layer(trg_seqs)
        # embedded_seqs shape: [batch_size, 1, embed_dim]

        # context_vectors shape: [batch_size, encoder_hidd_dim * 2]
        # changing shape to: [batch_size, 1, encoder_hidd_dim * 2]
        context_vectors = context_vectors.unsqueeze(1)

        # concatenating the embedded trg sequence with the context_vectors
        rnn_input = torch.cat((embedded_seqs, context_vectors), dim=2)
        # rnn_input shape: [batch_size, 1, embed_dim + encoder_hidd_dim * 2]

        # Step 2: feeding the input to the rnn and updating the decoder_h_t
        decoder_output, decoder_h_t = self.rnn(rnn_input, decoder_h_t)
        # decoder output shape: [batch_size, 1, num_dirs * hidd_dim]
        # decoder_h_t shape: [num_layers * num_dirs, batch_size, hidd_dim]


        # Step 3: updating the context vectors through attention
        context_vectors, atten_scores = self.attention(keys=encoder_outputs,
                                                       query=decoder_h_t,
                                                       mask=attention_mask)

        # Step 4: get the prediction vector

        # concatenating decoder_h_t with context_vectors to
        # create a prediction vector
        predictions_vector = torch.cat((decoder_h_t.view(decoder_h_t.shape[1], -1),
                                        context_vectors, embedded_seqs.squeeze(1)),
                                        dim=1)
        # predictions_vector: [batch_size, hidd_dim + encoder_hidd_dim * 2]

        # Step 5: feeding the prediction vector to the fc layer
        # to a make a prediction

        # apply dropout if needed
        predictions_vector = self.dropout_layer(predictions_vector)
        prediction = self.classification_layer(predictions_vector)
        # prediction shape: [batch_size, output_dim]

        return prediction, decoder_h_t, atten_scores, context_vectors

class Seq2Seq(nn.Module):
    """Seq2Seq model"""
    def __init__(self, encoder_input_dim, encoder_embed_dim,
                 encoder_hidd_dim, encoder_num_layers,
                 decoder_input_dim, decoder_embed_dim,
                 decoder_hidd_dim, decoder_num_layers,
                 decoder_output_dim,
                 morph_embeddings=None,
                 char_src_padding_idx=0,
                 word_src_padding_idx=0, trg_padding_idx=0,
                 dropout=0, trg_sos_idx=2):

        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(input_dim=encoder_input_dim,
                               char_embed_dim=encoder_embed_dim,
                               encoder_hidd_dim=encoder_hidd_dim,
                               decoder_hidd_dim=decoder_hidd_dim,
                               num_layers=encoder_num_layers,
                               morph_embeddings=morph_embeddings,
                               char_padding_idx=char_src_padding_idx,
                               word_padding_idx=word_src_padding_idx,
                               dropout=dropout)

        self.decoder = Decoder(input_dim=decoder_input_dim,
                               char_embed_dim=decoder_embed_dim,
                               decoder_hidd_dim=decoder_hidd_dim,
                               num_layers=decoder_num_layers,
                               encoder_hidd_dim=encoder_hidd_dim,
                               output_dim=decoder_input_dim,
                               padding_idx=trg_padding_idx,
                               dropout=dropout)

        self.char_src_padding_idx = char_src_padding_idx
        self.trg_sos_idx = trg_sos_idx
        self.sampling_temperature = 3

    def create_mask(self, src_seqs, src_padding_idx):
        mask = (src_seqs != src_padding_idx)
        return mask

    def forward(self, char_src_seqs, word_src_seqs, src_seqs_lengths, trg_seqs,
                teacher_forcing_prob=0.3):
        # trg_seqs shape: [batch_size, trg_seqs_length]
        # reshaping to: [trg_seqs_length, batch_size]
        trg_seqs = trg_seqs.permute(1, 0)
        trg_seqs_length, batch_size = trg_seqs.shape

        # passing the src to the encoder
        encoder_outputs, encoder_hidd = self.encoder(char_src_seqs,
                                                     word_src_seqs,
                                                     src_seqs_lengths)

        # creating attention masks
        attention_mask = self.create_mask(char_src_seqs,
                                          self.char_src_padding_idx)

        predictions = []
        decoder_attention_scores = []

        # initializing the trg_seqs to <s> token
        y_t = torch.ones(batch_size, dtype=torch.long) * self.trg_sos_idx

        # intializing the context_vectors to zero
        context_vectors = torch.zeros(batch_size, self.encoder.rnn.hidden_size * 2)
        # context_vectors shape: [batch_size, encoder_hidd_dim * 2]

        # initializing the hidden state of the decoder to the encoder hidden state
        decoder_h_t = encoder_hidd
        # decoder_h_t shape: [batch_size, decoder_hidd_dim]

        # moving y_t and context_vectors to the right device
        y_t = y_t.to(encoder_hidd.device)
        context_vectors = context_vectors.to(encoder_hidd.device)

        for i in range(0, trg_seqs_length):

            teacher_forcing = np.random.random() < teacher_forcing_prob
            # if teacher_forcing, use ground truth target tokens
            # as an input to the decoder
            if teacher_forcing:
                y_t = trg_seqs[i]

            # do a single decoder step
            prediction, decoder_h_t, atten_scores, context_vectors = self.decoder(trg_seqs=y_t,
                                                                                  encoder_outputs=encoder_outputs,
                                                                                  decoder_h_t=decoder_h_t,
                                                                                  context_vectors=context_vectors,
                                                                                  attention_mask=attention_mask)


            # If not teacher force, use the maximum 
            # prediction as an input to the decoder in 
            # the next time step
            if not teacher_forcing:
                # we multiply the predictions with a sampling_temperature
                # to make the probablities peakier, so we can be confident about the
                # maximum prediction
                pred_output_probs = F.softmax(prediction * self.sampling_temperature, dim=1)
                y_t = torch.argmax(pred_output_probs, dim=1)

            predictions.append(prediction)
            decoder_attention_scores.append(atten_scores)

        predictions = torch.stack(predictions)
        # predictions shape: [trg_seq_len, batch_size, output_dim]
        predictions = predictions.permute(1, 0, 2)
        # predictions shape: [batch_size, trg_seq_len, output_dim]

        decoder_attention_scores = torch.stack(decoder_attention_scores)
        # attention_scores_total shape: [trg_seq_len, batch_size, src_seq_len]
        decoder_attention_scores = decoder_attention_scores.permute(1, 0, 2)
        # attention_scores_total shape: [batch_size, trg_seq_len, src_seq_len]

        return predictions, decoder_attention_scores

    def serialize_model_args(self):
        return {'encoder_input_dim': self.encoder.input_dim,
                'encoder_embed_dim': self.encoder.char_embed_dim,
                'encoder_hidd_dim': self.encoder.encoder_hidd_dim,
                'encoder_num_layers': self.encoder.num_layers,
                'decoder_input_dim': self.decoder.input_dim,
                'decoder_embed_dim': self.decoder.char_embed_dim,
                'decoder_hidd_dim': self.decoder.decoder_hidd_dim,
                'decoder_num_layers': self.decoder.num_layers,
                'decoder_output_dim': self.decoder.output_dim,
                'morph_embeddings':  self.encoder.morph_embeddings,
                'char_src_padding_idx': self.encoder.char_src_padding_idx,
                'word_src_padding_idx': self.encoder.word_src_padding_idx,
                'trg_padding_idx': self.decoder.padding_idx,
                'dropout': self.encoder.dropout,
                'trg_sos_idx': self.decoder.trg_sos_idx
                }


