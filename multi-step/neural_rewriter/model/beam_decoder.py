from .greedy_decoder import BatchSampler
from queue import PriorityQueue
import torch
import torch.nn.functional as F

class BeamSearchNode:
    """A class to represent the node during the beam search"""
    def __init__(self, hidd_state, prev_node, word_idx, log_prob, length):
        """
        Args:
            hidd_state: decoder hidden state
            prev_node: the previous node (parent)
            word_idx: the word index
            log_prob: the log probability
            length: length of decoded token
        """
        self.h = hidd_state
        self.prevNode = prev_node
        self.wordid = word_idx
        self.logp = log_prob
        self.leng = length

    def eval(self, alpha=1):
        reward = 0
        # Add here a function for shaping a reward
        # the log prob will be normalized by the length of the token
        # as defined by Wu et. al: https://arxiv.org/pdf/1609.08144.pdf
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
        #return self.logp / float(self.leng)**alpha

    def __lt__(self, other):
       """Overriding the less than function to handle
       the case if two nodes have the same log_prob so
       they can fit in the priority queue"""
       return self.logp < other.logp

class BeamSampler(BatchSampler):
    """A subclass of BatchSampler that uses beam_search for decoding"""
    def __init__(self, model, src_vocab_char,
                 src_vocab_word, trg_vocab_char,
                 ged_tag_vocab, beam_width=10, topk=3, device='cpu'):

        super(BeamSampler, self).__init__(model, src_vocab_char,
                                          src_vocab_word, trg_vocab_char,
                                          ged_tag_vocab)
        self.beam_width = beam_width
        self.topk = topk
        self.device = device

    def beam_decode(self, token, add_side_constraints=False,
                    max_len=512):
        """
        Args:
            token: the source token
            topk: number of tokens to generate from beam search. Defaults to 3
            beam_width: the beam size. If 1, then we do greed search. Defaults to 5
            max_len: the maximum length of the decoded token. Defaults to 512

        Returns:
            decoded_tokens: list of tuples. Each tuple is (log_prob, decoded_token)
        """

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
        vectorized_src_token_char = torch.LongTensor([vectorized_src_token_char]).to(self.device)
        vectorized_src_token_word = torch.LongTensor([vectorized_src_token_word]).to(self.device)
        src_token_length = torch.LongTensor(src_token_length)


        # passing the src token to the encoder
        with torch.no_grad():
            encoder_outputs, encoder_h_t = self.model.encoder(vectorized_src_token_char,
                                                             vectorized_src_token_word,
                                                             src_token_length)

        # creating attention mask
        attention_mask = self.model.create_mask(vectorized_src_token_char, self.src_vocab_char.pad_idx)

        # initializing the first decoder_h_t to encoder_h_t
        decoder_hidden = encoder_h_t
        #decoder_hidden = torch.tanh(self.model.linear_map(encoder_h_t))

        context_vectors = torch.zeros(1, self.model.encoder.rnn.hidden_size * 2).to(self.device)

        # topk must be <= beam_width
        if self.topk > self.beam_width:
            raise Exception("topk candidates must be <= beam_width")

        decoded_batch = []

        # starting input to the decoder is the <s> token
        decoder_input = torch.LongTensor([self.trg_vocab_char.sos_idx]).to(self.device)

        # number of tokens to generate
        endnodes = []
        number_required = min((self.topk + 1), self.topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        # each element in the queue will be (-log_prob, beam_node)
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while max_len > 0:
            max_len -= 1
            # give up when decoding takes too long
            if qsize > 20000:
                print('hiiii')
                break

            # fetch the best node (i.e. node with minimum negative log prob)
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            # if we predict the </s> token, this means we finished decoding a token
            if n.wordid.item() == self.trg_vocab_char.eos_idx and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of tokens required, stop beam search
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            with torch.no_grad():
                decoder_output, decoder_hidden, atten_scores, context_vectors = self.model.decoder(trg_seqs=decoder_input,
                                                                                                   encoder_outputs=encoder_outputs,
                                                                                                   decoder_h_t=decoder_hidden,
                                                                                                   context_vectors=context_vectors,
                                                                                                   attention_mask=attention_mask
                                                                                                   )

            # obtaining log probs from the decoder predictions
            decoder_output = F.log_softmax(decoder_output, dim=1)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, self.beam_width)
            # indexes shape: [batch_size, beam_width]
            # log_prob shape: [batch_size, beam_width]

            # expanding the current beam (n)
            nextnodes = []

            for new_k in range(self.beam_width):
                decoded_t = indexes[0][new_k].unsqueeze(0)
                log_p = log_prob[0][new_k].item()
                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put the expanded beams in the queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))

            # increase qsize
            qsize += len(nextnodes) - 1

        # choose topk beams
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(self.topk)]

        # sorting the topk beams by their negative log probs
        endnodes = sorted(endnodes, key=lambda x: x[0])

        # decoding
        #TODO: Decoding currently works for one token at a time,
        #Bashar needs to make it work on the batch
        decoded_tokens = []
        for score, n in endnodes:
            decoded_token = []
            decoded_token.append(n.wordid.item())
            # backtrack 
            while n.prevNode != None:
                n = n.prevNode
                decoded_token.append(n.wordid.item())
            # reversing the decoding
            decoded_token = decoded_token[::-1]
            decoded_tokens.append((score, decoded_token))

        str_decoded_token = self.get_str(decoded_tokens[0][1], self.trg_vocab_char)
        str_decoded_tokens = [self.get_str(d_token[1], self.trg_vocab_char)
                              for d_token in decoded_tokens]

        return str_decoded_tokens

