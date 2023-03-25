import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from utils.data_utils import RawDataset
from utils.data_utils import Vectorizer
from utils.metrics import  accuracy_top_n
import json
import random
import numpy as np
import argparse
from model.seq2seq import Seq2Seq
from model.greedy_decoder import BatchSampler
from model.beam_decoder import BeamSampler
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MT_Dataset(Dataset):
    """MT Dataset as a PyTorch dataset"""
    def __init__(self, raw_dataset, vectorizer):
        """
        Args:
            - raw_dataset (RawDataset): raw dataset object
            - vectorizer (Vectorizer): vectorizer object
        """
        self.vectorizer = vectorizer
        self.train_examples = raw_dataset.train_examples
        self.dev_examples = raw_dataset.dev_examples
        self.test_examples = raw_dataset.test_examples
        self.lookup_split = {'train': self.train_examples,
                             'dev': self.dev_examples,
                             'test': self.test_examples}
        self.set_split('train')

    def get_vectorizer(self):
        return self.vectorizer

    @classmethod
    def load_data_and_create_vectorizer(cls, data_dir, add_side_constraints=False):
        raw_dataset = RawDataset(data_dir)
        # Note: we always create the vectorized based on the train examples
        vectorizer = Vectorizer.create_vectorizer(raw_dataset.train_examples,
                                                  add_side_constraints=add_side_constraints)
        return cls(raw_dataset, vectorizer)

    @classmethod
    def load_data_and_load_vectorizer(cls, data_dir, vec_path):
        raw_dataset = RawDataset(data_dir)
        vectorizer = cls.load_vectorizer(vec_path)
        return cls(raw_dataset, vectorizer)

    @staticmethod
    def load_vectorizer(vec_path):
        with open(vec_path) as f:
            return Vectorizer.from_serializable(json.load(f))

    def save_vectorizer(self, vec_path):
        with open(vec_path, 'w') as f:
            return json.dump(self.vectorizer.to_serializable(), f,
                             ensure_ascii=False)

    def set_split(self, split):
        self.split = split
        self.split_examples = self.lookup_split[self.split]
        return self.split_examples

    def __getitem__(self, index):
        example = self.split_examples[index]
        src_token, trg_token = example.src_token, example.trg_token
        ged_tag = example.ged_tag
        vectorized = self.vectorizer.vectorize(src_token, trg_token, ged_tag)
        return vectorized

    def __len__(self):
        return len(self.split_examples)

class Collator:
    def __init__(self, char_src_pad_idx, char_trg_pad_idx,
                 word_src_pad_idx):
        """
        Args:
            - char_src_pad_idx: source vocab padding index on the char level
            - char_trg_pad_idx: target vocab padding index on the char level
            - word_src_pad_idx: source vocab padding index on the word level
        """
        self.char_src_pad_idx = char_src_pad_idx
        self.word_src_pad_idx = word_src_pad_idx
        self.char_trg_pad_idx = char_trg_pad_idx

    def __call__(self, batch):
        # Sorting the batch by src seqs length in descending order
        sorted_batch = sorted(batch, key=lambda x: x['src_char'].shape[0],
                              reverse=True)

        src_char_seqs = [x['src_char'] for x in sorted_batch]
        src_word_seqs = [x['src_word'] for x in sorted_batch]

        assert len(src_word_seqs) == len(src_char_seqs)

        trg_x_seqs = [x['trg_x'] for x in sorted_batch]
        trg_y_seqs = [x['trg_y'] for x in sorted_batch]
        ged_tags = [x['ged_tag'] for x in sorted_batch]
        lengths = [len(seq) for seq in src_char_seqs]

        padded_src_char_seqs = pad_sequence(src_char_seqs, batch_first=True,
                                            padding_value=self.char_src_pad_idx)
        padded_src_word_seqs = pad_sequence(src_word_seqs, batch_first=True,
                                            padding_value=self.word_src_pad_idx)

        padded_trg_x_seqs = pad_sequence(trg_x_seqs, batch_first=True,
                                         padding_value=self.char_trg_pad_idx)
        padded_trg_y_seqs = pad_sequence(trg_y_seqs, batch_first=True,
                                         padding_value=self.char_trg_pad_idx)

        lengths = torch.tensor(lengths, dtype=torch.long)

        ged_tags = torch.tensor(ged_tags, dtype=torch.long)

        return {'src_char': padded_src_char_seqs,
                'src_word': padded_src_word_seqs,
                'trg_x': padded_trg_x_seqs,
                'trg_y': padded_trg_y_seqs,
                'src_lengths': lengths,
                'ged_tags': ged_tags
                }

def set_seed(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def train(model, dataloader, optimizer, criterion, device='cpu',
          teacher_forcing_prob=1, clip_grad=1.0):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}

        src_char = batch['src_char']
        src_word = batch['src_word']
        trg_x = batch['trg_x']
        trg_y = batch['trg_y']
        src_lengths = batch['src_lengths']

        preds, attention_scores = model(char_src_seqs=src_char,
                                        word_src_seqs=src_word,
                                        src_seqs_lengths=src_lengths.to('cpu'),
                                        trg_seqs=trg_x,
                                        teacher_forcing_prob=teacher_forcing_prob)

        # CrossEntropysLoss accepts matrices always! 
        # the preds must be of size (N, C) where C is the number 
        # of classes and N is the number of samples. 
        # The ground truth must be a Vector of size C!
        preds = preds.contiguous().view(-1, preds.shape[-1])
        trg_y = trg_y.view(-1)

        loss = criterion(preds, trg_y)
        epoch_loss += loss.item()
        # Backprop
        loss.backward()
        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        # Optimizer step
        optimizer.step()

    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device='cpu', teacher_forcing_prob=0):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            src_char = batch['src_char']
            src_word = batch['src_word']
            trg_x = batch['trg_x']
            trg_y = batch['trg_y']
            src_lengths = batch['src_lengths']


            preds, attention_scores = model(char_src_seqs=src_char,
                                            word_src_seqs=src_word,
                                            src_seqs_lengths=src_lengths.to('cpu'),
                                            trg_seqs=trg_x,
                                            teacher_forcing_prob=teacher_forcing_prob)

            # CrossEntropyLoss accepts matrices always! 
            # the preds must be of size (N, C) where C is the number 
            # of classes and N is the number of samples. 
            # The ground truth must be a Vector of size C!
            preds = preds.contiguous().view(-1, preds.shape[-1])
            trg_y = trg_y.view(-1)

            loss = criterion(preds, trg_y)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def inference(sampler, beam_sampler, dataloader, args):
    output_inf_file = open(args.preds_dir + '.inf.txt', mode='w',
                           encoding='utf8')
    output_beam = open(args.preds_dir + '.beam.top.1.txt', mode='w',
                           encoding='utf8')
    output_beam_top = open(args.preds_dir + '.beam.top.n.txt', mode='w',
                           encoding='utf8')

    greedy_stats = {}
    beam_stats = {}
    greedy_accuracy = 0
    beam_accuracy = 0

    for batch in dataloader:
        import pdb; pdb.set_trace()
        sampler.set_batch(batch)
        src = sampler.get_src_token(0)
        trg = sampler.get_trg_token(0)


        translated = sampler.greedy_decode(token=src,
                                           add_side_constraints=args.add_side_constraints)

        beam_trans = beam_sampler.beam_decode(token=src,
                                              add_side_constraints=args.add_side_constraints,
                                              max_len=512)


        greedy_accuracy += accuracy_top_n(trg=trg, top_n_preds=[translated])
        beam_accuracy += accuracy_top_n(trg=trg, top_n_preds=beam_trans)

        correct = 'CORRECT!' if trg == translated else 'INCORRECT!'
        # different_g = 'SAME!' if translated == beam_trans_1 else 'DIFF!'
        different = 'SAME!' if translated == beam_trans[0] else 'DIFF!'

        if translated == trg:
            greedy_stats['correct'] = 1 + greedy_stats.get('correct', 0)
        else:
            greedy_stats['incorrect'] = 1 + greedy_stats.get('incorrect', 0)

        if trg in beam_trans:
            beam_stats['correct'] = 1 + beam_stats.get('correct', 0)
        else:
            beam_stats['incorrect'] = 1 + beam_stats.get('incorrect', 0)


        output_inf_file.write(translated)
        output_inf_file.write('\n')
        output_beam.write(beam_trans[0])
        output_beam.write('\n')
        output_beam_top.write(str(beam_trans))
        output_beam_top.write('\n')

        logger.info(f'src:\t\t\t{src}')
        logger.info(f'trg:\t\t\t{trg}')
        logger.info(f'greedy:\t\t\t{translated}')
        logger.info(f'beam top 1:\t\t{beam_trans[0]}')
        logger.info(f'beam top {args.n_best}:\t\t{beam_trans}')

        logger.info(f'res:\t\t\t{correct}')
        logger.info(f'beam==greedy?:\t\t{different}')
        logger.info('\n\n')

    greedy_accuracy /= len(dataloader)
    beam_accuracy /= len(dataloader)

    output_inf_file.close()
    output_beam_top.close()
    output_beam.close()

    logger.info('*******STATS*******')
    assert sum([greedy_stats[x] for x in greedy_stats]) == sum([beam_stats[x]
                for x in beam_stats])
    total_examples = sum([greedy_stats[x] for x in greedy_stats])
    logger.info(f'TOTAL EXAMPLES: {total_examples}')
    logger.info('\n')

    total_correct_greedy = sum([v for k,v in greedy_stats.items()
                                if k == 'correct'])
    total_incorrect_greedy = sum([v for k,v in greedy_stats.items()
                                if k == 'incorrect'])

    logger.info('Results using greedy decoding:')
    logger.info(f'Total Correct: {total_correct_greedy}\t'
                f'Total Incorrect: {total_incorrect_greedy}')
    logger.info(f'Accuracy:\t{greedy_accuracy}')

    logger.info('\n')

    total_correct_beam = sum([v for k, v in beam_stats.items()
                                if k == 'correct'])
    total_incorrect_beam = sum([v for k, v in beam_stats.items()
                                if k == 'incorrect'])

    logger.info('Results using beam decoding:')
    logger.info(f'Total Correct: {total_correct_beam}\t'
                f'Total Incorrect: {total_incorrect_beam}')
    logger.info(f'Accuracy:\t{beam_accuracy}')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the src and trg files."
    )
    parser.add_argument(
        "--vectorizer_path",
        default=None,
        type=str,
        help="The path of the saved vectorizer."
    )
    parser.add_argument(
        "--cache_files",
        action="store_true",
        help="Whether to cache the vocab and the vectorizer objects or not."
    )
    parser.add_argument(
        "--reload_files",
        action="store_true",
        help="Whether to reload the vocab and the vectorizer objects"
             " from a cached file."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=20,
        type=int,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--embed_dim",
        default=32,
        type=int,
        help="The embedding dimensions of the model."
    )
    parser.add_argument(
        "--hidd_dim",
        default=64,
        type=int,
        help="The hidden dimensions of the model."
    )
    parser.add_argument(
        "--num_layers",
        default=1,
        type=int,
        help="The numbers of layers of the model."
    )
    parser.add_argument(
        "--dropout",
        default=0.0,
        type=float,
        help="Dropout rate."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Optimizer weight decay."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-4,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--clip_grad",
        default=1.0,
        type=float,
        help="Gradient clipping norm."
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU."
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        help="Whether to use the gpu or not."
    )
    parser.add_argument(
        "--seed",
        default=21,
        type=int,
        help="Random seed."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        default=None,
        help="The directory of the model."
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to run training or not."
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to run eval or not."
    )
    parser.add_argument(
        "--visualize_loss",
        action="store_true",
        help="Whether to visualize the loss during training and evaluation."
    )
    parser.add_argument(
        "--do_inference",
        action="store_true",
        help="Whether to do inference or not."
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="dev",
        help="The dataset to do inference on."
    )
    parser.add_argument(
        "--add_side_constraints",
        action="store_true",
        help="To use side constraints or not."
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for beam search."
    )
    parser.add_argument(
        "--n_best",
        type=int,
        default=5,
        help="Top n best decoded sequences to maintain in beam search."
    )
    parser.add_argument(
        "--analyzer_db_path",
        type=str,
        default=None,
        help="Path to the anaylzer database."
    )
    parser.add_argument(
        "--do_early_stopping",
        action="store_true",
        help="To do early stopping or not."
    )
    parser.add_argument(
        "--preds_dir",
        type=str,
        default=None,
        help="The directory to write the translations to."
    )

    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda else 'cpu')
    set_seed(args.seed, args.use_cuda)

    if args.reload_files:
        dataset = MT_Dataset.load_data_and_load_vectorizer(data_dir=args.data_dir,
                                                           vec_path=args.vectorizer_path)
    else:
        dataset = MT_Dataset.load_data_and_create_vectorizer(data_dir=args.data_dir,
                                                             add_side_constraints=args.add_side_constraints)
    # import pdb; pdb.set_trace()

    vectorizer = dataset.get_vectorizer()

    if args.cache_files:
        dataset.save_vectorizer(args.vectorizer_path)


    if args.do_early_stopping:
        patience = 0

    ENCODER_INPUT_DIM = len(vectorizer.src_vocab_char)
    DECODER_INPUT_DIM = len(vectorizer.trg_vocab_char)
    DECODER_OUTPUT_DIM = len(vectorizer.trg_vocab_char)
    CHAR_SRC_PAD_INDEX = vectorizer.src_vocab_char.pad_idx
    WORD_SRC_PAD_INDEX = vectorizer.src_vocab_word.pad_idx
    TRG_PAD_INDEX = vectorizer.trg_vocab_char.pad_idx
    TRG_SOS_INDEX = vectorizer.trg_vocab_char.sos_idx

    models_config = {'encoder_input_dim': ENCODER_INPUT_DIM,
                     'encoder_embed_dim': args.embed_dim,
                     'encoder_hidd_dim': args.hidd_dim,
                     'encoder_num_layers': args.num_layers,
                     'decoder_input_dim': DECODER_INPUT_DIM,
                     'decoder_embed_dim': args.embed_dim,
                     'decoder_hidd_dim': args.hidd_dim,
                     'decoder_num_layers': args.num_layers,
                     'decoder_output_dim': DECODER_OUTPUT_DIM,
                     'char_src_padding_idx': CHAR_SRC_PAD_INDEX,
                     'word_src_padding_idx': WORD_SRC_PAD_INDEX,
                     'trg_padding_idx': TRG_PAD_INDEX,
                     'trg_sos_idx': TRG_SOS_INDEX,
                     'dropout': args.dropout
                     }

    model = Seq2Seq(**models_config)


    # saving the model's args
    with open(args.model_path.replace('pt', 'config.json'), 'w') as f:
        json.dump(models_config, f)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=args.weight_decay)
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_INDEX)
    # lr scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     patience=2, factor=0.5)

    collator = Collator(char_src_pad_idx=CHAR_SRC_PAD_INDEX,
                        char_trg_pad_idx=TRG_PAD_INDEX,
                        word_src_pad_idx=WORD_SRC_PAD_INDEX)

    model = model.to(device)

    if args.do_train:
        logger.info('Training...')
        train_losses = []
        dev_losses = []
        best_loss = 1e10
        teacher_forcing_prob = 0.3
        clip_grad = args.clip_grad
        set_seed(args.seed, args.use_cuda)

        for epoch in range(args.num_train_epochs):
            dataset.set_split('train')
            dataloader = DataLoader(dataset, shuffle=True,
                                    batch_size=args.batch_size,
                                    collate_fn=collator, drop_last=False)

            train_loss = train(model, dataloader, optimizer, criterion,
                               device, teacher_forcing_prob=teacher_forcing_prob,
                               clip_grad=clip_grad)
            train_losses.append(train_loss)

            dataset.set_split('dev')
            dataloader = DataLoader(dataset, shuffle=True,
                                    batch_size=args.batch_size,
                                    collate_fn=collator, drop_last=False)
            dev_loss = evaluate(model, dataloader, criterion, device,
                                teacher_forcing_prob=0)
            dev_losses.append(dev_loss)

            #save best model
            if dev_loss < best_loss:
                best_loss = dev_loss
                torch.save(model.state_dict(), args.model_path)
                if args.do_early_stopping: patience = 0

            elif args.do_early_stopping:
                patience += 1
                if patience > 5:
                    logger.info(f"Dev loss hasn't decreased in {patience}"
                                " epochs. Stopping training..")
                    break

            scheduler.step(dev_loss)
            logger.info(f'Epoch: {(epoch + 1)}')
            logger.info(f"\tTrain Loss: {train_loss:.4f}   |   Dev Loss: {dev_loss:.4f}")

    if args.do_train and args.visualize_loss:
        plt.plot(range(1, 1 + args.num_train_epochs), np.asarray(train_losses),
                       'b-', color='blue',label='Training')
        plt.plot(range(1, 1 + args.num_train_epochs), np.asarray(dev_losses),
                      'b-', color='orange', label='Evaluation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(args.model_path + '.loss.png')

    if args.do_eval:
        logger.info('Evaluation')
        set_seed(args.seed, args.use_cuda)
        dev_losses = []
        for epoch in range(args.num_train_epochs):
            dataset.set_split('dev')
            dataloader = DataLoader(dataset, shuffle=True,
                                    batch_size=args.batch_size,
                                    collate_fn=collator)
            dev_loss = evaluate(model, dataloader, criterion, device,
                                teacher_forcing_prob=0)
            dev_losses.append(dev_loss)
            logger.info(f'Dev Loss: {dev_loss:.4f}')

    if args.do_inference:
        import pdb; pdb.set_trace()
        logger.info('Inference')
        set_seed(args.seed, args.use_cuda)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        model = model.to(device)
        dataset.set_split(args.inference_mode)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                collate_fn=collator)
        sampler = BatchSampler(model=model,
                               src_vocab_char=vectorizer.src_vocab_char,
                               src_vocab_word=vectorizer.src_vocab_word,
                               trg_vocab_char=vectorizer.trg_vocab_char,
                               ged_tag_vocab=vectorizer.ged_tag_vocab)

        beam_sampler =  BeamSampler(model=model,
                                    src_vocab_char=vectorizer.src_vocab_char,
                                    src_vocab_word=vectorizer.src_vocab_word,
                                    trg_vocab_char=vectorizer.trg_vocab_char,
                                    ged_tag_vocab=vectorizer.ged_tag_vocab,
                                    beam_width=args.beam_size,
                                    topk=args.n_best)

        inference(sampler, beam_sampler, dataloader, args)


if __name__ == "__main__":
    main()

