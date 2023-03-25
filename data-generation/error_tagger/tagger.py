import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import clip_grad_norm_
from torch import optim
import random
import numpy as np
import argparse
import logging
from .utils import ErrorTagDataset, Vectorizer
import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, input_size, embed_size, hidd_size,
                 num_layers, output_size, bidirectional=False, dropout=0.0):
        super(Model, self).__init__()
        self.embed_layer = nn.Embedding(input_size, embed_size, padding_idx=0)
        self.rnn = nn.GRU(input_size=embed_size,
                          hidden_size=hidd_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)

        self.linear = nn.Linear(2 * hidd_size if bidirectional else hidd_size,
                                output_size)

    def forward(self, seqs, seqs_lengths):
        embedding = self.embed_layer(seqs)

        packed_embedded_seq = pack_padded_sequence(embedding, seqs_lengths,
                                                   batch_first=True)

        output, hidd = self.rnn(packed_embedded_seq)

        # concatenating the forward and backward vectors for each layer
        # hidd = torch.cat([hidd[0:hidd.size(0):2], hidd[1:hidd.size(0):2]], dim=2)

        logits = self.linear(hidd[-1])
        return logits.squeeze()


def train(model, dataloader, optimizer, criterion, device='cpu', clip_grad=1.0):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        seqs = batch['words']
        targets = batch['tags']
        lengths = batch['lengths'].to('cpu').int()

        preds  = model(seqs=seqs, seqs_lengths=lengths)

        preds = preds.contiguous().view(-1, preds.shape[-1])
        targets = targets.view(-1)

        loss = criterion(preds, targets)

        epoch_loss += loss.item()

        # Backprop
        loss.backward()
        # Gradient clipping
        # clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        # Optimizer step
        optimizer.step()

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device='cpu'):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            seqs = batch['words']
            targets = batch['tags']
            lengths = batch['lengths'].to('cpu').int()

            preds  = model(seqs=seqs, seqs_lengths=lengths)

            preds = preds.contiguous().view(-1, preds.shape[-1])
            targets = targets.view(-1)

            loss = criterion(preds, targets)

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def inference(model, loader, vectorizer, device):
    predictions = []
    targets = []
    top5_preds = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        seqs = batch['words']
        gold = batch['tags']
        lengths = batch['lengths'].to('cpu').int()
        logits = model(seqs, lengths)

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).numpy()
        _, top5 = torch.topk(probs, k=5, dim=-1)
        predictions.extend(preds)
        targets.extend(gold.numpy())
        top5_preds.extend(top5.numpy())

    # vectorized_word = vectorizer.get_word_indices(word, pos).to(device)
    # length = torch.tensor([vectorized_word.shape[-1]]).to('cpu').int()

    predicted_tags = [vectorizer.error_tags_vocab.lookup_index(x)
                      for x in predictions]

    overall_accuracy = (np.asarray(predictions) == np.asarray(targets)).mean()

    top5_tags = [[vectorizer.error_tags_vocab.lookup_index(x) for x in ex] 
                for ex in top5_preds]
    top5 = []
    for i in range(len(targets)):
        if targets[i] in top5_preds[i]:
            top5.append(targets[i])
        else:
            top5.append(targets[0])

    top5_accuracy = (np.asarray(top5) == np.asarray(targets)).mean()
    print(f'Top 1 Accuacy: {overall_accuracy}')
    print(f'Top 5 Accuracy: {top5_accuracy}')
    return predicted_tags, top5_tags


def inference_single(model, vectorizer, word, morph_feats):
    model.eval()
    with torch.no_grad():
        vectorized_word = vectorizer.get_word_indices(word, morph_feats)
        length = torch.tensor([vectorized_word.shape[-1]]).to('cpu').int()
        logits = model(vectorized_word, length)
        probs = torch.softmax(logits, dim=-1)
        top_pred = torch.argmax(probs, dim=-1)
        top5_probs, top5 = torch.topk(probs, k=5, dim=-1)
        return {'top': vectorizer.error_tags_vocab.lookup_index(top_pred.item()),
                'top5': [(vectorizer.error_tags_vocab.lookup_index(pred.item()), top5_probs[i].item())
                         for i, pred in enumerate(top5.numpy())]
                }


def collate(batch):
    sorted_batch = sorted(batch, key=lambda x: x['word'].shape[0], reverse=True)

    words = [x['word'] for x in sorted_batch]
    tags = torch.tensor([x['tag'] for x in sorted_batch])
    lengths = torch.tensor([len(x['word']) for x in sorted_batch])

    padded_words = pad_sequence(words, batch_first=True, padding_value=0)

    return {'words': padded_words,
            'tags': tags,
            'lengths': lengths
           }

def set_seed(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        default=None,
        type=str,
        help="The train data path."
    )
    parser.add_argument(
        "--dev_path",
        default=None,
        type=str,
        help="The dev data path."
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

    train_dataset = ErrorTagDataset.load_data_and_create_vectorizer(args.train_path)
    logger.info(f'Training Examples: {len(train_dataset)}')
    
    dev_dataset = ErrorTagDataset.load_data(args.dev_path)

    vectorizer = train_dataset.vectorizer
    train_dataset.save_vectorizer('vectorizer.txt')

    dev_dataset.vectorizer = vectorizer


    input_size = len(vectorizer.char_vocab)
    output_size = len(vectorizer.error_tags_vocab)

    models_config = {'input_size': input_size,
                     'embed_size': args.embed_dim,
                     'hidd_size': args.hidd_dim,
                     'num_layers': args.num_layers,
                     'output_size': output_size,
                     'dropout': args.dropout
                    }

    model = Model(**models_config)
    model = model.to(device)


    # saving the model's args
    with open(args.model_path.replace('pt', 'config.json'), 'w') as f:
        json.dump(models_config, f)

    if args.do_train:
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                               weight_decay=args.weight_decay)
        # Loss function
        criterion = nn.CrossEntropyLoss()

        # lr scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                        patience=2, factor=0.5)

        if args.do_early_stopping:
            patience = 0

        logger.info('Training...')
        train_losses = []
        dev_losses = []
        best_loss = 1e10
        clip_grad = args.clip_grad
        set_seed(args.seed, args.use_cuda)

        for epoch in range(args.num_train_epochs):
            train_loader = DataLoader(train_dataset, shuffle=True,
                                    batch_size=args.batch_size,
                                    collate_fn=collate)
            train_loss = train(model, train_loader, optimizer, criterion,
                               device, clip_grad=clip_grad)

            dev_loader = DataLoader(dev_dataset, shuffle=False,
                                    batch_size=args.batch_size,
                                    collate_fn=collate)
            dev_loss = evaluate(model, dev_loader, criterion, device)

            train_losses.append(train_loss)
            dev_losses.append(dev_loss)

            if dev_loss < best_loss:
                best_loss = dev_loss
                torch.save(model.state_dict(), args.model_path)
                if args.do_early_stopping:
                    patience = 0

            elif args.do_early_stopping:
                patience += 1

                if patience > 10:
                    logger.info(f"Dev loss hasn't decreased in {patience}"
                                " epochs. Stopping training..")
                    break

            scheduler.step(dev_loss)
            logger.info(f'Epoch: {(epoch + 1)}')
            logger.info(f"\tTrain Loss: {train_loss:.4f}   |   Dev Loss: {dev_loss:.4f}")

    elif args.do_inference:
        set_seed(args.seed, args.use_cuda)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        model = model.to(device)
        dev_loader = DataLoader(dev_dataset, shuffle=False,
                                batch_size=args.batch_size,
                                collate_fn=collate)
        predicted_tags, top3_preds = inference(model, dev_loader, vectorizer, device)

        # with open('dev_preds.txt', mode='w') as f:
        #     for tag, top3 in zip(predicted_tags, top3_preds):
        #         f.write(f'{tag}\t{str(top3)}')
        #         f.write('\n')

if __name__ == '__main__':
    main()
