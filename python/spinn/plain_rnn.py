from functools import partial
import argparse
import itertools

import numpy as np
from spinn import util

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.blocks import Embed, to_gpu, MLP
from spinn.util.misc import Args, Vocab


def build_model(data_manager, initial_embeddings, vocab_size, num_classes, FLAGS):
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA
    model_cls = BaseModel

    return model_cls(model_dim=FLAGS.model_dim,
         word_embedding_dim=FLAGS.word_embedding_dim,
         vocab_size=vocab_size,
         initial_embeddings=initial_embeddings,
         num_classes=num_classes,
         embedding_keep_rate=FLAGS.embedding_keep_rate,
         use_sentence_pair=use_sentence_pair,
         use_difference_feature=FLAGS.use_difference_feature,
         use_product_feature=FLAGS.use_product_feature,
         classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
         mlp_dim=FLAGS.mlp_dim,
         num_mlp_layers=FLAGS.num_mlp_layers,
         mlp_bn=FLAGS.mlp_bn,
        )


class BaseModel(nn.Module):

    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 num_classes=None,
                 embedding_keep_rate=None,
                 use_sentence_pair=False,
                 classifier_keep_rate=None,
                 mlp_dim=None,
                 num_mlp_layers=None,
                 mlp_bn=None,
                 **kwargs
                ):
        super(BaseModel, self).__init__()

        self.use_sentence_pair = use_sentence_pair
        self.model_dim = model_dim

        classifier_dropout_rate = 1. - classifier_keep_rate

        args = Args()
        args.size = model_dim
        args.input_dropout_rate = 1. - embedding_keep_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        self.embed = Embed(args.size, vocab.size,
                        vectors=vocab.vectors,
                        )

        self.rnn = nn.LSTM(args.size, model_dim, num_layers=1, batch_first=True)

        mlp_input_dim = model_dim * 2 if use_sentence_pair else model_dim

        self.mlp = MLP(mlp_input_dim, mlp_dim, num_classes,
            num_mlp_layers, mlp_bn, classifier_dropout_rate)

    def run_rnn(self, x):
        batch_size, seq_len, model_dim = x.data.size()

        num_layers = 1
        bidirectional = False
        bi = 2 if bidirectional else 1
        h0 = Variable(to_gpu(torch.zeros(num_layers * bi, batch_size, self.model_dim)), volatile=not self.training)
        c0 = Variable(to_gpu(torch.zeros(num_layers * bi, batch_size, self.model_dim)), volatile=not self.training)

        # Expects (input, h_0):
        #   input => batch_size x seq_len x model_dim
        #   h_0   => (num_layers x num_directions[1,2]) x batch_size x model_dim
        #   c_0   => (num_layers x num_directions[1,2]) x batch_size x model_dim
        output, (hn, cn) = self.rnn(x, (h0, c0))

        return hn

    def run_embed(self, x):
        batch_size, seq_length = x.size()

        emb = self.embed(x)
        emb = torch.cat([b.unsqueeze(0) for b in torch.chunk(emb, batch_size, 0)], 0)

        return emb

    def forward(self, sentences, transitions, y_batch=None, **kwargs):
        # Useful when investigating dynamic batching.
        self.seq_lengths = sentences.shape[1] - (sentences == 0).sum(1)

        x = self.unwrap(sentences, transitions)
        emb = self.run_embed(x)
        hh = torch.squeeze(self.run_rnn(emb))
        h = self.wrap(hh)
        output = self.mlp(h)

        return output

    # --- Sentence Style Switches ---

    def unwrap(self, sentences, transitions):
        if self.use_sentence_pair:
            return self.unwrap_sentence_pair(sentences, transitions)
        return self.unwrap_sentence(sentences, transitions)

    def wrap(self, hh):
        if self.use_sentence_pair:
            return self.wrap_sentence_pair(hh)
        return self.wrap_sentence(hh)

    # --- Sentence Specific ---

    def unwrap_sentence_pair(self, sentences, transitions):
        batch_size = sentences.shape[0]

        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        return to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))

    def wrap_sentence_pair(self, hh):
        batch_size = hh.size(0) / 2
        h = torch.cat([hh[:batch_size], hh[batch_size:]], 1)
        return h

    # --- Sentence Pair Specific ---

    def unwrap_sentence(self, sentences, transitions):
        return to_gpu(Variable(torch.from_numpy(sentences), volatile=not self.training))

    def wrap_sentence(self, hh):
        return hh
