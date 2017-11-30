import itertools

import numpy as np
from spinn import util

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.blocks import Reduce, ReduceTensor
from spinn.util.blocks import LSTMState, Embed, MLP, Linear, LSTM, Lift
from spinn.util.blocks import reverse_tensor
from spinn.util.blocks import bundle, unbundle, to_cpu, to_gpu, treelstm, lstm
from spinn.util.blocks import get_h, get_c
from spinn.util.misc import Args, Vocab, Example
from spinn.util.blocks import HeKaimingInitializer

from spinn.data import T_SHIFT, T_REDUCE, T_SKIP, T_STRUCT


def build_model(data_manager, initial_embeddings, vocab_size, num_classes, FLAGS):
    model_cls = BaseModel
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA

    return model_cls(model_dim=FLAGS.model_dim,
         word_embedding_dim=FLAGS.word_embedding_dim,
         vocab_size=vocab_size,
         initial_embeddings=initial_embeddings,
         num_classes=num_classes,
         embedding_keep_rate=FLAGS.embedding_keep_rate,
         tracking_lstm_hidden_dim=FLAGS.tracking_lstm_hidden_dim,
         transition_weight=FLAGS.transition_weight,
         encode_style=FLAGS.encode_style,
         encode_reverse=FLAGS.encode_reverse,
         encode_bidirectional=FLAGS.encode_bidirectional,
         encode_num_layers=FLAGS.encode_num_layers,
         use_sentence_pair=use_sentence_pair,
         lateral_tracking=FLAGS.lateral_tracking,
         use_tracking_in_composition=FLAGS.use_tracking_in_composition,
         predict_use_cell=FLAGS.predict_use_cell,
         use_lengths=FLAGS.use_lengths,
         use_difference_feature=FLAGS.use_difference_feature,
         use_product_feature=FLAGS.use_product_feature,
         classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
         mlp_dim=FLAGS.mlp_dim,
         num_mlp_layers=FLAGS.num_mlp_layers,
         mlp_bn=FLAGS.mlp_bn,
         rl_mu=FLAGS.rl_mu,
         rl_baseline=FLAGS.rl_baseline,
         rl_reward=FLAGS.rl_reward,
         rl_weight=FLAGS.rl_weight,
         rl_whiten=FLAGS.rl_whiten,
         rl_entropy=FLAGS.rl_entropy,
         rl_entropy_beta=FLAGS.rl_entropy_beta,
         predict_leaf=FLAGS.predict_leaf,
         gen_h=FLAGS.gen_h,
        )


class Tracker(nn.Module):

    def __init__(self, size, tracker_size, lateral_tracking=True):
        super(Tracker, self).__init__()

        # Initialize layers.
        self.buf = Linear()(size, 4 * tracker_size, bias=False)
        self.stack1 = Linear()(size, 4 * tracker_size, bias=False)
        self.stack2 = Linear()(size, 4 * tracker_size, bias=False)

        if lateral_tracking:
            self.lateral = Linear(initializer=HeKaimingInitializer)(tracker_size, 4 * tracker_size)
        else:
            self.transform = Linear(initializer=HeKaimingInitializer)(4 * tracker_size, tracker_size)

        self.lateral_tracking = lateral_tracking
        self.state_size = tracker_size

        self.reset_state()

    def reset_state(self):
        self.c = self.h = None

    def forward(self, top_buf, top_stack_1, top_stack_2):
        tracker_inp = self.buf(top_buf.h)
        tracker_inp += self.stack1(top_stack_1.h)
        tracker_inp += self.stack2(top_stack_2.h)

        batch_size = tracker_inp.size(0)

        if self.lateral_tracking:
            if self.h is not None:
                tracker_inp += self.lateral(self.h)
            if self.c is None:
                self.c = to_gpu(Variable(torch.from_numpy(
                    np.zeros((batch_size, self.state_size),
                                  dtype=np.float32)),
                    volatile=tracker_inp.volatile))

            # Run tracking lstm.
            self.c, self.h = lstm(self.c, tracker_inp)

            return self.h, self.c
        else:
            outp = self.transform(tracker_inp)
            return outp, None

    @property
    def states(self):
        return unbundle((self.c, self.h))

    @states.setter
    def states(self, state_iter):
        if state_iter is not None:
            state = bundle(state_iter)
            self.c, self.h = state.c, state.h


class CSPINN(nn.Module):

    def __init__(self, args, vocab, predict_use_cell, use_lengths):
        super(CSPINN, self).__init__()

        # Optional debug mode.
        self.debug = False

        self.transition_weight = args.transition_weight
        self.use_lengths = use_lengths

        # Reduce function for semantic composition.
        self.reduce = ReduceTensor(args.size)
        self.choices = np.array([T_SHIFT, T_REDUCE], dtype=np.int32)

    def reset_state(self):
        self.memories = []

    def forward(self, example, use_internal_parser=False, validate_transitions=True):
        self.buffers_n = (example.tokens.data != 0).long().sum(1).view(-1).tolist()

        if self.debug:
            seq_length = example.tokens.size(1)
            assert all(buf_n <= (seq_length + 1) // 2 for buf_n in self.buffers_n), \
                "All sentences (including cropped) must be the appropriate length."

        self.bufs = example.bufs

        # Notes on adding zeros to bufs/stacks.
        # - After the buffer is consumed, we need one zero on the buffer
        #   used as input to the tracker.
        # - For the first two steps, the stack would be empty, but we add
        #   zeros so that the tracker still gets input.
        zeros = self.zeros = to_gpu(Variable(torch.from_numpy(
            np.zeros(self.bufs[0][0].size(), dtype=np.float32)),
            volatile=self.bufs[0][0].volatile))

        # Trim unused tokens.
        self.bufs = [[zeros] + b[-b_n:] for b, b_n in zip(self.bufs, self.buffers_n)]

        self.stacks = [[zeros, zeros] for buf in self.bufs]

        if hasattr(self, 'tracker'):
            self.tracker.reset_state()
        if not hasattr(example, 'transitions'):
            # TODO: Support no transitions. In the meantime, must at least pass dummy transitions.
            raise ValueError('Transitions must be included.')
        return self.run(example.transitions,
                        run_internal_parser=True,
                        use_internal_parser=use_internal_parser,
                        validate_transitions=validate_transitions)

    def validate(self, transitions, preds, stacks, bufs, zero_padded=True):
        # Note: There is one zero added to bufs, and two zeros added to stacks.
        # Make sure to adjust for this if using lengths of either.
        buf_adjust = 1 if zero_padded else 0
        stack_adjust = 2 if zero_padded else 0

        _transitions = np.array(transitions)
        _preds = preds.copy()
        _invalid = np.zeros(preds.shape, dtype=np.bool)

        incorrect = 0
        cant_skip = _transitions != T_SKIP
        must_skip = _transitions == T_SKIP

        # Fixup predicted skips.
        if len(self.choices) > 2:
            raise NotImplementedError("Can only validate actions for 2 choices right now.")

        buf_lens = [len(buf) - buf_adjust for buf in bufs]
        stack_lens = [len(stack) - stack_adjust for stack in stacks]

        # Cannot reduce on too small a stack
        must_shift = np.array([length < 2 for length in stack_lens])
        check_mask = np.logical_and(cant_skip, must_shift)
        _invalid += np.logical_and(_preds != T_SHIFT, check_mask)
        _preds[must_shift] = T_SHIFT

        # Cannot shift on too small buf
        must_reduce = np.array([length < 1 for length in buf_lens])
        check_mask = np.logical_and(cant_skip, must_reduce)
        _invalid += np.logical_and(_preds != T_REDUCE, check_mask)
        _preds[must_reduce] = T_REDUCE

        # If the given action is skip, then must skip.
        _preds[must_skip] = T_SKIP

        return _preds, _invalid

    def predict_actions(self, transition_output):
        transition_dist = F.log_softmax(transition_output)
        transition_dist = transition_dist.data.cpu().numpy()
        transition_preds = transition_dist.argmax(axis=1)
        return transition_preds

    def get_transitions_per_example(self, style="preds"):
        if style == "preds":
            source = "t_preds"
        elif style == "given":
            source = "t_given"
        else:
            raise NotImplementedError

        _transitions = [m[source].reshape(1, -1) for m in self.memories if m.get(source, None) is not None]
        transitions = np.concatenate(_transitions).T

        return transitions

    def t_shift(self, buf, stack, tracking, buf_tops, trackings):
        """SHIFT: Should dequeue buffer and item to stack."""
        buf_tops.append(buf.pop() if len(buf) > 0 else self.zeros)
        trackings.append(tracking)

    def t_reduce(self, buf, stack, tracking, lefts, rights, trackings):
        """REDUCE: Should compose top two items of the stack into new item."""

        # The right-most input will be popped first.
        for reduce_inp in [rights, lefts]:
            if len(stack) > 0:
                reduce_inp.append(stack.pop())
            else:
                if self.debug:
                    raise IndexError
                # If we try to Reduce, but there are less than 2 items on the stack,
                # then treat any available item as the right input, and use zeros
                # for any other inputs.
                # NOTE: Only happens on cropped data.
                reduce_inp.append(self.zeros)

        trackings.append(tracking)

    def t_skip(self):
        """SKIP: Acts as padding and is a noop."""
        pass

    def shift_phase(self, tops, trackings, stacks, idxs):
        """SHIFT: Should dequeue buffer and item to stack."""
        if len(stacks) > 0:
            shift_candidates = iter(tops)
            for stack in stacks:
                new_stack_item = next(shift_candidates)
                stack.append(new_stack_item)

    def reduce_phase(self, lefts, rights, trackings, stacks):
        if len(stacks) > 0:
            reduced = iter(self.reduce(
                lefts, rights))
            for stack in stacks:
                new_stack_item = next(reduced)
                stack.append(new_stack_item)

    def reduce_phase_hook(self, lefts, rights, trackings, reduce_stacks):
        pass

    def loss_phase_hook(self):
        pass

    def run(self, inp_transitions, run_internal_parser=False, use_internal_parser=False, validate_transitions=True):
        transition_loss = None
        transition_acc = 0.0
        num_transitions = inp_transitions.shape[1]
        batch_size = inp_transitions.shape[0]
        invalid_count = np.zeros(batch_size)

        # Transition Loop
        # ===============

        for t_step in range(num_transitions):
            transitions = inp_transitions[:, t_step]
            transition_arr = list(transitions)
            sub_batch_size = len(transition_arr)

            # A mask based on SKIP transitions.
            cant_skip = np.array(transitions) != T_SKIP
            must_skip = np.array(transitions) == T_SKIP

            # Memories
            # ========
            # Keep track of key values to determine accuracy and loss.
            self.memory = {}

            # Run if:
            # A. We have a tracking component and,
            # B. There is at least one transition that will not be skipped.
            if hasattr(self, 'tracker') and sum(cant_skip) > 0:

                # Prepare tracker input.
                if self.debug and any(len(buf) < 1 or len(stack) for buf, stack in zip(self.bufs, self.stacks)):
                    # To elaborate on this exception, when cropping examples it is possible
                    # that your first 1 or 2 actions is a reduce action. It is unclear if this
                    # is a bug in cropping or a bug in how we think about cropping. In the meantime,
                    # turn on the truncate batch flag, and set the eval_seq_length very high.
                    raise IndexError("Warning: You are probably trying to encode examples"
                          "with cropped transitions. Although, this is a reasonable"
                          "feature, when predicting/validating transitions, you"
                          "probably will not get the behavior that you expect. Disable"
                          "this exception if you dare.")
                top_buf = bundle(buf[-1] if len(buf) > 0 else self.zeros for buf in self.bufs)
                top_stack_1 = bundle(stack[-1] if len(stack) > 0 else self.zeros for stack in self.stacks)
                top_stack_2 = bundle(stack[-2] if len(stack) > 1 else self.zeros for stack in self.stacks)

                # Get hidden output from the tracker. Used to predict transitions.
                tracker_h, tracker_c = self.tracker(top_buf, top_stack_1, top_stack_2)

                if hasattr(self, 'transition_net'):
                    transition_inp = [tracker_h]
                    if self.use_lengths:
                        buf_lens = to_gpu(Variable(torch.FloatTensor([len(buf) for buf in self.bufs]), volatile=not self.training).view(-1, 1))
                        stack_lens = to_gpu(Variable(torch.FloatTensor([len(stack) for stack in self.stacks]), volatile=not self.training).view(-1, 1))
                        transition_inp += [buf_lens, stack_lens]
                    if self.predict_use_cell:
                        transition_inp += [tracker_c]
                    transition_inp = torch.cat(transition_inp, 1)
                    transition_output = self.transition_net(transition_inp)

                if hasattr(self, 'transition_net') and run_internal_parser:

                    # Predict Actions
                    # ===============

                    # Distribution of transitions use to calculate transition loss.
                    self.memory["t_logits"] = F.log_softmax(transition_output)

                    # Given transitions.
                    self.memory["t_given"] = transitions

                    # TODO: Mask before predicting. This should simplify things and reduce computation.
                    # The downside is that in the Action Phase, need to be smarter about which stacks/bufs
                    # are selected.
                    transition_preds = self.predict_actions(transition_output)

                    # Constrain to valid actions
                    # ==========================

                    validated_preds, invalid_mask = self.validate(transition_arr, transition_preds, self.stacks, self.bufs)
                    if validate_transitions:
                        transition_preds = validated_preds

                    # Keep track of which predictions have been valid.
                    invalid_count += invalid_mask

                    # If the given action is skip, then must skip.
                    transition_preds[must_skip] = T_SKIP

                    # Actual transition predictions. Used to measure transition accuracy.
                    self.memory["t_preds"] = transition_preds

                    # Binary mask of examples that have a transition.
                    self.memory["t_mask"] = cant_skip

                    # If this FLAG is set, then use the predicted actions rather than the given.
                    if use_internal_parser:
                        transition_arr = transition_preds.tolist()

            # Pre-Action Phase
            # ================

            # For SHIFT
            s_stacks, s_tops, s_trackings, s_idxs = [], [], [], []

            # For REDUCE
            r_stacks, r_lefts, r_rights, r_trackings = [], [], [], []

            batch = zip(transition_arr, self.bufs, self.stacks,
                        self.tracker.states if hasattr(self, 'tracker') and self.tracker.h is not None
                        else itertools.repeat(None))

            for batch_idx, (transition, buf, stack, tracking) in enumerate(batch):
                if transition == T_SHIFT: # shift
                    self.t_shift(buf, stack, tracking, s_tops, s_trackings)
                    s_idxs.append(batch_idx)
                    s_stacks.append(stack)
                elif transition == T_REDUCE: # reduce
                    self.t_reduce(buf, stack, tracking, r_lefts, r_rights, r_trackings)
                    r_stacks.append(stack)
                elif transition == T_SKIP: # skip
                    self.t_skip()

            # Action Phase
            # ============

            self.shift_phase(s_tops, s_trackings, s_stacks, s_idxs)
            self.reduce_phase(r_lefts, r_rights, r_trackings, r_stacks)
            self.reduce_phase_hook(r_lefts, r_rights, r_trackings, r_stacks)

            # Memory Phase
            # ============

            # APPEND ALL MEMORIES. MASK LATER.

            self.memories.append(self.memory)

        # Loss Phase
        # ==========
        self.loss_phase_hook()

        if self.debug:
            assert all(len(stack) == 3 for stack in self.stacks), \
                "Stacks should be fully reduced and have 3 elements: " \
                "two zeros and the sentence encoding."
            assert all(len(buf) == 1 for buf in self.bufs), \
                "Stacks should be fully shifted and have 1 zero."

        return [stack[-1] for stack in self.stacks], transition_acc, transition_loss


class BaseModel(nn.Module):

    optimize_transition_loss = True

    def __init__(self, model_dim=None,
                 word_embedding_dim=None,
                 vocab_size=None,
                 initial_embeddings=None,
                 num_classes=None,
                 embedding_keep_rate=None,
                 tracking_lstm_hidden_dim=4,
                 transition_weight=None,
                 encode_style=None,
                 encode_reverse=None,
                 encode_bidirectional=None,
                 encode_num_layers=None,
                 lateral_tracking=None,
                 use_tracking_in_composition=None,
                 predict_use_cell=None,
                 use_lengths=None,
                 use_sentence_pair=False,
                 use_difference_feature=False,
                 use_product_feature=False,
                 mlp_dim=None,
                 num_mlp_layers=None,
                 mlp_bn=None,
                 classifier_keep_rate=None,
                 use_projection=None,
                 **kwargs
                ):
        super(BaseModel, self).__init__()

        self.use_sentence_pair = use_sentence_pair
        self.use_difference_feature = use_difference_feature
        self.use_product_feature = use_product_feature
        self.hidden_dim = hidden_dim = model_dim

        args = Args()
        args.lateral_tracking = lateral_tracking
        args.use_tracking_in_composition = use_tracking_in_composition
        args.size = model_dim
        args.tracker_size = tracking_lstm_hidden_dim
        args.transition_weight = transition_weight

        self.initial_embeddings = initial_embeddings
        self.word_embedding_dim = word_embedding_dim
        self.model_dim = model_dim
        classifier_dropout_rate = 1. - classifier_keep_rate

        vocab = Vocab()
        vocab.size = initial_embeddings.shape[0] if initial_embeddings is not None else vocab_size
        vocab.vectors = initial_embeddings

        # Build classiifer.
        features_dim = self.get_features_dim()
        self.mlp = MLP(features_dim, mlp_dim, num_classes,
            num_mlp_layers, mlp_bn, classifier_dropout_rate)

        self.embedding_dropout_rate = 1. - embedding_keep_rate

        # Projection will effectively be done by the encoding network.
        use_projection = True if encode_style is None else False
        input_dim = args.size

        # Create dynamic embedding layer.
        self.embed = Embed(input_dim, vocab.size, vectors=vocab.vectors, use_projection=False)

        # Optionally build input encoder.
        if encode_style is not None:
            self.encode = self.build_input_encoder(encode_style=encode_style,
                word_embedding_dim=word_embedding_dim, model_dim=word_embedding_dim,
                num_layers=encode_num_layers, bidirectional=encode_bidirectional, reverse=encode_reverse,
                dropout=self.embedding_dropout_rate)
        
        # Create Lift layer
        self.lift = Lift(vocab.vectors.shape[1], input_dim * input_dim)

        # Build parsing component.
        self.spinn = self.build_spinn(args, vocab, predict_use_cell, use_lengths)

    def get_features_dim(self):
        features_dim = (self.hidden_dim * self.hidden_dim * 2) if self.use_sentence_pair else (self.hidden_dim * self.hidden_dim)
        if self.use_sentence_pair:
            if self.use_difference_feature:
                features_dim += (self.hidden_dim * self.hidden_dim)
            if self.use_product_feature:
                features_dim += (self.hidden_dim * self.hidden_dim)
        return features_dim

    def build_features(self, h):
        if self.use_sentence_pair:
            h_prem, h_hyp = h
            features = [h_prem, h_hyp]
            if self.use_difference_feature:
                features.append(h_prem - h_hyp)
            if self.use_product_feature:
                features.append(h_prem * h_hyp)
            features = torch.cat(features, 1)
        else:
            features = h[0]
        return features

    def build_input_encoder(self, encode_style="LSTM", word_embedding_dim=None, model_dim=None,
                            num_layers=None, bidirectional=None, reverse=None, dropout=None):
        if encode_style == "LSTM":
            encoding_net = LSTM(word_embedding_dim, model_dim,
                num_layers=num_layers, bidirectional=bidirectional, reverse=reverse,
                dropout=dropout)
        else:
            raise NotImplementedError
        return encoding_net

    def build_spinn(self, args, vocab, predict_use_cell, use_lengths):
        return CSPINN(args, vocab, predict_use_cell, use_lengths)

    def run_spinn(self, example, use_internal_parser, validate_transitions=True):
        self.spinn.reset_state()
        h_list, transition_acc, transition_loss = self.spinn(example,
                               use_internal_parser=use_internal_parser,
                               validate_transitions=validate_transitions)
        h = self.wrap(h_list)
        return h, transition_acc, transition_loss

    def output_hook(self, output, sentences, transitions, y_batch=None, embeds=None):
        pass

    def forward(self, sentences, transitions, y_batch=None,
                 use_internal_parser=False, validate_transitions=True):
        example = self.unwrap(sentences, transitions)

        b, l = example.tokens.size()[:2]

        embeds = self.embed(example.tokens)
        embeds = F.dropout(embeds, self.embedding_dropout_rate, training=self.training)
        embeds = torch.chunk(to_cpu(embeds), b, 0)

        if hasattr(self, 'encode'):
            to_encode = torch.cat([e.unsqueeze(0) for e in embeds], 0)
            encoded = self.encode(to_encode)
            embeds = [x.squeeze() for x in torch.chunk(encoded, b, 0)]

        embeds = torch.cat(embeds)
        embeds = self.lift(embeds)
        embeds = torch.chunk(to_cpu(embeds), b, 0)

        # Make Buffers
        embeds = [torch.chunk(x, l, 0) for x in embeds]
        buffers = [list(reversed(x)) for x in embeds]

        example.bufs = buffers

        h, transition_acc, transition_loss = self.run_spinn(example, use_internal_parser, validate_transitions)

        self.spinn_outp = h

        self.transition_acc = transition_acc
        self.transition_loss = transition_loss

        # Build features
        features = self.build_features(h)

        output = self.mlp(features)

        self.output_hook(output, sentences, transitions, y_batch, embeds)

        return output

    # --- Sentence Style Switches ---

    def unwrap(self, sentences, transitions):
        if self.use_sentence_pair:
            return self.unwrap_sentence_pair(sentences, transitions)
        return self.unwrap_sentence(sentences, transitions)

    def wrap(self, h_list):
        if self.use_sentence_pair:
            return self.wrap_sentence_pair(h_list)
        return self.wrap_sentence(h_list)

    # --- Sentence Model Specific ---

    def unwrap_sentence(self, sentences, transitions):
        batch_size = sentences.shape[0]

        # Build Tokens
        x = sentences

        # Build Transitions
        t = transitions

        example = Example()
        example.tokens = to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))
        example.transitions = t

        return example

    def wrap_sentence(self, h_list):
        hidden_dim = self.hidden_dim * self.hidden_dim
        h = get_h(torch.cat(h_list, 0), hidden_dim)
        return [h]

    # --- Sentence Pair Model Specific ---

    def unwrap_sentence_pair(self, sentences, transitions):
        batch_size = sentences.shape[0]

        # Build Tokens
        x_prem = sentences[:,:,0]
        x_hyp = sentences[:,:,1]
        x = np.concatenate([x_prem, x_hyp], axis=0)

        # Build Transitions
        t_prem = transitions[:,:,0]
        t_hyp = transitions[:,:,1]
        t = np.concatenate([t_prem, t_hyp], axis=0)

        example = Example()
        example.tokens = to_gpu(Variable(torch.from_numpy(x), volatile=not self.training))
        example.transitions = t

        return example

    def wrap_sentence_pair(self, h_list):
        hidden_dim = self.hidden_dim * self.hidden_dim
        batch_size = len(h_list) / 2

        premise = torch.cat(h_list[:batch_size], 0)
        hypothesis = torch.cat(h_list[batch_size:], 0)

        # Retrieve layer 2's hidden state
        h_premise = premise[:, (hidden_dim*3):]
        h_hypothesis = hypothesis[:, (hidden_dim*3):]

        return [h_premise, h_hypothesis]
