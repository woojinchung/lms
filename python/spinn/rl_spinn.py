import itertools
import copy

import numpy as np
from spinn import util

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from spinn.util.blocks import Reduce
from spinn.util.blocks import LSTMState, Embed, MLP
from spinn.util.blocks import bundle, unbundle, to_cpu, to_gpu, the_gpu, treelstm, lstm
from spinn.util.blocks import get_h, get_c
from spinn.util.misc import Args, Vocab, Example

from spinn.fat_stack import BaseModel as _BaseModel
from spinn.fat_stack import SPINN

from spinn.data import T_SHIFT, T_REDUCE, T_SKIP, T_STRUCT


def build_model(data_manager, initial_embeddings, vocab_size, num_classes, FLAGS):
    model_cls = BaseModel
    use_sentence_pair = data_manager.SENTENCE_PAIR_DATA

    return model_cls(model_dim=FLAGS.model_dim,
         word_embedding_dim=FLAGS.word_embedding_dim,
         vocab_size=vocab_size,
         initial_embeddings=initial_embeddings,
         num_classes=num_classes,
         mlp_dim=FLAGS.mlp_dim,
         embedding_keep_rate=FLAGS.embedding_keep_rate,
         classifier_keep_rate=FLAGS.semantic_classifier_keep_rate,
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
         num_mlp_layers=FLAGS.num_mlp_layers,
         mlp_bn=FLAGS.mlp_bn,
         rl_mu=FLAGS.rl_mu,
         rl_epsilon=FLAGS.rl_epsilon,
         rl_baseline=FLAGS.rl_baseline,
         rl_reward=FLAGS.rl_reward,
         rl_weight=FLAGS.rl_weight,
         rl_whiten=FLAGS.rl_whiten,
         rl_entropy=FLAGS.rl_entropy,
         rl_entropy_beta=FLAGS.rl_entropy_beta,
        )


class RLSPINN(SPINN):
    def predict_actions(self, transition_output):
        transition_dist = F.softmax(transition_output).data
        transition_greedy = transition_dist.cpu().numpy().argmax(axis=1)
        if self.training:
            transitions_sampled = torch.multinomial(transition_dist, 1).view(-1).cpu().numpy()
            r = np.random.binomial(1, self.epsilon, len(transitions_sampled))
            transition_preds = np.where(r, transitions_sampled, transition_greedy)
        else:
            transition_preds = transition_greedy
        return transition_preds


class BaseModel(_BaseModel):

    optimize_transition_loss = False

    def __init__(self,
                 rl_mu=None,
                 rl_baseline=None,
                 rl_reward=None,
                 rl_weight=None,
                 rl_whiten=None,
                 rl_epsilon=None,
                 rl_entropy=None,
                 rl_entropy_beta=None,
                 **kwargs):
        super(BaseModel, self).__init__(**kwargs)

        self.kwargs = kwargs

        self.rl_mu = rl_mu
        self.rl_baseline = rl_baseline
        self.rl_reward = rl_reward
        self.rl_weight = rl_weight
        self.rl_whiten = rl_whiten
        self.rl_entropy = rl_entropy
        self.rl_entropy_beta = rl_entropy_beta
        self.spinn.epsilon = rl_epsilon

        self.register_buffer('baseline', torch.FloatTensor([0.0]))

    def build_spinn(self, args, vocab, predict_use_cell, use_lengths):
        return RLSPINN(args, vocab, predict_use_cell, use_lengths)

    def run_greedy(self, sentences, transitions):
        inference_model_cls = BaseModel

        # HACK: This is a pretty simple way to create the inference time version of SPINN.
        # The reason a copy is necessary is because there is some retained state in the
        # memories and loss variables that break deep copy.
        inference_model = inference_model_cls(**self.kwargs)
        inference_model.load_state_dict(copy.deepcopy(self.state_dict()))
        inference_model.eval()

        if the_gpu.gpu >= 0:
            inference_model.cuda()
        else:
            inference_model.cpu()

        outputs = inference_model(sentences, transitions,
            use_internal_parser=True,
            validate_transitions=True)

        return outputs

    def build_reward(self, probs, target, rl_reward="standard"):
        if rl_reward == "standard": # Zero One Loss.
            rewards = torch.eq(probs.max(1)[1], target).float()
        elif rl_reward == "xent": # Cross Entropy Loss.
            _target = target.long().view(-1, 1)
            mask = torch.zeros(probs.size()).scatter_(1, _target, 1.0) # one hots
            log_inv_prob = torch.log(1 - probs) # get the log of the inverse probabilities
            rewards = -1 * (log_inv_prob * mask).sum(1)
        else:
            raise NotImplementedError

        if self.spinn.use_lengths:
            for i, (buf, stack) in enumerate(zip(self.spinn.bufs, self.spinn.stacks)):
                if len(buf) == 1 and len(stack) == 2:
                    rewards[i] += 1

        return rewards

    def build_baseline(self, output, rewards, sentences, transitions, y_batch=None, embeds=None):
        if self.rl_baseline == "ema":
            mu = self.rl_mu
            self.baseline[0] = self.baseline[0] * (1 - mu) + rewards.mean() * mu
            baseline = self.baseline[0]
        elif self.rl_baseline == "greedy":
            # Pass inputs to Greedy Max
            greedy_outp = self.run_greedy(sentences, transitions)

            # Estimate Reward
            probs = F.softmax(output).data.cpu()
            target = torch.from_numpy(y_batch).long()
            greedy_rewards = self.build_reward(probs, target, rl_reward="xent")

            if self.rl_reward == "standard":
                greedy_rewards = F.sigmoid(greedy_rewards)

            baseline = greedy_rewards
        else:
            raise NotImplementedError

        return baseline

    def reinforce(self, advantage):
        # TODO: Many of these ops are on the cpu. Might be worth shifting to GPU.

        t_preds = np.concatenate([m['t_preds'] for m in self.spinn.memories if m.get('t_preds', None) is not None])
        t_mask = np.concatenate([m['t_mask'] for m in self.spinn.memories if m.get('t_mask', None) is not None])
        t_logits = torch.cat([m['t_logits'] for m in self.spinn.memories if m.get('t_logits', None) is not None], 0)

        batch_size = advantage.size(0)
        seq_length = t_preds.shape[0] / batch_size

        a_index = np.arange(batch_size)
        a_index = a_index.reshape(1,-1).repeat(seq_length, axis=0).flatten()
        a_index = torch.from_numpy(a_index[t_mask]).long()

        t_index = to_gpu(Variable(torch.from_numpy(np.arange(t_mask.shape[0])[t_mask])).long())

        if self.use_sentence_pair:
            # Handles the case of SNLI where each reward is used for two sentences.
            advantage = torch.cat([advantage, advantage], 0)

        # Expand advantage.
        advantage = torch.index_select(advantage, 0, a_index)

        # Filter logits.
        t_logits = torch.index_select(t_logits, 0, t_index)

        actions = torch.from_numpy(t_preds[t_mask]).long().view(-1, 1)
        action_mask = torch.zeros(t_logits.size()).scatter_(1, actions, 1.0)
        action_mask = to_gpu(Variable(action_mask, volatile=not self.training))
        log_p_action = torch.sum(t_logits * action_mask, 1)

        # source: https://github.com/miyosuda/async_deep_reinforce/issues/1
        if self.rl_entropy:
            # TODO: Taking exp of a log is not the best way to get the initial probability...
            entropy = -1. * (t_logits * torch.exp(t_logits)).sum(1)
            self.avg_entropy = entropy.sum().data[0] / float(entropy.size(0))
        else:
            entropy = 0.0

        policy_loss = -1. * torch.sum(log_p_action * to_gpu(Variable(advantage, volatile=log_p_action.volatile)) + entropy * self.rl_entropy_beta)
        policy_loss /= log_p_action.size(0)
        policy_loss *= self.rl_weight

        return policy_loss

    def output_hook(self, output, sentences, transitions, y_batch=None, embeds=None):
        if not self.training:
            return

        probs = F.softmax(output).data.cpu()
        target = torch.from_numpy(y_batch).long()

        # Get Reward.
        rewards = self.build_reward(probs, target, rl_reward=self.rl_reward)

        # Get Baseline.
        baseline = self.build_baseline(output, rewards, sentences, transitions, y_batch, embeds)

        # Calculate advantage.
        advantage = rewards - baseline

        # Whiten advantage. This is also called Variance Normalization.
        if self.rl_whiten:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # Assign REINFORCE output.
        self.policy_loss = self.reinforce(advantage)
