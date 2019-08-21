# encoding=utf-8

"""Attentional Encoder Decoder Model"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .utils import try_load_state


class Encoder(nn.Module):
    def __init__(self, hps, pad_id=1, batch_first=True):
        super().__init__()
        self._hps = hps
        self._batch_first = batch_first
        self.embedding = nn.Embedding(self._hps.vocab_size, self._hps.embed_dim, padding_idx=pad_id)
        # bidirectional 1-layer LSTM
        self.lstm = nn.LSTM(self._hps.embed_dim, self._hps.hidden_dim, num_layers=1, batch_first=self._batch_first,
                            bidirectional=True)
        # W_h in Equation 1
        self.W_h = nn.Linear(2 * self._hps.hidden_dim, 2 * self._hps.hidden_dim, bias=False)
        # Reduce the dim of the last hidden state
        self.reduce_h = nn.Linear(2 * self._hps.hidden_dim, self._hps.hidden_dim)
        self.reduce_c = nn.Linear(2 * self._hps.hidden_dim, self._hps.hidden_dim)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients[module.name] = {
            'grad_input': grad_input,
            'grad_output': grad_output
        }


    def forward(self, enc_inps, enc_seq_lens):
        """
        :param enc_inps: batch_size x max_seq_len
        :param seq_lens: batch_size
        :return:
            enc_outputs: batch_size x max_seq_len x 2*hidden_dim
            enc_features: batch_size x max_seq_len x 2*hidden_dim
            s_0: tuple of two batch_size x 2*hidden_dim
        """
        # batch_size x max_seq_len -> batch_size x max_seq_len x embed_dim
        enc_embeddings = self.embedding(enc_inps)
        # batch_size x max_seq_len x embed_dim -> packed sequences
        packed_inps = pack_padded_sequence(enc_embeddings, enc_seq_lens, batch_first=self._batch_first)
        # enc_h_t & enc_c_t: 2 x batch_size x hidden_dim
        packed_outputs, (enc_h_t, enc_c_t) = self.lstm(packed_inps)
        # packed sequences -> batch_size x max_seq_len x 2*hidden_dim
        enc_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=self._batch_first)
        # batch_size x max_seq_len x 2*hidden_dim
        enc_features = self.W_h(enc_outputs)
        # 2 x batch_size x hidden_dim -> batch_size x 2*hidden_dim
        enc_h_t = enc_h_t.transpose(0, 1).reshape(-1, 2 * self._hps.hidden_dim)
        enc_c_t = enc_c_t.transpose(0, 1).reshape(-1, 2 * self._hps.hidden_dim)
        # 1 x batch_size x 2*hidden_dim
        reduced_h_t = F.relu(self.reduce_h(enc_h_t)).unsqueeze(0)
        reduced_c_t = F.relu(self.reduce_c(enc_c_t)).unsqueeze(0)
        s_0 = (reduced_h_t, reduced_c_t)
        return enc_outputs, enc_features, s_0


class AttentionDecoder(nn.Module):
    """
    Procedure

    dec_embeddings = embedding(y_t_1)
    lstm_input = [c_t_1, dec_embedding]
    lstm_output, s_t = lstm(lstm_input, s_t_1)
    # enc_seq_len
    e_t = v^T tanh(enc_features + W_s*s_t + b_{attn})
    a_t = softmax(e_t)
    Mask pads
    # element-wise
    c_t = sum(a_t * enc_outputs, -1)
    vocab_dist = softmax(V'(V[lstm_output,c_t] + b) + b')
    """
    def __init__(self, hps, pad_id=1, batch_first=True):
        super().__init__()
        self._hps = hps
        self._batch_first = batch_first
        self.W_s = nn.Linear(2 * self._hps.hidden_dim, 2 * self._hps.hidden_dim)
        self.v = nn.Linear(2 * self._hps.hidden_dim, 1, bias=False)
        self.embedding = nn.Embedding(self._hps.vocab_size, self._hps.embed_dim, padding_idx=pad_id)
        # concatenate x with c_t_1
        self.x_context = nn.Linear(self._hps.embed_dim + 2 * self._hps.hidden_dim, self._hps.embed_dim)
        # uni-directional
        self.lstm = nn.LSTM(self._hps.embed_dim, self._hps.hidden_dim, num_layers=1, batch_first=self._batch_first)
        # NOTE: different from atulkum's implemenation, I concatenate s_t instead of lstm_output with h_t_star
        # which conforms to Equation 4 of the original paper
        self.V1 = nn.Linear(3 * self._hps.hidden_dim, self._hps.hidden_dim)
        # self.V1 = nn.Linear(4 * self._hps.hidden_dim, self._hps.hidden_dim)
        self.V2 = nn.Linear(self._hps.hidden_dim, self._hps.vocab_size)

        if self._hps.pointer_gen:
            # project c_t + s_t + x_t
            self.p_gen_linear = nn.Linear(2 * 2 * self._hps.hidden_dim + self._hps.embed_dim, 1)

        if self._hps.is_coverage:
            self.W_cover = nn.Linear(1, 2*self._hps.hidden_dim, bias=False)

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients[module.name] = {
            'grad_input': grad_input,
            'grad_output': grad_output
        }

    def tensor_hook(self, name):
        def hook(grad):
            self.gradients[name] = grad
        return hook

    def forward(self, y_t_1, s_t_1, c_t_1, enc_outputs, enc_features, enc_pad_mask, extend_vocab_zeros,
                enc_inps_extended, coverage_t):
        """
        :param y_t_1: batch_size x 1
        :param s_t_1: (1 x batch_size x hidden_dim, 1 x batch_size x hidden_dim)
        :param c_t_1: batch_size x 1 x 2*hidden_dim
        :param enc_outputs: batch_size x max_seq_len x 2*hidden_dim
        :param enc_features: batch_size x max_seq_len x 2*hidden_dim
        :param enc_pad_mask: batch_size x max_seq_len
        :param extend_vocab_zeros: batch_size x extend_vocab_size or None
        :param enc_inps_extended: batch_size x enc_max_seq_len
        :param coverage_t: batch_size x enc_max_seq, the coverage vector of the current step, which is the sum
            of the attention_dist from step 0 to step t-1
        :return:
            vocab_dist: batch_size x vocab_size
            s_t: (1 x batch_size x hidden_size, 1 x batch_size x hidden_size)
            c_t: batch_size x 1 x 2*hidden_dim
            attn_dist
            coverage_t
        """
        # STEP 1: calculate s_t
        enc_max_seq_len = enc_features.size()[1]
        # batch_size -> batch_size x 1 x embed_dim
        dec_embeddings = self.embedding(y_t_1.view(-1, 1))
        # batch_size x  1 x (embed_dim+2*hidden_state) -> batch_size x 1 x embed_dim
        lstm_input = self.x_context( torch.cat([dec_embeddings, c_t_1], dim=-1) )
        # lstm_output: batch_size x 1 x hidden_dim
        # s_t: (1 x batch_size x hidden_size, 1 x batch_size x hidden_size)
        lstm_output, s_t = self.lstm(lstm_input, s_t_1)

        # STEP2: calculate c_t, i.e., context vector
        # 1 x batch_size x 2*hidden_size
        s_t_cat = torch.cat(s_t, -1)
        # batch_size x 1 x 2*hidden_size
        s_t_cat_T = s_t_cat.transpose(0, 1)
        # 1 x batch_size x 2*hidden_state -> batch_size x enc_max_seq_len x 2*hidden_state
        s_t_hat = s_t_cat_T.expand(-1, enc_max_seq_len, -1).contiguous()


        if self._hps.is_coverage:
            # batch x enc_max_seq x 1
            coverage_t_hat = coverage_t.unsqueeze(2)
            # batch x enc_max_seq x different_dim -> batch_size x enc_max_seq_len
            e_t = self.v(torch.tanh( enc_features + self.W_s(s_t_hat) + self.W_cover(coverage_t_hat))).squeeze(-1)
        else:
            # batch x enc_max_seq x different_dim -> batch_size x enc_max_seq_len
            e_t = self.v(torch.tanh( enc_features + self.W_s(s_t_hat) )).squeeze(-1)

        # batch_size x enc_max_seq_len
        a_t_1 = F.softmax(e_t, dim=-1)
        # mask pads in enc_inps
        a_t = a_t_1 * enc_pad_mask
        # each item is the sum of that batch
        # batch_size x 1
        normalizer = a_t.sum(dim=-1, keepdim=True)
        # sum of a_i * hi can be calculated using bmm
        # batch_size x enc_max_seq_len
        attn_dist = a_t / (normalizer + self._hps.eps)

        if self._hps.is_coverage:
            # batch_size x enc_max_seq_len
            coverage_t = coverage_t + attn_dist

        # batch_size x 1 x enc_max_seq_len bmm batch_size x enc_max_seq_len x 2*hidden_dim
        # -> batch x 1 x 2*hidden_dim
        # c_t is the context vector
        c_t = torch.bmm(attn_dist.unsqueeze(1), enc_outputs)

        # STEP3: calculate the vocab_dist using lstm_output and c_t
        # NOTE: in abisee's implementation, they use lstm_output instead of s_t
        # this is different from the equations in the original paper
        # batch x 3*hidden_dim
        dec_output = torch.cat((lstm_output, c_t), dim=-1).squeeze(1)
        # dec_output = torch.cat( (s_t_cat.squeeze(0), h_t_star), -1 )
        # batch_size x vocab_size
        vocab_dist = F.softmax( self.V2( self.V1(dec_output) ), dim=-1)

        # Add pointer mechanism
        if self._hps.pointer_gen:
            p_gen_input = torch.cat((c_t, s_t_cat_T, dec_embeddings), dim=-1)
            # batch x 1 x 1 -> batch x 1
            p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input)).view(-1, 1)
            # batch x vocab_size
            vocab_dist_ = p_gen * vocab_dist
            # batch x extend_vocab_size
            if extend_vocab_zeros is not None:
                vocab_dist_ = torch.cat( (vocab_dist_, extend_vocab_zeros), dim=-1)
            # batch x enc_max_seq_len
            attn_dist_ = (1 - p_gen) * attn_dist
            # enc_inps_extended: batch x enc_max_seq_len
            final_dist = vocab_dist_.scatter_add(1, enc_inps_extended, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, c_t, attn_dist, coverage_t


class PointerEncoderDecoder(object):
    def __init__(self, hps, model_file_path, pad_id=1, is_eval=False):
        if is_eval:
            device = hps.eval_device
        else:
            device = hps.device
        print(device)
        encoder = Encoder(hps, pad_id)
        decoder = AttentionDecoder(hps, pad_id)

        decoder.embedding.weight = encoder.embedding.weight
        if is_eval:
            encoder = encoder.eval()
            decoder = decoder.eval()

        device = torch.device(device)
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        self.encoder = encoder
        self.decoder = decoder

        if model_file_path is not None:
            state = try_load_state(model_file_path)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            # since we need to leverage coverage
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)

    @property
    def parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())
