# -*- coding: utf-8 -*-

import re
import os
import csv
import sys
import time
import random
import torch
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import torch.multiprocessing as mp

import fire

from . import utils
from .decode import BeamSearch
from .pointer_model import PointerEncoderDecoder
from prsum.dataset import data, batcher
from prsum.dataset.train_util import get_input_from_batch, get_output_from_batch
from .utils import dump_json_file
from myrouge.rouge import Rouge

# deterministic
random.seed(318)
torch.manual_seed(318)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

csv.field_size_limit(sys.maxsize)


def reward_function(decoded_seqs, ref_seqs, device):
    decoded_seqs = [utils.prepare_rouge_text(seq) for seq in decoded_seqs]
    ref_seqs = [utils.prepare_rouge_text(seq) for seq in ref_seqs]
    rouge = Rouge()
    try:
        scores = rouge.get_scores(decoded_seqs, ref_seqs)
    except Exception:
        print("Rouge failed for multi sentence evaluation.. Finding exact pair")
        scores = []
        for i in range(len(decoded_seqs)):
            try:
                score = rouge.get_scores(decoded_seqs[i], ref_seqs[i])
            except Exception:
                print("Error occured at:")
                print("decoded_sents:", decoded_seqs[i])
                print("original_sents:", ref_seqs[i])
                score = [{"rouge-l": {"f": 0.0}}]
            scores.append(score[0])
        sys.stdout.flush()
    rouge_l_f1 = [score["rouge-l"]["f"] for score in scores]
    rouge_l_f1 = torch.tensor(rouge_l_f1, dtype=torch.float, device=device)
    return rouge_l_f1


class Procedure(object):
    """Base class of all process-related classes in order to share similar process"""

    def __init__(self, params, is_eval=False):
        self.vocab = data.Vocab(params.vocab_path, params.vocab_size)
        train_data_path = os.path.join(params.data_dir, "train." + params.data_file_suffix)
        eval_data_path = os.path.join(params.data_dir, "valid." + params.data_file_suffix)
        if not is_eval:
            self.batcher = batcher.Batcher(params, train_data_path, self.vocab, mode='train',
                                           batch_size=params.batch_size, single_pass=False)
        else:
            self.batcher = batcher.Batcher(params, eval_data_path, self.vocab, mode='eval',
                                           batch_size=params.batch_size, single_pass=True)
        self.pad_id = self.vocab.word2id(data.PAD_TOKEN)
        self.end_id = self.vocab.word2id(data.STOP_DECODING)
        self.unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        assert(self.pad_id == 1)
        self.dump_dir = None
        self.params = params
        self.is_eval = is_eval

    def infer_one_batch(self, batch, iter=None, is_eval=False):
        if is_eval:
            device = self.params.eval_device
        else:
            device = self.params.device
        device = torch.device(device)
        train_ml = getattr(self.params, "train_ml", True)
        train_rl = getattr(self.params, "train_rl", False)

        # c_t_1: batch_size x 2*hidden_dim
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extended, extend_vocab_zeros, c_t_1, coverage_0 = \
            get_input_from_batch(self.params, batch, device)
        # get encoder_output
        enc_outputs, enc_features, s_0 = self.model.encoder(enc_batch, enc_lens)

        enc_package = [s_0, c_t_1, coverage_0, enc_outputs, enc_features, enc_padding_mask, extend_vocab_zeros,
                       enc_batch_extended]

        if train_ml:
            ml_loss = self.infer_one_batch_ml(batch, *enc_package, iter, device=device)
        else:
            ml_loss = torch.tensor(0.0, dtype=torch.float, device=device)

        if train_rl:
            rl_loss, reward = self.infer_one_batch_rl(batch, *enc_package, iter, device=device)
        else:
            rl_loss = torch.tensor(0.0, dtype=torch.float, device=device)
            reward = torch.tensor(0.0, dtype=torch.float, device=device)

        rl_weight = getattr(self.params, "rl_weight", 0.0)
        loss = rl_weight * rl_loss + (1 - rl_weight) * ml_loss

        if not is_eval:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item(), reward.item()

    def infer_one_batch_ml(self, batch, s_0, c_t_1, coverage_0, enc_outputs, enc_features, enc_padding_mask,
                           extend_vocab_zeros, enc_batch_extended, iter, device):
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(self.params, batch, device)

        s_t_1 = s_0
        c_t_1 = c_t_1.unsqueeze(1)
        coverage_t = coverage_0

        teacher_forcing_ratio = getattr(self.params, "teacher_forcing_ratio", 1.0)
        teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        step_losses = []
        for di in range(min(max_dec_len, self.params.max_dec_steps)):
            if di == 0 or teacher_forcing:
                y_t_1 = dec_batch[:, di]
            else:
                y_t_1 = y_t
            # first we have coverage_t_1, then we have a_t
            final_dist, s_t_1, c_t_1, attn_dist, coverage_t_plus = self.model.decoder(y_t_1, s_t_1, c_t_1, enc_outputs,
                                        enc_features, enc_padding_mask, extend_vocab_zeros, enc_batch_extended, coverage_t)
            # if pointer_gen is True, the target will use the extend_vocab
            target = target_batch[:, di]
            # batch
            y_t = final_dist.max(1)[1]
            # batch x extend_vocab_size -> batch x 1 -> batch
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + self.params.eps)

            if self.params.is_coverage:
                # batch
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage_t), dim=1)
                step_loss = step_loss + self.params.cov_loss_wt * step_coverage_loss
                coverage_t = coverage_t_plus

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens_var

        loss = torch.mean(batch_avg_loss)

        return loss

    def infer_one_batch_rl(self, batch, s_0, c_t_1, coverage_0, enc_outputs, enc_features, enc_padding_mask,
                           extend_vocab_zeros, enc_batch_extended, iter, device):
        if self.params.is_coverage == True:
            raise ValueError("do not support training rl loss with coverage now")

        s_t_1 = s_0
        c_t_1 = c_t_1.unsqueeze(1)
        coverage_t = coverage_0

        # decode one batch
        decode_input = [batch, s_t_1, c_t_1, enc_outputs, enc_features, enc_padding_mask, extend_vocab_zeros,
                        enc_batch_extended, coverage_t, device]
        sample_seqs, rl_log_probs = self.decode_one_batch_rl(False, *decode_input)
        with torch.autograd.no_grad():
            baseline_seqs, _ = self.decode_one_batch_rl(True, *decode_input)

        sample_reward = reward_function(sample_seqs, batch.original_abstracts, device=device)
        baseline_reward = reward_function(baseline_seqs, batch.original_abstracts, device=device)
        rl_loss = -(sample_reward - baseline_reward) * rl_log_probs
        rl_loss = torch.mean(rl_loss)
        batch_reward = torch.mean(sample_reward)

        return rl_loss, batch_reward

    def decode_one_batch_rl(self, greedy, batch, s_t_1, c_t_1, enc_outputs, enc_features, enc_padding_mask,
                            extend_vocab_zeros, enc_batch_extended, coverage_t, device):
        # No teacher forcing for RL
        dec_batch, _, max_dec_len, dec_lens_var, target_batch = get_output_from_batch(self.params, batch, device)
        log_probs = []
        decode_ids = []
        # we create the dec_padding_mask at the runtime
        dec_padding_mask = []
        y_t = dec_batch[:, 0]
        mask_t = torch.ones(len(enc_outputs), dtype=torch.long, device=device)
        # there is at least one token in the decoded seqs, which is STOP_DECODING
        for di in range(min(max_dec_len, self.params.max_dec_steps)):
            y_t_1 = y_t
            # first we have coverage_t_1, then we have a_t
            final_dist, s_t_1, c_t_1, attn_dist, coverage_t_plus = self.model.decoder(y_t_1, s_t_1, c_t_1, enc_outputs,
                                                                                      enc_features, enc_padding_mask,
                                                                                      extend_vocab_zeros,
                                                                                      enc_batch_extended, coverage_t)
            if not greedy:
                # sampling
                multi_dist = Categorical(final_dist)
                y_t = multi_dist.sample()
                log_prob = multi_dist.log_prob(y_t)
                log_probs.append(log_prob)

                y_t = y_t.detach()
                dec_padding_mask.append(mask_t.detach().clone())
                mask_t[(mask_t == 1) + (y_t == self.end_id) == 2] = 0
            else:
                # baseline
                y_t = final_dist.max(1)[1]
                y_t = y_t.detach()

            decode_ids.append(y_t)
            # for next input
            is_oov = (y_t >= self.vocab.size()).long()
            y_t = (1 - is_oov) * y_t + is_oov * self.unk_id

        decode_ids = torch.stack(decode_ids, 1)

        if not greedy:
            dec_padding_mask = torch.stack(dec_padding_mask, 1).float()
            log_probs = torch.stack(log_probs, 1) * dec_padding_mask
            dec_lens = dec_padding_mask.sum(1)
            log_probs = log_probs.sum(1) / dec_lens
            if (dec_lens == 0).any():
                print("Decode lengths encounter zero!")
                print(dec_lens)

        decoded_seqs = []
        for i in range(len(enc_outputs)):
            dec_ids = decode_ids[i].cpu().numpy()
            article_oovs = batch.art_oovs[i]
            dec_words = data.outputids2decwords(dec_ids, self.vocab, article_oovs,
                                                self.params.pointer_gen)
            if len(dec_words) < 2:
                dec_seq = "xxx"
            else:
                dec_seq = " ".join(dec_words)
            decoded_seqs.append(dec_seq)

        return decoded_seqs, log_probs


class Train(Procedure):
    def __init__(self, params, model_file_path=None):
        super().__init__(params, is_eval=False)
        # wait for creating threads
        time.sleep(10)
        cur_time = int(time.time())
        if model_file_path is None:
            train_dir = os.path.join(self.params.model_root, 'train_%d' % (cur_time))
        else:
            # model_file_path is expected to be train_dir/model/model_name
            train_dir = os.path.dirname( os.path.dirname(os.path.abspath(model_file_path)) )
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # dump the params
        param_path = os.path.join(train_dir, 'params_{}.json'.format(cur_time))
        print("Dump hyper-parameters to {}.".format(param_path))
        params.save(param_path)

        self.model_file_path = model_file_path
        self.summary_writer = tf.summary.FileWriter(train_dir)
        self.summary_flush_interval = self.params.summary_flush_interval
        self.print_interval = self.params.print_interval
        self.save_interval = self.params.save_interval

    def _get_save_path(self, iter):
        cur_time = time.time()
        if self.params.is_coverage:
            prefix = 'coverage_model_{}_{}'
            param_prefix = 'coverage_params_{}_{}'
        else:
            prefix = 'model_{}_{}'
            param_prefix = 'params_{}_{}'

        if self.params.train_rl:
            prefix = 'rl_' + prefix
            param_prefix = 'rl_' + param_prefix
        model_save_path = os.path.join(self.model_dir, prefix.format(iter, cur_time))
        param_save_path = os.path.join(self.model_dir, param_prefix.format(iter, cur_time))
        return model_save_path, param_save_path

    def save_model(self, iter, running_avg_loss, model_save_path):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path):
        # check params
        rl_weight = getattr(self.params, "rl_weight", 0.0)
        if self.params.train_rl and rl_weight == 0.0:
            raise ValueError("Train RL is True, while rl_weight is 0.0. Contradiction!")

        self.model = PointerEncoderDecoder(self.params, model_file_path, pad_id=self.pad_id)
        initial_lr = self.params.lr if not self.params.is_coverage else self.params.lr_coverage
        optim_name = self.params.optim.lower()
        if optim_name == "adam":
            self.optimizer = Adam(self.model.parameters, lr=initial_lr)
        else:
            raise ValueError("Unknow optim {}".format(optim_name))

        start_iter, start_loss = 0, 0

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda  storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            train_rl = self.params.train_rl
            reoptim = self.params.reoptim
            if not train_rl and reoptim:
                raise ValueError("Not training rl but recreate the optimizer")

            # We need not to load the checkpoint if we use coverage to retrain
            if not self.params.is_coverage and not reoptim:
                print("Load the optimizer...")
                sys.stdout.flush()
                self.optimizer.load_state_dict(state['optimizer'])
                if utils.use_cuda(self.params.device):
                    device = torch.device(self.params.device)
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.to(device)

        return start_iter, start_loss

    def train_one_batch(self, batch, iter):
        return self.infer_one_batch(batch, iter, is_eval=False)

    def train(self, n_iters=None, eval=False):
        """
        :param n_iters: the iterations of training process
        :param model_file_path: the stored model file
        :return:
            do not return anything, but will print logs and store models
        """
        eval_processes = []
        if n_iters == None:
            n_iters = self.params.max_iterations
        iter, running_avg_loss = self.setup_train(self.model_file_path)
        start_iter = iter
        total_iter = n_iters - start_iter
        start = time.time()
        start_time = start
        print("start training.")
        sys.stdout.flush()
        loss_total = 0
        reward_total = 0

        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss, reward = self.train_one_batch(batch, iter)

            running_avg_loss = utils.calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            loss_total += loss
            reward_total += reward
            iter += 1

            if iter % self.summary_flush_interval == 0:
                self.summary_writer.flush()
            if iter % self.print_interval == 0:
                elapse, remain = utils.time_since(start_time, (iter - start_iter) / total_iter)
                iter_num = iter - start_iter
                print('Train steps %d, seconds for %d batch: %.2f , loss: %f, reward: %f, elapse: %s, remain: %s' %
                      (iter, self.print_interval,time.time() - start, loss_total/iter_num, reward_total/iter_num,
                       elapse, remain))
                sys.stdout.flush()
                start = time.time()
                if np.isnan(loss) or np.isnan(running_avg_loss):
                    raise ValueError("Loss becomes nan")

            if iter % self.save_interval == 0:
                model_save_path, param_save_path = self._get_save_path(iter)
                self.save_model(iter, running_avg_loss, model_save_path)
                self.params.save(param_save_path)

                if eval:
                    kwargs = {
                        "params": self.params,
                        "model_path": model_save_path,
                        "ngram_filter": False,
                        "data_file_prefix": "valid."
                    }
                    # p = mp.Process(target=PRSum.eval_raw, kwargs=kwargs)
                    # decode instead of evaluate
                    p = mp.Process(target=PRSum.decode_raw, kwargs=kwargs)
                    eval_processes.append(p)
                    p.start()

        for cur_p in eval_processes:
            cur_p.join()
        print("end training.")


class PRSum(object):
    @classmethod
    def train(cls, param_path, model_path=None, eval=False):
        """
        :param param_path: path of the params file
        :param model_path: path of the model to be loaded, None means train from scratch
        :param eval: whether to evaluate after saving
        """
        if model_path is not None:
            print("Try to resume from trained model {}".format(model_path))
        params = utils.Params(param_path)
        train_processor = Train(params, model_file_path=model_path)
        train_processor.train(eval=eval)

    @classmethod
    def decode(cls, param_path, model_path, ngram_filter, data_file_prefix="test."):
        params = utils.Params(param_path)
        cls.decode_raw(params, model_path, ngram_filter, data_file_prefix)

    @classmethod
    def decode_raw(cls, params, model_path, ngram_filter, data_file_prefix):
        decode_processor = BeamSearch(params, model_path, data_file_prefix=data_file_prefix, ngram_filter=ngram_filter)
        decode_processor.decode()

    @staticmethod
    def find_model_path(model_dir, model_name_pattern, iter, files):
        model_prefix = model_name_pattern.format(iter)
        names = []
        for file in files:
            if file.startswith(model_prefix):
                names.append(file)
        name = sorted(names)[-1]
        model_path = os.path.join(model_dir, name)
        return model_path

    @classmethod
    def select_model(cls, param_path, model_pattern, start_iter=3000, end_iter=36000):
        """
        :param param_path:
        :param model_pattern: model_{}_
        :param start_iter:
        :param end_iter:
        :return:
        """
        model_dir = os.path.dirname(model_pattern)
        model_name_pattern = os.path.basename(model_pattern)
        assert(os.path.isdir(model_dir))
        files = []
        for file in os.listdir(model_dir):
            if os.path.isfile(os.path.join(model_dir, file)):
                files.append(file)
        params = utils.Params(param_path)
        save_interval = params.save_interval
        l_f_scores = {}
        for iter in range(start_iter, end_iter, save_interval):
            model_path = cls.find_model_path(model_dir, model_name_pattern, iter, files)
            print("Param path {}".format(param_path))
            print("Model path {}".format(model_path))
            decode_processor = BeamSearch(params, model_path, data_file_prefix="valid.", ngram_filter=False)
            result_dict = decode_processor.decode()
            l_f_scores[iter] = result_dict['rouge_l_f_score']
        items = sorted(l_f_scores.items(), key=lambda x:x[1], reverse=True)
        output_file = os.path.join(model_dir, 'valid_decode_results.json')
        dump_json_file(output_file, items)
        print(items)

    def repair_missing_valid(cls, param_path, model_prefix, start_iter=1000, end_iter=36000):
        """
        :param param_path: models/train_xxx/valid.decode_model_,
        :param model_prefix:
        :param start_iter:
        :param end_iter:
        :return:
        """
        model_dir = os.path.dirname(model_prefix)
        model_name_prefix = os.path.basename(model_prefix)
        model_pattern = '_'.join(model_name_prefix.split('_')[1:]) + "{}_"
        assert (os.path.isdir(model_dir))
        files = []
        for file in os.listdir(model_dir):
            if os.path.isfile(os.path.join(model_dir, file)):
                files.append(file)
        iters, _, _ = find_valid_results(model_dir, model_name_prefix)

        params = utils.Params(param_path)
        save_iterval = params.save_interval
        all_iters = set(range(start_iter, end_iter, save_iterval))
        missing_iters = all_iters - set(iters)
        print("Missing Iters: {}".format(missing_iters))
        for iter in tqdm(missing_iters):
            model_path = cls.find_model_path(model_dir, model_pattern, iter, files)
            print("Param path {}".format(param_path))
            print("Model path {}".format(model_path))
            decode_processor = BeamSearch(params, model_path, data_file_prefix="valid.", ngram_filter=False)
            decode_processor.decode()
        print("Done!")

    @classmethod
    def collect_valid_results(cls, model_prefix, interval=1000, start_iter=1000, end_iter=26000, type="ml"):
        """
        :param model_prefix: like models/train_xxx/valid.decode_model_, the prefix of the dir storing the ROUGE results
        :param interval: 1000
        :param start_iter: 1000
        :param end_iter: 26000
        :return:
        """
        model_dir = os.path.dirname(model_prefix)
        model_name_prefix = os.path.basename(model_prefix)
        assert (os.path.isdir(model_dir))
        scores = {}
        iters, result_files, missing_iters = find_valid_results(model_dir, model_name_prefix)
        if missing_iters:
            raise FileNotFoundError("Missing iters {}".format(missing_iters))
        for iter, result_json_file in zip(iters, result_files):
            rouge_l_f = utils.load_json_file(result_json_file)['rouge_l_f_score']
            if iter in scores:
                raise ValueError("already found iteration {}".format(iter))
            scores[iter] = rouge_l_f
        sorted_iters = list(range(start_iter, end_iter, interval))
        sorted_keys = sorted(list(scores.keys()))
        if sorted_iters != sorted_keys:
            print("Miss some iterations!\n"
                  "Expect: {}\n"
                  "Get: {}".format(sorted_iters, sorted_keys))
            raise ValueError("Miss iterations")
        items = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        output_file = os.path.join(model_dir, type+'.valid_decode_results.json')
        dump_json_file(output_file, items)
        print(items)


def find_valid_results(model_dir, model_name_prefix):
    pattern = model_name_prefix + r'(\d+)_[\d.]+'
    iters = []
    result_files = []
    missing_iters = []
    for file in os.listdir(model_dir):
        match = re.match(pattern, file)
        if match:
            iter = int(match[1])
            result_json_file = os.path.join(model_dir, file, 'ROUGE_results.json')
            if not os.path.isfile(result_json_file):
                missing_iters.append(iter)
                continue
            iters.append(iter)
            result_files.append(result_json_file)
    return iters, result_files, missing_iters


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    fire.Fire(PRSum)
