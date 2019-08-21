# encoding=utf-8

import re
import os
import sys
import csv
import json
import math
import time
import torch
import tempfile
from myrouge.rouge import Rouge
from nltk import sent_tokenize
import subprocess as sp
from typing import List
import logging
from pyrouge import Rouge155
from pyrouge.utils import log
import tensorflow as tf

try:
    _ROUGE_PATH = os.environ['ROUGE']
except KeyError:
    print('Warning: ROUGE is not configured')
    _ROUGE_PATH = None

csv.field_size_limit(sys.maxsize)


class Params():
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def use_cuda(device):
    return device and device.startswith("cuda") and torch.cuda.is_available()


def load_json_file(path :str):
    with open(path, 'r') as f:
        obj = json.load(f)
    return obj


def dump_json_file(path :str, obj :object):
    with open(path, 'w') as f:
        json.dump(obj, f)


def load_csv_file(path :str, fieldnames :List = None) -> List[dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        if fieldnames is not None:
            _ = next(reader)
        for row in reader:
            rows.append(row)
    return rows


def dump_csv_file(path :str, obj : List[dict]) -> None:
    obj = list(obj)
    with open(path, 'w') as csvfile:
        out_field_names = list(obj[0].keys())
        writer = csv.DictWriter(csvfile, out_field_names, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(obj)


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
    if running_avg_loss == 0:  # on the first iteration just take the loss
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
    running_avg_loss = min(running_avg_loss, 12)  # clip
    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
    summary_writer.add_summary(loss_sum, step)
    return running_avg_loss


def print_results(article, abstract, decoded_output):
  print ("")
  print('ARTICLE:  %s', article)
  print('REFERENCE SUMMARY: %s', abstract)
  print('GENERATED SUMMARY: %s', decoded_output)
  print( "")


def make_html_safe(s):
  s = s.replace("<", "&lt;")
  s = s.replace(">", "&gt;")
  return s


def rouge_eval(ref_dir, dec_dir, dec_pattern='(\d+)_decoded.txt', ref_pattern='#ID#_reference.txt',
               cmd="-c 95 -r 1000 -n 2 -m", system_id=1):
    # only print rouge 1 2 L
    assert _ROUGE_PATH is not None
    log.get_global_console_logger().setLevel(logging.WARNING)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dec_dir = os.path.join(tmp_dir, 'dec')
        tmp_ref_dir = os.path.join(tmp_dir, 'ref')
        Rouge155.convert_summaries_to_rouge_format(
            dec_dir, tmp_dec_dir)
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, tmp_ref_dir)
        Rouge155.write_config_static(
            tmp_dec_dir, dec_pattern,
            tmp_ref_dir, ref_pattern,
            os.path.join(tmp_dir, 'settings.xml'),
            system_id
        )
        cmd = (os.path.join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
               + ' -e {} '.format(os.path.join(_ROUGE_PATH, 'data'))
               + cmd
               + ' -a {}'.format(os.path.join(tmp_dir, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output


def rouge_result_to_dict(rouge_result):
    return Rouge155().output_to_dict(rouge_result)


def rouge_log(results, dir_to_write):
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    print("Writing final ROUGE results to %s..." % (results_file))
    with open(results_file, "w") as f:
        f.write(results)


def write_for_rouge(reference, decoded_words, ex_index, _rouge_ref_dir, _rouge_dec_dir):
    """
    require un_sent_tokenize text, and will use ntlk to conduct sent_tokenize
    """
    decoded_abstract = " ".join(decoded_words)
    write_for_rouge_raw(reference, decoded_abstract, ex_index, _rouge_ref_dir, _rouge_dec_dir)


def replace_nl(text):
    return re.sub(r'\s*<nl>\s*', r'\n', text)


def get_ref_file(ref_dir, index):
    return os.path.join(ref_dir, "%06d_reference.txt" % index)


def get_dec_file(dec_dir, index):
    return os.path.join(dec_dir, "%06d_decoded.txt" % index)


def prepare_rouge_text(text):
    # replace <nl> to \n
    text = replace_nl(text)
    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    text = make_html_safe(text)
    sents = sent_tokenize(text)
    text = "\n".join(sents)
    return text


def write_for_rouge_raw(reference, decoded_abstract, ex_index, _rouge_ref_dir, _rouge_dec_dir):
    """
    require un_sent_tokenize text, and will use ntlk to conduct sent_tokenize
    """
    decoded_abstract = prepare_rouge_text(decoded_abstract)
    reference = prepare_rouge_text(reference)

    ref_file = get_ref_file(_rouge_ref_dir, ex_index)
    decoded_file = get_dec_file(_rouge_dec_dir, ex_index)

    with open(ref_file, "w") as f:
        f.write(reference)
    with open(decoded_file, "w") as f:
        f.write(decoded_abstract)
    # print("Wrote example %i to file" % ex_index)


def make_rouge_dir(decode_dir):
    rouge_ref_dir = os.path.join(decode_dir, 'rouge_ref')
    rouge_dec_dir = os.path.join(decode_dir, 'rouge_dec_dir')
    for p in [decode_dir, rouge_ref_dir, rouge_dec_dir]:
        if not os.path.exists(p):
            os.makedirs(p)
    return rouge_ref_dir, rouge_dec_dir


def try_load_state(model_file_path):
    counter = 0
    state = None
    while True:
        if counter >= 10:
            raise FileNotFoundError
        try:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
        except:
            time.sleep(30)
            counter += 1
            continue
        break
    return state


def as_minutes(s):
    m = s // 60
    s = math.ceil(s % 60)
    return "{}m {}s".format(m, s)


def time_since(since, percent):
    s = time.time() - since
    total = s / percent
    remain = total - s
    return as_minutes(s), as_minutes(remain)


def sentence_end(text):
    pattern = r'.*[.!?]$'
    if re.match(pattern, text, re.DOTALL):
        return True
    else:
        return False


def ext_art_preprocess(text):
    paras = text.split(' <para-sep> ')
    cms = paras[0].split(' <cm-sep> ')
    sents = cms + paras[1:]
    new_sents = []
    for s in sents:
        s = s.strip()
        # although we already add . when preprocessing
        if s:
            if not sentence_end(s):
                s = s + ' .'
            new_sents.append(s)
    return " ".join(new_sents)


def ext_art_sent_tokenize(text):
    art = ext_art_preprocess(text)
    art_sents = sent_tokenize(art)
    return art_sents


def ext_abs_sent_tokenize(text):
    return sent_tokenize(text)
